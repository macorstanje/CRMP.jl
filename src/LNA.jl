const ℝ{N} = SVector{N,Float64}
const ℤ{N} = SVector{N,Int64}

"""
Implementation
"""

function vectorise(V::T, z::K) where {T<:SMatrix, K<:SVector}
    vcat(ℝ{Size(V)[1]*Size(V)[2]}(V), ℝ{length(z)}(z))
end
function vectorise(V::T, z::K) where {T<:SMatrix,K<:Real}
    vcat(ℝ{Size(V)[1]*Size(V)[2]}(V), ℝ{1}(z))
end
function vectorise(V::T, z::K) where {T<:Real, K<:SVector}
    vcat(ℝ{1}(V), ℝ{length(z)}(z))
end
function vectorise(V::T, z::K) where {T<:Real, K<:Real}
    vcat(ℝ{1}(V), ℝ{1}(z))
end

function vectorise(ψ::T, G::T, z::K) where {T,K}
    m = typeof(ψ) <: Real ? 1 : Size(ψ)[1]
    d = typeof(z) <: Real ? 1 : length(z)
    vcat(ℝ{m*m}(ψ), ℝ{m*m}(G),  ℝ{d}(z))
end



# condition on endpoint for backward system for V and z
struct endpoint
    V
    z
end

# condition on endpoint for backward system for ψ, G and z (no restart)
struct endpoint_nR
    ψ
    G
    z
end

# Converts array u into d×d matrix V and d-vector z
function static_accessor_Vz(u::T, P::ChemicalReactionProcess) where {T}
    d = CRMP.getd(P) ; m = Int64(sqrt(length(u)-d))
    Vind = ℤ{m * m}(1:m*m)
    zind = ℤ{d}((m*m+1):(m*m+d))
    return (SMatrix{m,m,Float64}(reshape(u[Vind], Size(m, m))), SVector{d,Float64}(u[zind]))
end

function static_accessor_ψGz(u::T, P::ChemicalReactionProcess) where{T}
    d = getd(P) ; m = Int64(sqrt((length(u)-d)/2))
    ψind = ℤ{m * m}(1:m*m)
    Gind = ℤ{m * m}((m*m+1):2*m*m)
    zind = ℤ{d}((2*m*m+1):(2*m*m+d))
    return (SMatrix{m,m,Float64}(reshape(u[ψind], Size(m,m))), 
                SMatrix{m,m,Float64}(reshape(u[Gind], Size(m,m))),  
                SVector{d,Float64}(u[zind]))
end
# Drift function of the CLE
function α(t, x, P::ChemicalReactionProcess)
    sum([ℓ.λ(t, x) * ℓ.ξ for ℓ in reaction_array(P)])
end

# squared diffusion coefficient of CLE
function β(t, x, P::ChemicalReactionProcess)
    sum([ℓ.λ(t, x) * ℓ.ξ * ℓ.ξ' for ℓ in reaction_array(P)])
end

function F(t, z, P::ChemicalReactionProcess)
    if CRMP.getd(P) == 1 
        return ForwardDiff.derivative(x -> α(t,x,P),z[1])
    else
        return ForwardDiff.jacobian(x -> α(t, x, P), z)
    end
end

function dVz(t, x, P::ChemicalReactionProcess)
    V, z = static_accessor_Vz(x, P)
    _α, _β, _F = α(t, z, P), β(t, z, P), F(t, z, P)
    if getd(P) == 1
        dV = SMatrix{1,1,Float64}(V[1,1] * _F' + _β[1] + _F * V[1,1])
        dz = SVector{1,Float64}(_α)
    else
        dV::typeof(V) = V * _F' + _β + _F * V
        dz::typeof(z) = _α
    end
    return vectorise(dV, dz)
end
dVz_DE(x, P, t) = dVz(t, x, P)

function dψGz(t,x,P::ChemicalReactionProcess)
    ψ, G, z = static_accessor_ψGz(x,P)
    _α, _β, _F, = α(t,z,P), β(t,z,P), F(t,z,P)

    if getd(P) == 1
        dψ = SMatrix{1,1,Float64}(_β[1]/(G[1,1]^2))
        dG = SMatrix{1,1,Float64}(_F*G[1,1])
        dz = SVector{1,Float64}(_α)
    else
        dψ::typeof(ψ) = inv(G)*_β*inv(G)'
        dG::typeof(G) = _F*G
        dz::typeof(z) = _α
    end
    return vectorise(dψ, dG, dz)   
end
dψGz_DE(x,P,t) = dψGz(t,x,P)

# Solves ODE backward with condition at time T on a grid tt
function ode_Vz_backward!(method, P::ChemicalReactionProcess, tt, (Vt, zt), hT::endpoint)
    saved_values = SavedValues(Float64, Tuple{typeof(hT.V),typeof(hT.z)})
    yT = vectorise(hT.V, hT.z)

    prob = ODEProblem(
        dVz_DE,                         # Increment
        yT,                             # initial condition
        (tt[end], tt[1]),               # timespan (reversed)
        P                               # Parameter
    )

    callback = SavingCallback(
        (u, t, integrator) -> staticaccessor_Vz(u, P),
        saved_values;
        saveat=reverse(tt),
        tdir=-1
    )

    integrator = init(prob, method, callback=callback, save_everystep=false,reltol=1e-6,)

    DifferentialEquations.solve!(integrator)

    ss = saved_values.saveval  # these are in reversed order, for Ht and Ft we need to reverse
    for i in eachindex(ss)
        Vt[end-i+1] = ss[i][1]
        zt[end-i+1] = ss[i][2]
    end
    Vt, zt
end

# Solves ODE forward with condition at time t on a grid tt, saves only final value
function ode_Vz_forward!(method, P::ChemicalReactionProcess, tt, (Vt, zt), hT::endpoint)
    saved_values = SavedValues(Float64, Tuple{typeof(hT.V),typeof(hT.z)})
    yT = vectorise(hT.V, hT.z)

    prob = ODEProblem(
        dVz_DE,                         # Increment
        yT,                             # initial condition
        (tt[1], tt[end]),               # timespan (not reversed)
        P                               # Parameter
    )

    callback = SavingCallback(
        (u, t, integrator) -> static_accessor_Vz(u, P),
        saved_values;
        saveat=ℝ{2}(tt[1], tt[end]),
        tdir=1
    )
    integrator = init(prob, method, callback=callback, save_everystep=false, reltol=1e-6,)
    DifferentialEquations.solve!(integrator)

    ss = saved_values.saveval  # these are in reversed order, so C is obtained from the last index and for Ht and Ft we need to reverse
    for i in eachindex(ss)
        Vt[i] = ss[i][1]
        zt[i] = ss[i][2]
    end
    Vt, zt
end

function ode_ψGz_forward!(method, P::ChemicalReactionProcess, tt, (ψt, Gt, zt), hT::endpoint_nR)
    saved_values = SavedValues(Float64, Tuple{typeof(hT.ψ), typeof(hT.G), typeof(hT.z)})
    yT = vectorise(hT.ψ, hT.G, hT.z)

    prob = ODEProblem(
        dψGz_DE,
        yT,
        (tt[1], tt[end]),
        P
    )

    callback = SavingCallback(
        (u,t,integrator) -> static_accessor_ψGz(u,P),
        saved_values;
        saveat = tt,
        tdir = 1
    )
    integrator = init(prob, method, callback=callback, save_everystep = true, reltol = 1e-6,)
    DifferentialEquations.solve(integrator)

    ss = saved_values.saveval
    for i in eachindex(ss)
        ψt[i] = ss[i][1]
        Gt[i] = ss[i][2]
        zt[i] = ss[i][3]
    end
    ψt, Gt, zt
end

abstract type LNA <: Guided_Process end
# abstract type pois  end


struct LNAR <: LNA # with restart
    obs::partial_observation
    P::ChemicalReactionProcess
end

struct LNAR_death <: LNA # with restart, specifically for death process
    obs::partial_observation
    P::ChemicalReactionProcess 
end

mutable struct LNA_nR <: LNA # no restart
    obs::partial_observation
    tt
    P::ChemicalReactionProcess 
    ψt 
    Gt 
    zt
    LNA_nR(obs, tt, P) = new(obs, tt, P) 
end

struct diff_death <: LNA # CHANGE, WORKS FINE FOR NOW 
    obs::partial_observation
    a::Float64 # diffusion coefficient of auxiliary process
    P::ChemicalReactionProcess
end

abstract type pois <: Guided_Process end
struct pois_death <: pois
    obs::partial_observation_poisson
    P::ChemicalReactionProcess
end

# getL(P::T) where {T<:Union{LNA, pois}} = P.L
# getv(P::T) where {T<:Union{LNA, pois}} = P.v
# gett(P::T) where {T<:Union{LNA, pois}} = P.T
getC(P::T) where {T<:Union{LNA,pois}} = getm(P) == 1 ? getϵ(P.obs) : SMatrix{getm(P), getm(P), Float64}(getϵ(P)*I)
# getm(P::T) where {T<:Union{LNA, pois}} = CRMP.getd(P.P) == 1 ? 1 : Size(L)[1]
# getd(P::T) where {T<:Union{LNA, pois}} = CRMP.getd(P.P)



"""
    guiding terms
"""
function logp(t, x, type::LNAR_death)
    v, T, c = getv(type), gett(type), type.P.ℛ.λ(1.0,1)
    ect = exp(-c*(T-t))
    return logpdf(Normal(x*ect, sqrt( x*ect*(1-ect))), v)
end

function logp(t,x,type::diff_death)
    v, T, a, ϵ = getv(type), gett(type), type.a, getC(type)
    return -(v-x)^2/( 2*a*(T-t+ϵ) )
end

function logp(t,x,type::pois_death)
    v,T,θ = getv(type), gett(type), getC(type)
    if v > x 
        return -Inf
    else
        return v == x ? 0.0 : logpdf(Poisson(θ*(T-t)), x-v)
    end
end

function logp(t, x, type::LNAR)
    L, v, T, C, P, d = getL(type), getv(type), gett(type), getC(type), type.P, getd(type.P)

    z = ℝ{d}(x)
    V = SMatrix{d,d,Float64}(zeros(d, d))
    zt = [z, z]
    Vt = [V, V]
    Δ = (T - t) / 100
    ode_Vz_forward!(Vern7(), P, LinRange(t, T, 100), (Vt, zt), endpoint(V, z)) #endpoint is confusing, is actualy the starting contition at time t
    zT, VT = zt[end], Vt[end]
    # println("t = $t") ; println("x = $x") ; println("zT = $zT") ; println("VT = $VT")
    if d == 1 
        return logpdf(Normal(L * zT[1], max(0., VT[1,1] * L^2 + C)), v)
    else
        # println("t = $t, x = $x, VT = $VT")
        return logpdf(MvNormal(L * zT, 0.5*(L * VT * L' + C + transpose(L * VT * L' + C))), v)  # matrix could become non-hermitian due to numerical error
    end
end

function fill_grid!(type::LNA_nR, x₀)
    L, v, T, C, P, d = getL(type), getv(type), gett(type), getC(type), type.P, getd(type.P)
    ψ = SMatrix{d,d,Float64}(zeros(d,d)) ; G = SMatrix{d,d,Float64}(I) ; z = ℝ{d}(x₀)
    tt = type.tt
    ψt = [ψ for i in eachindex(tt)]
    Gt = [G for i in eachindex(tt)]
    zt = [z for i in eachindex(tt)]
    ode_ψGz_forward!(Vern7(), P, tt, (ψt, Gt, zt), endpoint_nR(ψ, G, z))
    type.ψt = ψt
    type.Gt = Gt
    type.zt = zt
    type
end

function logp(t, x, type::LNA_nR)
    L, v, C, d = getL(type), getv(type), gett(type), getC(type), type.P, getd(type.P)
    ψt, Gt, zt = type.ψt, type.Gt, type.zt
    
    ind = findmin(abs(tt .- t))[2]
    GTt = Gt[end]*inv(Gt[ind])
    ψTt = Gt[ind]*(ψt[end]-ψt[ind])*Gt[ind]'
    
    if d == 1
        return logpdf(Normal(L*(zt[end] + GTt*(x-zt[ind])) , sqrt( L*GTt*ψTt*GTt'*L' + C) ) , v)
    else
        return  logpdf(MvNormal(L*(zt[end] + GTt*(x-zt[ind])) , 0.5*(L*GTt*ψTt*GTt'*L' + C  + transpose( L*GTt*ψTt*GTt'*L' + C))) , v)
    end
end
p(t, x, type::T) where {T<:Union{LNA, pois}} = exp(logp(t,x,type))

log_guiding_term(info, type::TP) where {TP<:LNA} = (ℓ,t,x) -> logp(t,x+ℓ.ξ,type) - logp(t,x,type)

function log_guiding_term(info, type::pois_death) 
    function fun(ℓ,t,x)  
        if getv(type) > x 
            return 0.0
        elseif getv(type) == x 
            return -Inf
        else
            return log( (x-getv(type.obs))/(getC(type)*(gett(type.obs)-t)) )
        end
    end
    return fun
end

guiding_term(info, type::TP) where {TP<:LNA} = (ℓ, t, x) -> exp(log_guiding_term(info, type)(ℓ,t,x,))
guiding_term(info, type::pois_death) = (ℓ,t,x) -> x == getv(type.obs) ? 0.0 : (x-getv(type.obs))/(getC(type)*(gett(type.obs)-t))


"""
    Forward simulation

functions to get the next reaction time cf. section 6 of the paper and to simulate the forward guided process using the δ-method. 
"""

# import CRMP.gettime
# function gettime(::increasing_rate, ℓ::reaction{T}, t, x::Union{T, Array{T,1}}, type::TP, setδ::Function) where {T<:Real, TP<:Union{LNA, pois}}
#     accepted = false
#     t_start = t
#     t₀ = t
#     if abs(ℓ.λ(t,x)*guiding_term(type)(ℓ, t,x) ) < 1e-8
#         return 1e10
#     else
#         τ = 1e10
#         counter = 0
#         while !accepted
#             δ = setδ(ℓ, t₀, x)     # (T-t₀)/2 or a smarter choice               # work on (t₀, t₀+δ)
#             logλ̄ = log(ℓ.λ(t₀+δ,x))+log_guiding_term(type)(ℓ,t₀+δ,x)            # upper bound
#             τ = -log(rand())/exp(logλ̄)                                          # proposal
#             if t+τ <= t₀ + δ # The proposed time must lie in (t₀,t₀+δ))
#                 logacc = log(ℓ.λ(t+τ,x)) + log_guiding_term(type)(ℓ,t+τ,x) - logλ̄
#                 accepted = log(rand()) <= logacc
#                 t += τ
#             else
#                 t₀ += δ
#                 t = t₀
#             end
#             counter = counter + 1
#             if counter > 1e4
#                 return 1e10
#                 counter = 0
#             end
#         end
#         return t - t_start
#     end
# end



function simulate_forward_monotone(x₀, type::TP, info) where {TP<:Union{LNA, pois}}
    ℛ = reaction_array(type.P)
    T = gett(type)
    t, x = 0.0, x₀
    tt, xx = [t], [x]

    diff(ℓ, t, x) = FiniteDifferences.central_fdm(2, 1)(s -> log_guiding_term(info, type)(ℓ,s,x) , t)
    # diff(ℓ, t, x) = central_fdm(5,1)(s -> log_guiding_term(type)(ℓ,s,x), t)
    while t < T
        Δ = [10e6 for ℓ in ℛ] # initialized reaction times for all reactions
        for (i, ℓ) in enumerate(ℛ)
            if ℓ.λ(t, x) > 0
                if diff(ℓ,t,x) > 0 # ℓ is a reaction that takes X closer to v
                    Δ[i] = gettime(increasing_rate(), ℓ, t, x, info, type, (ℓ,t,x) -> (T-t)/2)
                else
                    Δ[i] = gettime(decreasing_rate(), ℓ, t, x, info, type)
                end
            end
        end
        dt, μ = findmin(Δ)
        t = t + dt
        x = x + ℛ[μ].ξ
        push!(xx, x)
        push!(tt, t)
    end
    tt = vcat(tt[1:end-1], T)
    xx[end] = xx[end-1]
    return tt, xx
end



"""
    Likelihood computations of a given path. 

Computation of the term h̃(0,x₀)*Ψ(tt, xx), or the log of that term. 
"""
# finds ℓ such that ℓ.ξ = y-x
function find_reaction(x, y, P::ChemicalReactionProcess)
    ℛ = reaction_array(P)
    return ℛ[findall(ℓ -> ℓ.ξ == y-x, ℛ)][1]
end

# Expression from Sherlock and Golightly. Should result in the same likelihood, this is just a check. 
function loglikelihood_SG(tt,xx,GP::TP) where {TP<:Union{LNA, pois}}
    T  = gett(GP) ; ℛ = reaction_array(GP.P)

    if !iscorrect_1obs(GP, tt, xx)
        return -Inf # likelihood 0 if the conditioning is not satisfied
    end

    out = 0.0
    for i in 1:length(tt)-1
        t, x = tt[i], xx[i]
        if x != xx[i+1]
            ℓ = find_reaction(x, xx[i+1], GP.P)
            ℓᵒ = condition_reaction(ℓ, guiding_term(GP))
            out += log(ℓ.λ(tt[i+1],x)/ℓᵒ.λ(tt[i+1],x))
        end
        ℒh̃(s) = sum([ ℓ.λ(s,x)*(guiding_term(GP)(ℓ, s,x) - 1) for ℓ in ℛ ]) #ℒh̃/h̃
        # out += Integrals.solve(IntegralProblem(ℒh̃, tt[i], tt[i+1]) , HCubatureJL() ; reltol = 1e-3, abstol = 1e-3).u
        out += ℒh̃(tt[i])*(tt[i+1]-tt[i])
    end
    # out -= logp(T,xx[end], GP) # -0.5*H*xx[end]^2+ F*xx[end] # Why did I add this?
    return out
end

# exact log-likelihood computation using only the guiding term of the process
# function loglikelihood_direct(tt,xx,GP::TP) where {TP<:Union{LNA, pois}}
#     ℛ = reaction_array(GP.P) ; N = length(tt)
#     out = 0.0 #Not logp(tt[1],xx[1], GP) # log h(0,x₀) included in summation

#     if !iscorrect_1obs(GP, tt, xx)
#         return -Inf         # likelihood 0 if conditioning is not satisfied. 
#     end

#     integrand(x) = (s,p) -> sum([ ℓ.λ(s,x)*(guiding_term(GP)(ℓ, s, x) - 1.0) for ℓ in ℛ ])
#     for j in 1:N-2
#         ℓj = find_reaction(xx[j], xx[j+1], GP.P)
#         out -= log_guiding_term(GP)(ℓj, tt[j+1], xx[j])
#         out += tt[j] == tt[j+1] ? 0.0 : Integrals.solve(IntegralProblem(integrand(xx[j]), tt[j], tt[j+1]) , HCubatureJL() ; reltol = 1e-5, abstol = 1e-5).u
#         # out += quadgk(integrand(xx[j]),tt[j],tt[j+1], rtol = 1e-3)[1]
#         # out += integrand(xx[j])(tt[j], 1.0)*(tt[j+1]-tt[j])
#     end
#     out += Integrals.solve(IntegralProblem(integrand(xx[N-1]), tt[N-1], tt[N]) , HCubatureJL() ; reltol = 1e-5, abstol = 1e-5).u
#     # out += integrand(xx[N-1])(tt[N-1], 1.0)*(tt[N]-tt[N-1])
#     # out += quadgk(integrand(xx[N-1]),tt[N-1],tt[N], rtol = 1e-3)[1]
#     return out
# end


