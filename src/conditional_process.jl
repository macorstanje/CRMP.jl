"""
    partial_observation{T::Union{Real, AbstractArray{Real, 1}}}

A partial observation ``v~N(LX(t), ϵ*L\\tilde{a}L')`` at time `t` with an ``m \\times d``-matrix `L` and ``m \\ times m`` matrix `C`.
Use

```julia-repl
julia> obs = partial_observation(1.0, [2, 5], [1 0 0 ; 0 1 0], 0.05)
julia> gett(obs) # returns 1.0
julia> getv(obs) # returns [2, 5]
julia> getL(obs) # returns [1 0 0 ; 0 1 0]
julia> getϵ(obs) # returns 0.05
julia> getm(obs) # returns 2
julia> getd(obs) # returns 3
```
"""
struct partial_observation{T}
    t::Float64
    v::T
    L
    ϵ
end
gett(obs::partial_observation{T}) where {T} = obs.t
getv(obs::partial_observation{T}) where {T} = obs.v
getL(obs::partial_observation{T}) where {T} = obs.L
getϵ(obs::partial_observation{T}) where {T} = obs.ϵ
getm(obs::partial_observation{T}) where {T} = length(getv(obs))
getd(obs::partial_observation{T}) where {T} = typeof(obs.L) <: Union{Matrix, SMatrix} ? size(obs.L)[2] : 1


"""
    partial_observation_poisson{T::Union{Real, AbstractArray{Real, 1}}}

A partial observation for a process with a monotone component 
Assume an observation `v = LX(t)` with `L`` of the form `[L₁ 0 ; 0 L₂]`. 
`L₂` for the monotone component, 
Use:

```julia-repl
julia> obs = partial_observation_poisson(1.0, [2, 5], [ [1 0], [1] ],  0.05)
julia> gett(obs) # returns 1.0
julia> getv(obs) # returns [2, 5]
julia> getL(obs) # returns [1 0 0 ; 0 0 1]
julia> getL₁(obs)# returns [1 0] 
julia> getL₂(obs)# returns [1]
julia> getϵ(obs) # returns 0.05
julia> getm(obs) # returns 2
julia> getd(obs) # returns 3

```
"""
struct partial_observation_poisson{T}
    t::Float64
    v::T
    L
    ϵ
end
gett(obs::partial_observation_poisson{T}) where {T} = obs.t
getv(obs::partial_observation_poisson{T}) where {T} = obs.v
getL₁(obs::partial_observation_poisson{T}) where {T} = obs.L[1]
getL₂(obs::partial_observation_poisson{T}) where {T} = obs.L[2]

function getL(obs::partial_observation_poisson{T}) where {T} 
    L₁ = getL₁(obs) ; L₂ = getL₂(obs)
    if isequal(L₁, nothing)
        L = L₂
    elseif isequal(L₂, nothing)
        L = L₁
    else
        m1 , Z = size(L₁) ; m2, Y = size(L₂)
        L = zeros(m1+m2 , Z+Y)
        for i in 1:m1
            for j in 1:Z
                L[i,j] = L₁[i,j]
            end
        end
        for i in 1:m2
            for j in 1:Y
                L[m1+i , Z+j] = L₂[i,j]
            end
        end
    end
    return L     
end

getϵ(obs::partial_observation_poisson{T}) where {T} = obs.ϵ
getm(obs::partial_observation_poisson{T}) where {T} = length(getv(obs))
function getd(obs::partial_observation_poisson{T}) where {T}
    if isequal(getL₁(obs), nothing)
        Z = 0
    else
        Z = typeof(getL₁(obs)) <: Real ? 1 : size(getL₁(obs))[2]
    end
    if isequal(getL₂(obs), nothing)
        Y = 0
    else
        Y = typeof(getL₂(obs)) <: Real ? 1 : size(getL₂(obs))[2]
    end
    return Y+Z
end

abstract type Guided_Process end
"""
    diffusion_guiding_term{T}

Guided process through array of partial observations `obs`, with an array ``d \\times d``-matrices stored in 
`a` and an original `ChemicalReactionProcess` `P`. 

```julia-repl
# Guided GTT process for one partial observation at time 1.0 using a=I
julia> partial_obs = [partial_observation(1.0, [2, 5], [1 0 0 ; 0 1 0], 0.05)]
julia> GP = diffusion_guiding_term( partial_obs , [ [1 0 0 ; 0 1 0 ; 0 0 1] ] , GTT(κ₁,κ₂,dₚ,dₘ) )
```
"""
struct diffusion_guiding_term{T} <: Guided_Process
    obs::Union{partial_observation{T}, Array{partial_observation{T},1}}
    a::Union{Matrix{Float64}, Array{Matrix{Float64},1}, Float64, Array{Float64,1}}
    P::ChemicalReactionProcess
end

"""
    poisson_guiding_term{T}

Guided process through array of partial observations `obs`, with an array ``d \\times d``-matrices stored in 
`a`, an integer `Y` that represents the amount of poisson terms and an original `ChemicalReactionProcess` `P`. 

```julia-repl
# Guided GTT process for one partial observation at time 1.0 using a=I
julia> partial_obs = [partial_observation(1.0, [2, 5], [1 0 0 ; 0 1 0], 0.05)]
julia> GP = Guided_Process( partial_obs , [ [1 0 0 ; 0 1 0 ; 0 0 1] ] , 0, GTT(κ₁,κ₂,dₚ,dₘ) )
```
"""
struct poisson_guiding_term{T} <: Guided_Process
    obs::Union{partial_observation_poisson{T}, Array{partial_observation_poisson{T},1}}
    a::Union{Matrix{Float64}, Array{Matrix{Float64},1}, Float64, Array{Float64,1}}
    P::ChemicalReactionProcess
end
gett(GP::T) where {T<:Guided_Process} = getn(GP) == 1 ? gett(GP.obs) : map(o -> gett(o), GP.obs)
getv(GP::T) where {T<:Guided_Process} = getn(GP) == 1 ? getv(GP.obs) : map(o -> getv(o), GP.obs)
getn(GP::T) where {T<:Guided_Process} = 1
getn(GP::diffusion_guiding_term) = typeof(GP.obs) <: partial_observation ? 1 : length(GP.obs)
getn(GP::poisson_guiding_term) = typeof(GP.obs) <: partial_observation_poisson ? 1 : length(GP.obs)
getL₁(GP::poisson_guiding_term) = getn(GP) == 1 ? getL₁(GP.obs) : map(o -> getL₁(o), GP.obs)
getL₂(GP::poisson_guiding_term) = getn(GP) == 1 ? getL₂(GP.obs) : map(o -> getL₂(o), GP.obs)
getL(GP::T) where {T<:Guided_Process} = getn(GP) == 1 ? getL(GP.obs) : map(o -> getL(o), GP.obs)
getϵ(GP::T) where {T<:Guided_Process} = getn(GP) == 1 ? getϵ(GP.obs) : map(o -> getϵ(o), GP.obs)
getm(GP::T) where {T<:Guided_Process} = getn(GP) == 1 ? getm(GP.obs) : map(o -> getm(o), GP.obs)
getd(GP::T) where {T<:Guided_Process} = getn(GP) == 1 ? getd(GP.obs) : getd(GP.obs[1])
geta(GP::diffusion_guiding_term) = GP.a
# Amount of poisson terms is the difference between d and the dimensions of a. Poisson terms are assumed 
# to be the the last terms of the guided process
function poisson_terms(GP::poisson_guiding_term) 
    if getn(GP) == 1
        return typeof(getL₂(GP.obs)) <: Real ? 1 : size(getL₂(GP.obs))[2]
    else
        return typeof(getL₂(GP.obs[1])) <: Real ? 1 : size(getL₂(GP.obs[1]))[2]
    end
end

function geta(GP::poisson_guiding_term)
    if poisson_terms(GP) == getd(GP)
        return nothing
    else
        if getn(GP) == 1
            return GP.a[1:getd(GP)-poisson_terms(GP) , 1:getd(GP)-poisson_terms(GP)]
        else
            return map(A -> A[1:getd(GP)-poisson_terms(GP) , 1:getd(GP)-poisson_terms(GP)], GP.a)
        end
    end
end

function getθ(GP::poisson_guiding_term)
    if poisson_terms(GP) == 0
        return nothing
    end
    if getn(GP) == 1
        return typeof(GP.a) <: Real ? GP.a : GP.a[getd(GP)-poisson_terms(GP)+1:getd(GP) , getd(GP)-poisson_terms(GP)+1:getd(GP)]
    else
        return map(A -> typeof(A) <: Real ? A : A[getd(GP)-poisson_terms(GP)+1:getd(GP) , getd(GP)-poisson_terms(GP)+1:getd(GP)], GP.a)
    end
end

# Amount of poisson terms is the difference between d and the dimensions of a. Poisson terms are assumed 
# to be the the last terms of the guided process
# Can be changed

"""
    getk(t, times)

Given an array of times `times` and a time `t`, returns `k` such that `t` lies in `[times[k-1],times[k])`
"""
function getk(times::Array{T,1}, t::T) where {T<:Real}
    k = searchsortedfirst(times, t)
    if times[k] == t 
        k += 1
    end
    return k
end #  t in [times[k-1], times[k])
getk(times::T, t::T) where {T<:Real} = 1 # return 1 when there is just 1 observation

function get_log_upper_bound_1obs_1dim(ℓ,t,x,GP::diffusion_guiding_term)
    v, ϵ, a, tₖ = getv(GP.obs), getϵ(GP.obs), GP.a, gett(GP.obs)
    λ = a
    return  0.5*(v-x)^2/(ϵ*λ) - 0.5*(v-x-ℓ.ξ)^2/((tₖ+ϵ)*λ)
end

function get_log_upper_bound_1obs(ℓ,t,x,GP::diffusion_guiding_term)
    v, L, ϵ, a, tₖ = getv(GP.obs), getL(GP.obs), getϵ(GP.obs), GP.a, gett(GP.obs)
    λmin, λmax = eigmin(L*a*L'), eigmax(L*a*L')
    return  0.5*dot(v-L*x,v-L*x)/(ϵ*λmin) - 0.5*dot(v-L*(x+ℓ.ξ), v-L*(x+ℓ.ξ))/((tₖ+ϵ)*λmax)
end

function get_log_upper_bound_1dim(ℓ,t,x,GP::diffusion_guiding_term)
    k = getk(gett(GP), t)
    v, ϵ, a, tₖ = getv(GP.obs[k]), getϵ(GP.obs[k]), GP.a[k], gett(GP.obs[k])
    λ = a
    return  0.5*(v-x)^2/(ϵ*λ) - 0.5*(v-x-ℓ.ξ)^2/((tₖ+ϵ)*λ)
end

function get_log_upper_bound(ℓ, t, x, GP::diffusion_guiding_term)
    if getn(GP) == 1 
        if getd(GP) == 1
            return get_log_upper_bound_1obs_1dim(ℓ,t,x,GP)
        else
            return get_log_upper_bound_1obs(ℓ,t,x,GP)
        end
    else
        if getd(GP) == 1
            return get_log_upper_bound_1dim(ℓ,t,x,GP)
        else
            k = getk(gett(GP), t)
            v, L, ϵ, a, tₖ = getv(GP.obs[k]), getL(GP.obs[k]), getϵ(GP.obs[k]), GP.a[k], gett(GP.obs[k])
            λmin, λmax = eigmin(L*a*L'), eigmax(L*a*L')
            return  0.5*dot(v-L*x,v-L*x)/(ϵ*λmin) - 0.5*dot(v-L*(x+ℓ.ξ), v-L*(x+ℓ.ξ))/((tₖ+ϵ)*λmax)
        end
    end
end

function get_log_upper_bound(t,x, GP::diffusion_guiding_term)
    return [get_log_upper_bound(ℓ,t,x,GP) for ℓ in GP.P.ℛ]
end

function get_upper_bound(ℓ, t, x, GP::diffusion_guiding_term)
    return exp(get_log_upper_bound(ℓ, t, x, GP))
end

function get_upper_bound(t, x, GP::diffusion_guiding_term)
    return [exp(get_log_upper_bound(ℓ,t,x,GP)) for ℓ in GP.P.ℛ]
end


"""
    filter_backward(GP::Guided_Process)

Returns the quadruple `(H,F, LaL⁻¹, LC⁻¹)`. All are arrays of size `n` that containns 
``H(tₖ)``, ``F(tₖ)``, ``(Lₖ ãₖ Lₖ)⁻¹`` and ``L'( ϵ Lₖ ãₖ Lₖ)⁻¹  k=1,\\dots,n``.
Note, we return ``H̄(tₖ)`` and ``F̄(tₖ)``
"""
function filter_backward(GP::diffusion_guiding_term)
    if getd(GP) == 1
        return filter_backward_1dim(GP)
    else
        obs = GP.obs
        n = getn(GP) ; times = gett(GP) ; d = getd(GP) ; a = GP.a
        if n == 1
            L, ϵ, v = getL(obs) , getϵ(obs), getv(obs)
            LaL⁻¹ = inv(L*a*L') ; LC⁻¹ = L'*LaL⁻¹/ϵ  ; H = LC⁻¹*L ; F =  LC⁻¹*v
        else
            L, ϵ, v = getL(obs[n]) , getϵ(obs[n]), getv(obs[n])
            H = [zeros(d,d) for k in 1:n] ; F = [zeros(d) for k in 1:n]
            LaL⁻¹ = [zeros(getm(o), getm(o)) for o in obs] ; LC⁻¹ = [zeros(d,getm(o)) for o in obs]
            LaL⁻¹[n] = inv(L*a[n]*L')
            LC⁻¹[n] = L'*LaL⁻¹[n]/ϵ ; H[n] = LC⁻¹[n]*L ; F[n] =  LC⁻¹[n]*v
            for i in 2:n
                k=n-i+1
                L, ϵ, v  = getL(obs[k]) , getϵ(obs[k]), getv(obs[k])
                LaL⁻¹[k] = inv(L*a[k]*L') ; LC⁻¹[k] = L'*LaL⁻¹[k]/ϵ
                z = inv( Matrix{Float64}(I,d,d) + H[k+1]*a[k+1]*(times[k+1]-times[k]) ) # z_{k+1}(t_k)
                H[k] = z*H[k+1]+LC⁻¹[k]*L # First H(t_k+) then L'C⁻¹L
                F[k] = z*F[k+1]+LC⁻¹[k]*v # First F(t_k+) then L'C⁻¹v
            end
        end
    end
    return (H, F, LaL⁻¹ , LC⁻¹)
end

function filter_backward_1dim(GP::diffusion_guiding_term)
    obs = GP.obs
    n = getn(GP) ; times = gett(GP) ; a = GP.a
    if n == 1
        ϵ, v = getϵ(obs), getv(obs)
        LaL⁻¹ = 1/a ; LC⁻¹ = LaL⁻¹/ϵ  ; H = LC⁻¹ ; F =  LC⁻¹*v
    else
        H = zeros(n) ; F = zeros(n) ;  LaL⁻¹ = zeros(n) ; LC⁻¹ = zeros(n)
        ϵ, v = getϵ(obs[n]), getv(obs[n])
        LaL⁻¹[n] = 1/a[n] ; LC⁻¹[n] = LaL⁻¹[n]/ϵ ; H[n] = LC⁻¹[n] ; F[n] =  LC⁻¹[n]*v
        for i in 2:n
            k=n-i+1
            ϵ, v  = getϵ(obs[k]), getv(obs[k])
            LaL⁻¹[k] = 1/a[k] ; LC⁻¹[k] = LaL⁻¹[k]/ϵ
            z = 1/( 1.0 + H[k+1]*a[k+1]*(times[k+1]-times[k]) ) # z_{k+1}(t_k)
            H[k] = z*H[k+1]+LC⁻¹[k] # First H(t_k+) then L'C⁻¹L
            F[k] = z*F[k+1]+LC⁻¹[k]*v # First F(t_k+) then L'C⁻¹v
        end
    end
    return (H, F, LaL⁻¹ , LC⁻¹)
end

function GP(GPP::poisson_guiding_term)
    a = geta(GPP) ; m = getm(GPP) ; obs = GPP.obs 
    if getn(GPP) == 1
        @assert !isequal(getL₁(obs) , nothing) "Brownian term is never observed"
        _obs = partial_observation(gett(obs) , getv(obs)[1:size(getL₁(obs))[1]] , getL₁(obs), obs.ϵ)
        return diffusion_guiding_term(_obs, a,GPP.P)
    else
        _obs = map(o -> partial_observation(gett(o) , getv(o)[1:size(getL₁(o))[1]] , getL₁(o), o.ϵ) , obs)
        _obs = _obs[map(o -> .!isequal(getL₁(o), nothing), obs)]
        @assert !isempty(_obs) "Brownian term is never observed"
        return diffusion_guiding_term(_obs, a[map(o -> .!isequal(getL₁(o), nothing), obs)], GPP.P)
    end
end

filter_backward(GPP::poisson_guiding_term) = filter_backward(GP(GPP))

#         obs = GP.obs
#         n = getn(GP) ; times = gett(GP) ; a = geta(GP) ; m = getm(GP)
#         θ = getθ(GP) ; Y = poisson_terms(GP) ; Z = d-Y
#         if n == 1
#             L₁, ϵ, v = getL₁(obs) , getϵ(obs), getv(obs)
#             LaL⁻¹ = inv(L₁*a*L₁') ; LC⁻¹ = L'*LaL⁻¹/ϵ  ; H = LC⁻¹*L ; F =  LC⁻¹*v
#         else
#             L, ϵ, v = getL(obs[n]) , getϵ(obs[n]), getv(obs[n])
#             H = [zeros(d,d) for k in 1:n] ; F = [zeros(d) for k in 1:n]
#             LaL⁻¹ = [zeros(getm(o), getm(o)) for o in obs] ; LC⁻¹ = [zeros(d,getm(o)) for o in obs]
#             LaL⁻¹[n] = n == 1 ? inv(L*a*L') : inv(L*a[n]*L')
#             LC⁻¹[n] = L'*LaL⁻¹[n]/ϵ ; H[n] = LC⁻¹[n]*L ; F[n] =  LC⁻¹[n]*v
#             for i in 2:n
#                 k=n-i+1
#                 L, ϵ, v  = getL(obs[k]) , getϵ(obs[k]), getv(obs[k])
#                 LaL⁻¹[k] = inv(L*a[k]*L') ; LC⁻¹[k] = L'*LaL⁻¹[k]/ϵ
#                 z = inv( Matrix{Float64}(I,d,d) + H[k+1]*a[k+1]*(times[k+1]-times[k]) ) # z_{k+1}(t_k)
#                 H[k] = z*H[k+1]+LC⁻¹[k]*L # First H(t_k+) then L'C⁻¹L
#                 F[k] = z*F[k+1]+LC⁻¹[k]*v # First F(t_k+) then L'C⁻¹v
#             end
#         end
#     end
#     return (H, F, LaL⁻¹ , LC⁻¹)
# end

"""
    log_guiding_term(info, GP::diffusion_guiding_term)

Returns a function ``(ℓ, t, x) \\mapsto log h̃(t,x+ξ_ℓ) - log h̃(t,x)``. 
`info` should be the quadruple `(H, F, LaL⁻¹ , LC⁻¹)` that results from `filter_backward`. 
"""
function log_guiding_term(info, GP::diffusion_guiding_term)
    times, a, d = gett(GP), GP.a, getd(GP)
    H,F,LaL⁻¹,LC⁻¹ = info
    if getn(GP) > 1
        function fun1(ℓ,t,x)
            k = searchsortedfirst(times, t) # t in (times[k-1],times[k]]
            if t == times[k]
                Ht = k == getn(GP) ? H[k] : H[k] - LC⁻¹[k]*getL(GP)[k]
                Ft = k == getn(GP) ? F[k] : F[k] - LC⁻¹[k]*getv(GP)[k]
            else
                z = d == 1 ? 1/(1.0 + H[k]*a[k]*(times[k]-t)) : inv(Matrix{Float64}(I,d,d) + H[k]*a[k].*(times[k]-t))
                Ht, Ft = z*H[k] , z*F[k]
            end
            return d == 1 ? ℓ.ξ*(Ft - Ht*(x+0.5*ℓ.ξ)) : dot(ℓ.ξ, Ft - Ht*(x+0.5*ℓ.ξ) )
        end
        return fun1
    else
        function fun2(ℓ,t,x)
            if t == times
                Ht = H
                Ft = F
            else
                z = d == 1 ? 1/(1.0 + H*a*(times-t)) : inv(Matrix{Float64}(I,d,d) + H*a.*(times-t))
                Ht, Ft = z*H , z*F
            end
            return d == 1 ? ℓ.ξ*(Ft - Ht*(x+0.5*ℓ.ξ)) : dot(ℓ.ξ, Ft - Ht*(x+0.5*ℓ.ξ) )
        end
        return fun2
    end
end

"""
    log_guiding_term_poisson_only(GPP::poisson_guiding_term)

Returns the log-guiding term as a function of `(ℓ,t,x)` for only poisson components
"""
function log_guiding_term_poisson_only(GPP::poisson_guiding_term)
    function pois_part(ℓ,t,x) 
        ξ = typeof(ℓ.ξ) <: Real ? ℓ.ξ : ℓ.ξ[end]
        y = typeof(x) <: Real ? x : x[end]
        if ξ == 0
            return 0.0
        end
        if getn(GPP) == 1 # Y = 1 , n=1
            θ = getθ(GPP)
            yT = typeof(getv(GPP)) <: Real ? getv(GPP) :  getv(GPP)[end] ; T = gett(GPP)
            return  -abs(ξ)*log(θ*(T-t)) + sum([ log(abs(yT - y)- j) for j in 0:1:abs(ξ)-1 ]) 
        else # Y=1 , n > 1
            if ξ == 0
                return 0.0
            else
                θ = getθ(GPP)
                vv = map(v -> typeof(v) <: Real ? v : v[end] , getv(GPP)) ; tt = gett(GPP) ; k = getk(tt, t)
                out = -abs(ξ)*log(θ[k]*(tt[k]-t)) + sum([ log(abs(vv[k] - y)- j) for j in 0:1:abs(ξ)-1 ])
                if k == getn(GPP)
                    return θ[k] == 0.0 ? -Inf : out
                else
                    # out += sum([ -ℓ.ξ[end]*log(θ[i+1]*(tt[i+1]-tt[i])) + sum([ log(vv[i+1] - vv[i]- j) for j in 0:1:ℓ.ξ[end]-1 ]) for i in k:getn(GPP)-1])
                    return θ[k] == 0.0 ? -Inf : out
                end
            end
        end
    end
    return pois_part
end
# DISCLAIMER: ONLY WORKS FOR FULL OBSERVATIONS OF THE POISSON PART
function log_guiding_term(info, GPP::poisson_guiding_term)
    d = getd(GPP) ; Y = poisson_terms(GPP)
    if d == Y 
        return log_guiding_term_poisson_only(GPP)
    end
    θ = Y == 1 ? GPP.a[end,end] : diag(GPP.a[end-Y+1:end , end-Y+1:end])
    SDE_part(ℓ,t,x) = log_guiding_term(info, GP(GPP))(reaction(ℓ.λ,ℓ.ξ[1:d-Y]),t,x[1:d-Y])
    function pois_part(ℓ,t,x) 
        if Y == 1
            y = x[end]
            if ℓ.ξ[end] == 0
                return 0.0
            end
            if getn(GPP) == 1 # Y = 1 , n=1
                θ = GPP.a[end,end]
                yT = getv(GPP)[end] ; T = gett(GPP)
                return  -ℓ.ξ[end]*log(θ*(T-t)) + sum([ log(yT - y- j) for j in 0:1:ℓ.ξ[end]-1 ]) 
            else # Y=1 , n > 1
                if ℓ.ξ[end] == 0
                    return 0.0
                else
                    θ = map(a -> a[end,end], GPP.a)
                    vv = map(v -> v[end] , getv(GPP)) ; tt = gett(GPP) ; k = getk(tt, t)
                    out = -ℓ.ξ[end]*log(θ[k]*(tt[k]-t)) + sum([ log(vv[k] - y- j) for j in 0:1:ℓ.ξ[end]-1 ])
                    if k == getn(GPP)
                        return θ[k] == 0.0 ? -Inf : out
                    else
                        # out += sum([ -ℓ.ξ[end]*log(θ[i+1]*(tt[i+1]-tt[i])) + sum([ log(vv[i+1] - vv[i]- j) for j in 0:1:ℓ.ξ[end]-1 ]) for i in k:getn(GPP)-1])
                        return θ[k] == 0.0 ? -Inf : out
                    end
                end
            end
        else
            # This part should be reviewed first 
            println("The part with multiple  poisson components should be reviewed first")
            y = x[d-Y+1:d]
            if getn(GPP) == 1
                θ = diag(GPP.a[end-Y+1:end , end-Y+1:end])
                yT = getv(GPP)[end-Y+1:end] ; T = gett(GPP)
                return sum([ ξ == 0 ? 0 : ξ*log(θ[i]*(T-t)) + sum([ log(yT[i] - y[i]- j) for j in 0:1:ξ-1 ]) for (i,ξ) in enumerate(ℓ.ξ[end-Y+1:end])])
            else
                θ = map(a -> diag(a[end-Y+1:end , end-Y+1:end]), GPP.a)
                vv = map(v -> v[end-Y+1:end], getv(GPP)) ; tt = gett(GPP) ; k = getk(tt,t)
                out = sum([ -ℓ.ξ[Y+i]*log(θ[k]*(tt[k]-t)) + sum([log(vv[k][i]-y[i]-j) for j in 0:1:ℓ.ξ[Y+i]-1]) for i in eachindex(y)]) 
                for l in k:getn(GPP)-1
                    out += sum([ -ℓ.ξ[Y+i]*log(θ[l+1]*(tt[l+1]-tt[l])) + sum([log(vv[l+1][i]-vv[l][i]-j) for j in 0:1:ℓ.ξ[Y+i]-1]) for i in eachindex(y)]) 
                end
                return out
            end
        end
    end
    return (ℓ,t,x) -> SDE_part(ℓ,t,x) + pois_part(ℓ,t,x)
end

"""
    guiding_term(info, GP::T) where {T<:Guided_Process}

exp of the log guiding terms
"""
guiding_term(info, GP::T) where {T<:Guided_Process} = (ℓ,t,x) -> exp(log_guiding_term(info, GP)(ℓ,t,x))


# function log_guiding_term(info, GPP::Guided_Proces_Poisson)
#     out = log_guiding_term(info, GP(GPP))

# end
function log_poisson_density(θ::Float64 ,s::Float64, x::T, t::Float64, y::T) where {T<:Real}
    @assert θ>0 "Parameter must be positive"
    -θ*(t-s)+(y-x)*log(θ*(t-s)) - logfactorial(y-x)
end

function poisson_density(θ::Float64 ,s::Float64, x::T, t::Float64, y::T) where {T<:Real}
    @assert θ>0 "Parameter must be positive"
    return exp(log_poisson_density(θ,s,x, t,y))
end

function log_poisson_density(θ::Array{Float64,1}, x::T,s::Float64, times::Array{Float64,1},values::Array{T,1}) where {T<:Real}
    k = getk(times, s)
    out = log_poisson_density(θ[1], s, x, times[1],values[1])
    for i in 2:length(times)-1
        out += log_poisson_density(θ[i],times[i], values[i], times[i+1], values[i+1])
    end
    out
end

function poisson_density(θ::Array{Float64,1}, x::T,s::Float64, times::Array{Float64,1},values::Array{T,1}) where {T<:Real}
    exp(log_poisson_density(θ,s,x,times,values))
end

# """
#     guiding_term_scaledBM(ϕ, obs::observation{T}) where {T}

# Induces the guiding term ``Exp(|xT - x|^2/2ϕ(T-t))``. This is the guiding term induced
# by the auxiliary process ``√ϕ Wₜ``, where ``W`` is a standard Brownian motion.
# """
# function guiding_term_scaledBM(ϕ, obs::observation{T}) where {T}
#     t₁, xT = gett(obs), getx(obs)
#     fun = (ℓ, t, x) -> exp(-(dot(xT .- x .- ℓ.ξ, xT .- x .- ℓ.ξ) - dot(xT .- x , xT .- x))/(2*ϕ*(t₁-t)))
#     return fun
# end

# """
#     guiding_term_scaledBM(ϕ, obs::Array{observation{T}, 1}) where {T}

# Same as `guiding_term_scaledBM` for multiple observation. Assumes that the array of observations is ordered by time.
# """
# function guiding_term_scaledBM(ϕ, obs::Array{observation{T}, 1}) where {T}
#     times, locations = map(o -> gett(o), obs) , map(o -> getx(o), obs)
#     d(x,y) = dot(y .- x, y .- x)
#     function fun(ℓ, t, x)
#         k = getk(t,times)
#         first = exp(-(d(locations[k], x+ℓ.ξ) - d(locations[k],x))/(2*ϕ*(times[k]-t)))
#         return first
#     end
#     return fun
# end

# """
#     guiding_term_partial(L::Union{Array{Bool,2},Array{Int64,2}}, ϕ, obs::observation{T}) where {T}   

# Guiding term for a partial observation ``LX(T)=v`` obtained from the density of the measure 
# ``ℙ(L\\sqrt{\\phi}W(T) \\in \\cdot \\mid \\sqrt{\\phi}W(t) = x)``. `obs` is assumed to contain `v` and `L` an integer-valued matrix. 
# """
# function guiding_term_partial(L::Union{Array{Bool,2},Array{Int64,2}}, ϕ, obs::observation{T}) where {T}
#     t₁, v = gett(obs), getx(obs)
#     Σ = inv(L*ϕ*L')
#     d(x,y) = (y-x)'*Σ*(y-x)
#     fun = (ℓ,t,x) -> exp(-(d(v, L*(x+ℓ.ξ))- d(v, L*x))/(2*(t₁-t)))
#     return fun
# end

# """
#     guiding_term_partial(L::Union{Array{Array{Bool,2},1}, Array{Array{Int64,2},1}}, ϕ, obs::Array{observation{T}, 1}) where {T}

# Guiding term for an array of partial observations ``L_i X(t_i)=v_i``, ``i=1,\\dots,n``. `L` is assumed to be an array of 
# integer-valued matrices of the same length as `obs. `
# """
# function guiding_term_partial(L::Union{Array{Array{Bool,2},1}, Array{Array{Int64,2},1}}, ϕ, obs::Array{observation{T}, 1}) where {T}
#     times, locations, n = map(o -> gett(o), obs) , map(o -> getx(o), obs) , length(obs)
#     function fun(ℓ, t, x)
#         k = getk(t, times)
#         vt = [locations[i] for i in k:n]
#         Lt = [L[i] for i in k:n]
    
#         first = guiding_term_partial(L[k],ϕ,obs[k])(ℓ,t,x)
#         return first
#     end
#     return fun
# end

"""
    condition_reaction(ℓ::reaction{S}, xT::Union{S, Array{S,1}}, T::Real, guiding_term::Function) where {S<:Real}

Input is a reaction `ℓ`, a desired endpoint `xT`, end time `T` and a guiding term that is a function of (ℓ, t, x) and returns
``λ̃(t, x+ℓ.ξ)/λ̃(t,x)``. Returns a reaction with the same difference vector `ξ`, but with the conditioned rate specified in the paper.
"""
function condition_reaction(ℓ::reaction{S}, guiding_term::Function) where {S<:Real}
    λᵒ(t,x) = ℓ.λ(t,x) == 0.0 ? 0.0 : ℓ.λ(t,x)*guiding_term(ℓ, t, x)
    return reaction{S}(λᵒ, ℓ.ξ)
end

function uncondition_reaction(ℓᵒ::reaction{S}, guiding_term::Function) where {S<:Real}
    λ(t,x) = ℓᵒ.λ(t,x)/guiding_term(ℓᵒ, t, x)
    return reaction{S}(λ, ℓᵒ.ξ)
end

"""
    condition_process(P::ChemicalReactionProcess{S}, guiding_term::Function) where {S<:Real}

Returns a new `ChemicalReactionProcess{S}` where all reaction in the network are conditioned using `condition_reaction`.
"""
function condition_process(P::ChemicalReactionProcess{S}, guiding_term::Function) where {S<:Real}
    return ChemicalReactionProcess{S}(P.𝒮 , [condition_reaction(ℓ, guiding_term) for ℓ in reaction_array(P)])
end

function uncondition_process(P::ChemicalReactionProcess{S}, guiding_term::Function) where {S<:Real}
    return ChemicalReactionProcess{S}(P.𝒮 , [uncondition_reaction(ℓ, guiding_term) for ℓ in reaction_array(P)])
end


# function setδ(η, ℓ::reaction{S}, t::Real, x::S, T::Real, xT::S, d²::Function) where {S<:Real}
#     return T-t- 1/( 2*log(η)/C(ℓ, x, xT, d²) + 1/(T-t) )
# end
