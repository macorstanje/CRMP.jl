"""
    loglikelihood(tt, xx, GP::Guided_Process, info)

Specifically for a guiding term exp(-0.5 x' H(t)x + F(t)'x) 
"""
function loglikelihood(tt, xx, GP::diffusion_guiding_term, info)
    n, d = getn(GP), getd(GP)
    if n == 1 
        if d ==1 
            return loglikelihood_1obs_1dim(tt,xx,GP,info)
        else
            return loglikelihood_1obs(tt,xx,GP,info)
        end
    else
        if d == 1
            return loglikelihood_1dim(tt,xx,GP,info)
        else
            return loglikelihood_ddims_nobs(tt,xx,GP,info)
        end
    end
end

function loglikelihood_1obs(tt,xx,GP::diffusion_guiding_term,info)
    T, a , d = gett(GP), geta(GP), getd(GP)
    (H, F, LaL⁻¹ , LC⁻¹) = info
    # Array of size n with the indices of the first reaction after the observation times 
    # in the vector tt of the realization (or if n=1 the final time )
    z0 = inv(Matrix{Float64}(I,d,d) +H*a*T )
    out = -0.5*dot(xx[1], z0*H*xx[1]) + dot(z0*F, xx[1]) # log h̃(0,x₀)

    # use process between t_k-1 and t_k 
    z(t) = inv(Matrix{Float64}(I,d,d) +H*a*(T-t) )
    for i in 1:length(tt)-1
        t, x = tt[i], xx[i]
        logh̃(s) = -0.5*dot(x, z(s)*H*x) + dot( z(s)*F, x)
        ℒh̃(s,p) = sum([ ℓ.λ(s,x)*(exp(dot(ℓ.ξ ,z(s)*(F-H*(x+0.5*ℓ.ξ)) )) - 1) for ℓ in reaction_array(GP.P) ])
        # extraterm(s,p) = -0.5*dot(z(s)*F, a*z(s)*F) 
        out += logh̃(tt[i+1])-logh̃(tt[i])
        out += solve(IntegralProblem(ℒh̃,tt[i], tt[i+1]) , HCubatureJL() ; reltol = 1e-3, abstol = 1e-3).u
        # out += solve(IntegralProblem(extraterm,tt[i],tt[i+1]) , HCubatureJL() ; reltol = 1e-3, abstol=1e-3).u
    end
    out -= -0.5*dot(xx[end], H*xx[end])+dot(F,xx[end])
    return out
end

function loglikelihood_1obs_1dim(tt,xx,GP::diffusion_guiding_term,info)
    T, a  = gett(GP), geta(GP)
    (H, F, LaL⁻¹ , LC⁻¹) = info
    # Array of size n with the indices of the first reaction after the observation times 
    # in the vector tt of the realization (or if n=1 the final time )
    z0 = 1/(1.0 + H*a*T )
    out = -0.5*z0*H*xx[1]^2 + z0*F*xx[1] # log h̃(0,x₀)

    # use process between t_k-1 and t_k 
    z(t) = 1/(1.0 + H*a*(T-t) )
    for i in 1:length(tt)-1
        t, x = tt[i], xx[i]
        logh̃(s) = -0.5*z(s)*H*x^2 + z(s)*F*x
        # extraterm(s,p) = -0.5*a*(z(s)*F)^2
        ℒh̃(s,p) = sum([ ℓ.λ(s,x)*(exp( ℓ.ξ*z(s)*(F-H*(x+0.5*ℓ.ξ)) ) - 1) for ℓ in reaction_array(GP.P)]) #ℒh̃/h̃
        out += logh̃(tt[i+1])-logh̃(tt[i])
        out += solve(IntegralProblem(ℒh̃,tt[i], tt[i+1]) , HCubatureJL() ; reltol = 1e-3, abstol = 1e-3).u
        # out += solve(IntegralProblem(extraterm,tt[i],tt[i+1]) , HCubatureJL() ; reltol = 1e-3, abstol=1e-3).u
    end
    out -= -0.5*H*xx[end]^2+ F*xx[end] # Why did I add this?
    return out 
end

function loglikelihood_1dim(tt,xx,GP::diffusion_guiding_term,info)
    times, a = gett(GP), geta(GP)
    (H, F, LaL⁻¹ , LC⁻¹) = info
    # Array of size n with the indices of the first reaction after the observation times 
    # in the vector tt of the realization (or if n=1 the final time )
    ind = [searchsortedfirst(tt, times[k]) for k in eachindex(times)]
    z0 = 1/(1.0 +H[1]*a[1]*times[1] )
    out = -0.5*z0*H[1]*xx[1]^2 + z0*F[1]*xx[1] # log h̃(0,x₀)
    for k in eachindex(ind)
        # use process between t_k-1 and t_k
        # ttk, xxk is a this process, which is constant before the first reaction and after the last
        if k == 1 
            ttk = vcat(tt[1:ind[k]-1], times[k])
            xxk = vcat(xx[1:ind[k]-1], xx[ind[k]-1])
        else
            ttk = vcat(times[k-1],tt[ind[k-1]:ind[k]-1], times[k])
            xxk = vcat(xx[ind[k-1]], xx[ind[k-1]:ind[k]-1], xx[ind[k]-1])
        end
        zk(t) = 1/(1.0 +H[k]*a[k]*(times[k]-t))
        for i in 1:length(ttk)-1
            t, x = ttk[i], xxk[i]
            logh̃(s) = -0.5*zk(s)*H[k]*x^2 + zk(s)*F[k]*x
            # extraterm(s,p) = -0.5*a[k]*(zk(s)*F[k])^2
            ℒh̃(s,p) = sum([ ℓ.λ(s,x)*(exp(ℓ.ξ*zk(s)*(F[k]-H[k]*(x+0.5*ℓ.ξ)) ) - 1) for ℓ in reaction_array(GP.P) ])
            out += logh̃(ttk[i+1])-logh̃(ttk[i])
            out += solve(IntegralProblem(ℒh̃,ttk[i], ttk[i+1]) , HCubatureJL() ; reltol = 1e-3, abstol = 1e-3).u
            # out += solve(IntegralProblem(extraterm,tt[i],tt[i+1]) , HCubatureJL() ; reltol = 1e-3, abstol=1e-3).u
        end
        out -= -0.5*H[k]*xxk[end]^2+ F[k]*xxk[end]
    end
    return out 
end

function loglikelihood_ddims_nobs(tt, xx, GP::diffusion_guiding_term, info)
    times,a , d =  gett(GP), geta(GP), getd(GP)
    (H, F, LaL⁻¹ , LC⁻¹) = info
    # Array of size n with the indices of the first reaction after the observation times 
    # in the vector tt of the realization (or if n=1 the final time )
    ind = [searchsortedfirst(tt, times[k]) for k in eachindex(times)]
    z0 = inv(Matrix{Float64}(I,d,d) +H[1]*a[1]*times[1] )
    out = -0.5*dot(xx[1], z0*H[1]*xx[1]) + dot(z0*F[1], xx[1]) # log h̃(0,x₀)
    for k in eachindex(ind)
        tk = times[k]
        # use process between t_k-1 and t_k 
        if k == 1 
            ttk = vcat(tt[1:ind[k]-1], times[k])
            xxk = vcat(xx[1:ind[k]-1], [xx[ind[k]-1]])
        else
            ttk = vcat(times[k-1],tt[ind[k-1]:ind[k]-1], times[k])
            xxk = vcat([xx[ind[k-1]]], xx[ind[k-1]:ind[k]-1], [xx[ind[k]-1]])
        end
        zk(t) = inv(Matrix{Float64}(I,d,d) +H[k]*a[k]*(times[k]-t) )
        for i in 1:length(ttk)-1
            t, x = ttk[i], xxk[i]
            logh̃(s) = -0.5*dot(x, zk(s)*H[k]*x) + dot( zk(s)*F[k], x)
            # extraterm(s,p) = -0.5*dot(zk(s)*F[k], a[k]*zk(s)*F[k])
            ℒh̃(s,p) = sum([ ℓ.λ(s,x)*(exp(dot(ℓ.ξ ,zk(s)*(F[k]-H[k]*(x+0.5*ℓ.ξ)) )) - 1) for ℓ in reaction_array(GP.P) ])
            out += logh̃(ttk[i+1])-logh̃(ttk[i])
            out += solve(IntegralProblem(ℒh̃,ttk[i], ttk[i+1]) , HCubatureJL() ; reltol = 1e-3, abstol = 1e-3).u
            # out += solve(IntegralProblem(extraterm,tt[i],tt[i+1]) , HCubatureJL() ; reltol = 1e-3, abstol=1e-3).u
            # for j in 1:10
            #     # ∫ log h̃(s,X(s)) ds
            #     logh̃(s) = -0.5*dot(x, zk(s)*H[k]*x) + dot( zk(s)*F[k], x)
            #     out += ( -0.5*dot(x, zk(t)*H[k]*x) + dot( zk(t)*F[k], x) )*Δt/10 
            #     # ∫ (ℒ h̃)/h̃ (s,X(s)) ds
            #     out += sum([ ℓ.λ(t,x)*(exp(dot(ℓ.ξ ,zk(t)*(F[k]-H[k]*(x+0.5*ℓ.ξ)) )) - 1) for ℓ in GP.P.ℛ ])*Δt/10
            #     t += Δt/10
            # end
        end
        out -= -0.5*dot(xxk[end], H[k]*xxk[end])+dot(F[k],xxk[end])
    end
    return out 
end

function loglikelihood(tt, xx, GPP::poisson_guiding_term{T}, info) where {T}
    if getn(GPP) == 1
        if poisson_terms(GPP) == 1
            return loglikelihood_1obs_1dim(tt,xx, GPP,info)
        end
    end
end


### ONLY WORKS WHEN WE HAVE 1 POISSON COMPONENT
function loglikelihood_1obs_1dim(tt,xx, GPP::poisson_guiding_term{T},info) where {T}
    t₁, a, θ  = gett(GPP), geta(GPP), getθ(GPP)[1,1]
    d = getd(GPP) ; Y = poisson_terms(GPP) ; Z = d-Y
    yT = getv(GPP.obs)[end] 
    @assert poisson_terms(GPP) == 1 "This function currently only works when there is 1 poisson component"
    (H, F, LaL⁻¹ , LC⁻¹) = info
    # Array of size n with the indices of the first reaction after the observation times 
    # in the vector tt of the realization (or if n=1 the final time )
    if !(getL(GPP.obs)*xx[end] == getv(GPP.obs))
        return -Inf
    end
    # log h̃(0,x₀)
    # Brownian term 
    zt(t) = Z == 0 ? 0.0 : ( Z == 1 ? 1/(1.0 + H*a*(T-t) ) : inv(Matrix{Float64}(I,Z,Z) +H*a*(t₁-t)) )
    if Z == 0
        out = 0.0
    elseif Z == 1
        out = -0.5*zt(0.0)*H*xx[1][1]^2 + zt(0.0)*F*xx[1][1]
    else
        out = -0.5*dot(xx[1][1:Z], zt(0.0)*H*xx[1][1:Z]) + dot(zt(0.0)*F, xx[1][1:Z])
    end
    # Poisson term
    out += logpdf(Poisson(θ*(t₁)), abs(yT-xx[1][end]))+θ*t₁
    # out += abs(yT-xx[1][end])*log(θ*t₁)-sum([log(abs(yT-xx[1][end])-j) for j in 0:1:yT-xx[1][end]-1]) 

    # use process between t_k-1 and t_k 
    for i in 1:length(tt)-1
        t, x = tt[i], xx[i]
        (z,y) = Z == 0 ? (nothing , x) : (x[1:Z] , x[end])
        # Brownian term
        if Z == 1
            out += -0.5*zt(tt[i+1])*H*x^2 + zt(tt[i+1])*F*x - ( -0.5*zt(t)*H*x^2 + zt(t)*F*x )
        elseif Z>1
            out += -0.5*dot(z, zt(tt[i+1])*H*z) + dot(zt(tt[i+1])*F,z) - ( -0.5*dot(z, zt(t)*H*z) + dot(zt(t)*F,z) )
        end
        # Poisson term
        out += t == tt[end-1] ? 0.0 : abs(yT-y)*(log((t₁-tt[i+1])/(t₁-t)))
        ℒh̃(s,p) = sum([ ℓ.λ(s,x)*(guiding_term(info, GPP)(ℓ,s,x) - 1.0) for ℓ in reaction_array(GPP.P) ]) #ℒh̃/h̃
        out += solve(IntegralProblem(ℒh̃,tt[i], tt[i+1]) , HCubatureJL() ; reltol = 1e-3, abstol = 1e-3).u
        # out += solve(IntegralProblem(extraterm,tt[i],tt[i+1]) , HCubatureJL() ; reltol = 1e-3, abstol=1e-3).u
    end
    out -= Z == 0 ? 0.0 : -0.5*dot(xx[end][1:Z], H*xx[end][1:Z]) + dot(F,xx[end][1:Z])
    return out 
end



"""
    loglikelihood_general_1obs(tt,xx, GP::T, info) where {T<:Guided_Process}

General log-likelihood computation, only dependent on guiding term, main method used
"""
function loglikelihood_general_1obs(tt,xx, GP::T, info) where {T<:Guided_Process}
    ℛ = reaction_array(GP.P) ; N = length(tt)
    out = 0.0 #Not logp(tt[1],xx[1], GP) # log h(0,x₀) included in summation

    if !iscorrect_1obs(GP, tt, xx)
        return -Inf         # likelihood 0 if conditioning is not satisfied. 
    end

    integrand(x) = (s,p) -> sum([ ℓ.λ(s,x)*(guiding_term(info, GP)(ℓ, s, x) - 1.0) for ℓ in ℛ ])
    for j in 1:N-2
        ℓj = find_reaction(xx[j], xx[j+1], GP.P)
        out -= log_guiding_term(info, GP)(ℓj, tt[j+1], xx[j])
        out += tt[j] == tt[j+1] ? 0.0 : Integrals.solve(IntegralProblem(integrand(xx[j]), tt[j], tt[j+1]) , HCubatureJL() ; reltol = 1e-5, abstol = 1e-5).u
        # out += quadgk(integrand(xx[j]),tt[j],tt[j+1], rtol = 1e-3)[1]
        # out += integrand(xx[j])(tt[j], 1.0)*(tt[j+1]-tt[j])
    end
    out += Integrals.solve(IntegralProblem(integrand(xx[N-1]), tt[N-1], tt[N]) , HCubatureJL() ; reltol = 1e-5, abstol = 1e-5).u
    # out += integrand(xx[N-1])(tt[N-1], 1.0)*(tt[N]-tt[N-1])
    # out += quadgk(integrand(xx[N-1]),tt[N-1],tt[N], rtol = 1e-3)[1]
    return out
end

"""
    likelihood_general_1obs(tt,xx, GP::T, info) where {T<:Guided_Process}

General likelihood computation, only dependent on guiding term, main method used
"""
likelihood_general_1obs(tt,xx, GP::T, info) where {T<:Guided_Process} = exp(loglikelihood_general_1obs(tt,xx, GP, info))

function find_reaction(x, y, P::ChemicalReactionProcess)
    ℛ = reaction_array(P)
    return ℛ[findall(ℓ -> ℓ.ξ == y-x, ℛ)][1]
end

"""
    iscorrect_1obs(GP::T, tt, xx) where {T<:Guided_Process}

Checks whether LX(T) = v, returns a boolean. 
"""
iscorrect_1obs(GP::T, tt, xx) where {T<:Guided_Process} = (getL(GP)*xx[end] == getv(GP))