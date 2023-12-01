"""
    method

supertype for simulation methods, might find another name for it later
"""
abstract type method end

"""
    constant_rate <: method

used when the rates of the reactions in the network are constant in time
"""
struct constant_rate <: method end

"""
    decreasing_rate <: method

used when the rates of the reactions in the network are decreasing in time. 
This method utilizes a Poisson thinning with the current state as upper bound
"""
struct decreasing_rate <: method end

"""
    increasing_rate <: method

used when the rates of the reactions in the network are increasing in time
This method utilizes a Poisson thinning with the time at t+δ as upper bound. 
"""
struct increasing_rate <: method end

"""
    thinning <: method

Can be used when the rates of the reactions in the network are bounded. 
Utilizes the bounds in lemma ... (lower bound on h̃(t,x), upper bound on h̃(t,x+ℓ.ξ))
"""
struct thinning <: method end


"""
    Gillespie <: method

Gillespie method.  Only used in next_jump function, not in gettime. 
"""
struct Gillespie <: method end

"""
    gettime(::constant_rate, ℓ::reaction{T}, t::Real, x::Union{T, Array{T,1}}) where {T<:Real}

Returns an `Exp(ℓ.λ(t,x))`-random variable, provided `ℓ.λ(t,x)>0`. Else returns `1e10`
"""
function gettime(::constant_rate, ℓ::reaction{T}, t::Real, x::Union{T, Array{T,1}}) where {T<:Real}
    return ℓ.λ(t,x)>0 ? -log(rand())/ℓ.λ(t,x) : 1e10
end

"""
    gettime(::decreasing_rate, ℓ::reaction, t, x::Union{T, Array{T,1}}) where {T<:Real}

Returns the reaction time of reaction `ℓ` if `ℓ.λ` is decreasing in time. Uses a thinning algorithm
with upper bound `ℓ.λ(t,x)`. Returns `1e10` when `ℓ.λ(t,x)=0`. 
"""
function gettime(::decreasing_rate, ℓ::reaction{T}, t, x::Union{T, Array{T,1}}, info, GP::TP) where {T<:Real, TP<:Guided_Process}
    accepted = false
    t₀ = t
    t₁ = getn(GP) == 1 ? gett(GP) : gett(GP)[getk(gett(GP), t)]
    if ℓ.λ(t,x) < 1e-8
        return 1e10
    else
        τ = 1e10 # initialized time, to be replaced
        counter = 0
        while !accepted
            logλ̄ = log(ℓ.λ(t₀,x)) + log_guiding_term(info, GP)(ℓ, t₀, x) # Decreasing rate --> upper bound at time t₀
            τ = -log(rand())/exp(logλ̄) # proposal
            if getn(GP) == 1
                if t+τ >= gett(GP)
                    return t+τ
                end
            else
                if t+τ >= gett(GP)[getk(gett(GP), t)]
                    return t+τ
                end
            end
            try
                accepted = log(rand()) <= log(ℓ.λ(t+τ,x)) +log_guiding_term(info, GP)(ℓ, t+τ, x) - logλ̄
            catch err # Boundserror occurs if t+τ falls after tₙ . Accept in that case ; not a relevant reaction anyway. 
                if isa(err, BoundsError)
                    accepted = true
                    # println("BoundsError, decreasing, t = $t, τ = $τ, λ̄ = $λ̄, t+τ = $(t+τ)")
                end
            end
            counter = counter + 1
            if counter > 1e4 # If no accepted times found in 1e4 attempts, return 1e10. 
                # println("Nothing accepted, decreasing")
                return 1e10
                counter = 0
            end
            t = t+τ # Move t until first accepted time 
        end 
        return  t - t₀ # time elapsed. 
    end
end


"""
    gettime(::increasing_rate, ℓ::reaction{T}, t, x::Union{T, Array{T,1}}, setδ::Function) where {T<:Real}

Returns the reaction time of reaction `ℓ` if `ℓ.λ` is increasing in time. Uses a thinning algorithm on compact intervals
`[t, t+δ]` where `δ` is obtained from `setδ` and time by δ is moved if the reaction time is not below `δ`.
"""
function gettime(::increasing_rate, ℓ::reaction{T}, t, x::Union{T, Array{T,1}}, info, GP::TP, setδ::Function) where {T<:Real, TP<:Guided_Process}
    accepted = false
    t_start = t
    t₀ = t
    if abs( ℓ.λ(t,x)*guiding_term(info, GP)(ℓ, t,x) ) < 1e-8
        return 1e10
    else
        τ = 1e10
        counter = 0
        while !accepted
            δ = setδ(ℓ, t₀, x)
            logλ̄ = log(ℓ.λ(t₀+δ,x)) + log_guiding_term(info, GP)(ℓ,t₀+δ, x)
            τ = -log(rand())/exp(logλ̄)
            if t+τ <= t₀ + δ # The proposed time must lie in (t₀,t₀+δ))
                logacc = log(ℓ.λ(t+τ,x)) + log_guiding_term(info,GP)(ℓ,t+τ, x) - logλ̄
                accepted = log(rand()) <= logacc
                t += τ
            else
                t₀ += δ
                t = t₀
            end
            counter = counter + 1
            if counter > 1e4
                return 1e10
                counter = 0
            end
        end
        return t - t_start
    end
end

"""
    setδ(η::Real, diff::Float64, obs::observation{T}) where {T}

returns the function ``(ℓ, t, x) ↦ T - t - 1/( 2*log(η)/diff + 1/(T-t) )`` for the thinning algorithm.
"""
function setδ(η::Real, diff::Float64 , obs::Union{partial_observation{T}, partial_observation_poisson{T}}) where {T}
    @assert diff < 0 "diff must be negative"
    t₁, ϵ= gett(obs), getϵ(obs)
    function fun(ℓ, t, x)
        out = t₁ - t + ϵ - 1/(2*log(η)/diff + 1/(t₁-t+ϵ)) # the correct δ
        if t+out < t₁
            return out
        else
            return t₁ - t  - 1/(2*log(η)/diff + 1/(t₁-t)) # Alternative for when proposal is above T
        end
    end
    return fun
end

function gettime(::thinning, ℓ::reaction, t, x::Union{T, Array{T,1}}, log_guiding_term::Function, GP::TP) where {T<:Real, TP<:Guided_Process}
    accepted = false
    t₀ = t
    if ℓ.λ(t,x) == 0
        return 1e10
    else
        τ = 1e10
        while !accepted
            logλbar = get_log_upper_bound(ℓ, t, x, GP)
            τ = -log(rand())/(ℓ.λ(t,x)*exp(logλbar))
            try
                accepted = log(rand()) <= log_guiding_term(ℓ,t+τ,x) - logλbar
            catch err
                if isa(err, BoundsError)
                    accepted = true
                end
                counter = counter + 1
                if counter > 1e6
                    println("stuck in a loop at decresing with t=$t and x=$x")
                    return 1e10
                    counter = 0
                end
            end
        end
        return τ+t-t₀
    end
end

# """
#     next_reaction(t,x, GP::Guided_Process, info)

# Returns (ℓ,τ), the next reaction and the corresponding reaction time. 
# """
# function next_reaction(t,x, GP::Guided_Process, info)
#     ℛ = GP.P.ℛ
#     (H,F,LaL⁻¹,LC⁻¹) = info
#     k = getk(gett(GP), t)
#     v, L = getv(GP)[k], getL(GP)[k]
#     diff = [ dot(v-L*(x+ℓ.ξ), LaL⁻¹[k](v-L*(x+ℓ.ξ))) - dot(v-L*x, LaL⁻¹[k](v-L*x)) for ℓ in ℛ ]
#     τ = [1e10 for j in eachindex(ℛ)]

#     # Get candidates for reaction times
#     for j in 1:nr_reactions(GP.P)
#         if diff[j] < 0
#             δ = setδ(ℓ, t, x, 0.5, diff[j], GP.obs[k])
#             λ̄ = ℓ.λ(t+δ,x)*guiding_term(info,GP)(ℓ,t+δ,x)
#             τ[j] = -log(rand())/λ̄
#         else
#             λ̄ = ℓ.λ(t,x)
#             τ[j] = -log(rand())/λ̄
#         end
#     end

#     # Accept/reject in increasing order until the first accepted time (for increasing before δ)
#     j = 1
#     while !accepted
#         j_prop = findfirst(x -> x == j,   invperm(sortperm(τ))) #go through sorted array
#         if τ[j_prop] <= δ
#             if diff[j_prop] > 0 # Decreasing rate
#                 try
#                     accepted = log(rand()) <= log(ℓ.λ(t+τ[j_prop],x)) - log(λ̄)
#                 catch err
#                     if isa(err, BoundsError)
#                         accepted = true
#                     end
#                 end
#             else # increasing rate
#                 acc = log_guiding_term(info,GP)(ℓ,t+τ[j_prop], x) - log_guiding_term(info,GP)(ℓ,t+δ,x)
#                 accepted = log(rand()) <= acc
#             end
#         end
#         j += 1
#     end
# end

# function setδ(ℓ,t,x, η::Real, diff::Float64 , obs::partial_observation)
#     @assert diff < 0 "diff must be negative"
#     t₁,ϵ= gett(obs), getϵ(obs)
#     out = t₁ - t + ϵ - 1/(2*log(η)/diff + 1/(t₁-t+ϵ))
#     if t+out < t₁
#         return out
#     else
#         return t₁ - t  - 1/(2*log(η)/diff + 1/(t₁-t))
#     end
# end