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

used when the rates of the reactions in the network are decreasing in time
"""
struct decreasing_rate <: method end

"""
    increasing_rate <: method

used when the rates of the reactions in the network are increasing in time
"""
struct increasing_rate <: method end

"""
    gettime(::constant_rate, ℓ::reaction{T}, t::Real, x::Union{T, Array{T,1}}) where {T<:Real}

Returns an Exp(ℓ.λ(t,x))-random variable, provided ''ℓ.λ(t,x)>0'. Else returns 1e10
"""
function gettime(::constant_rate, ℓ::reaction{T}, t::Real, x::Union{T, Array{T,1}}) where {T<:Real}
    return ℓ.λ(t,x)>0 ? -log(rand())/ℓ.λ(t,x) : 1e10
end

"""
    gettime(::decreasing_rate, ℓ::reaction, t, x::Union{T, Array{T,1}}) where {T<:Real}

Returns the reaction time of reaction ℓ if ℓ.λ is decreasing in time. Uses a thinning algorithm
with upper bound 'ℓ.λ(t,x)'
"""
function gettime(::decreasing_rate, ℓ::reaction, t, x::Union{T, Array{T,1}}) where {T<:Real}
    accepted = false
    if ℓ.λ(t,x) == 0
        return 1e10
    else
        τ = 1e10
        while !accepted
            λ̄ = ℓ.λ(t,x)
            τ = -log(rand())/λ̄
            accepted = log(rand()) <= log(ℓ.λ(t+τ,x)) - log(λ̄)
        end
        return τ
    end
end

"""
    gettime(::increasing_rate, ℓ::reaction{T}, t, x::Union{T, Array{T,1}}, setδ::Function) where {T<:Real}

Returns the reaction time of reaction ℓ if ℓ.λ is increasing in time. Uses a thinning algorithm on compact intervals
[t, t+δ] where δ is obtained from 'setδ' and time by δ is moved if the reaction time is not below δ.
"""
function gettime(::increasing_rate, ℓ::reaction{T}, t, x::Union{T, Array{T,1}}, setδ::Function) where {T<:Real}
    accepted = false
    t₀ = t
    if ℓ.λ(t,x) == 0
        return 1e10
    else
        τ = 1e10
        while !accepted
            δ = setδ(ℓ, t, x)
            λ̄ = ℓ.λ(t+δ,x)
            τ = -log(rand())/λ̄
            if τ <= δ
                accepted = log(rand()) <= log(ℓ.λ(t+τ,x)) - log(λ̄)
            else
                t = t+δ
            end
        end
        return τ + t - t₀
    end
end
