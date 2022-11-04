abstract type method end

struct constant_rate <: method end
struct decreasing_rate <: method end
struct increasing_rate <: method end
struct Euler <: method end

function gettime(::constant_rate, ℓ::reaction{T}, t::Real, x::Union{T, Array{T,1}}) where {T<:Real}
    return ℓ.λ(t,x)>0 ? -log(rand())/ℓ.λ(t,x) : 1e10
end

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
