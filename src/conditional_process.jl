"""
    Guided processes for chemical reactions
"""

# Can be changed
"""
    dist²(ϕ)

Given a latent variable `ϕ` (possible matrix-valued), returns a quadratic distance
function on 𝕊. For now, we keep it as ``dist²(ϕ)(x,y)=|y-x|^2/ϕ``.
"""
dist²(ϕ) = (x,y) -> dot(x .- y , x .- y)/ϕ

"""
    C(ℓ::reaction{T}, x::Union{T, Array{T,1}}, xT::Union{T, Array{T,1}}, d²::Function) where {T<:Real}

Returns the sign of reaction `ℓ` at `x` given a squared distance function `d²` and a desired endpoint `xT`.
Returns ``d²(xT, x+ℓ.ξ) - d²(xT,x)``
"""
C(ℓ::reaction{T}, x::Union{T, Array{T,1}}, xT::Union{T, Array{T,1}}, d²::Function) where {T<:Real} = d²(xT, x+ℓ.ξ) - d²(xT,x)

"""
    condition_reaction(ℓ::reaction{S}, xT::Union{S, Array{S,1}}, T::Real, d²::Function) where {S<:Real}

Given a reaction `ℓ`, a desired endpoint `xT`, end time `T` and squared distance function `d²`,
returns a reaction with the same difference vector `ξ`, but with the conditioned rate specified in the paper.
"""
function condition_reaction(ℓ::reaction{S}, xT::Union{S, Array{S,1}}, T::Real, d²::Function) where {S<:Real}
    λᵒ(t,x) = ℓ.λ(t,x)*exp(-C(ℓ,x,xT,d²)/(2*(T-t)))
    return reaction{S}(λᵒ, ℓ.ξ)
end

"""
    condition_process(P::ChemicalReactionProcess{S}, xT::Union{S, Array{S,1}}, T::Real, d²::Function) where {S<:Real}

Returns a new `ChemicalReactionProcess{S}` where all reaction in the network are conditioned using `condition_reaction`.
"""
function condition_process(P::ChemicalReactionProcess{S}, xT::Union{S, Array{S,1}}, T::Real, d²::Function) where {S<:Real}
    return ChemicalReactionProcess{S}(P.𝒮 , [condition_reaction(ℓ, xT, T, d²) for ℓ in P.ℛ])
end

"""
     setδ(η::Real, T::Real ,xT::Union{S, AbstractArray{S,1}}, d²::Function) where {S<:Real}

returns the function ``(ℓ, t, x) ↦ T-t- 1/( 2*log(η)/C(ℓ, x, xT, d²) + 1/(T-t) )`` for the thinning algorithm.
"""
function setδ(η::Real, T::Real ,xT::Union{S, AbstractArray{S,1}}, d²::Function) where {S<:Real}
    fun(ℓ, t, x) = T-t- 1/( 2*log(η)/C(ℓ, x, xT, d²) + 1/(T-t) )
    return fun
end

# function setδ(η, ℓ::reaction{S}, t::Real, x::S, T::Real, xT::S, d²::Function) where {S<:Real}
#     return T-t- 1/( 2*log(η)/C(ℓ, x, xT, d²) + 1/(T-t) )
# end
