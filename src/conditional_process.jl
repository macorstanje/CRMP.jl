"""
    Guided processes for chemical reactions
"""

# Can be changed
dist²(ϕ) = (x,y) -> dot(x .- y , x .- y)/ϕ
C(ℓ::reaction{T}, x::Union{T, Array{T,1}}, xT::Union{T, Array{T,1}}, d²::Function) where {T<:Real} = d²(xT, x+ℓ.ξ) - d²(xT,x)

function condition_reaction(ℓ::reaction{S}, xT::Union{S, Array{S,1}}, T::Real, d²::Function) where {S<:Real}
    λᵒ(t,x) = ℓ.λ(t,x)*exp(-C(ℓ,x,xT,d²)/(2*(T-t)))
    return reaction{S}(λᵒ, ℓ.ξ)
end

function condition_process(P::ChemicalReactionProcess{S}, xT::Union{S, Array{S,1}}, T::Real, d²::Function) where {S<:Real}
    return ChemicalReactionProcess{S}(P.𝒮 , [condition_reaction(ℓ, xT, T, d²) for ℓ in P.ℛ])
end

function setδ(η, T ,xT, d²)
    fun(ℓ, t, x) = T-t- 1/( 2*log(η)/C(ℓ, x, xT, d²) + 1/(T-t) )
    return fun
end

# function setδ(η, ℓ::reaction{S}, t::Real, x::S, T::Real, xT::S, d²::Function) where {S<:Real}
#     return T-t- 1/( 2*log(η)/C(ℓ, x, xT, d²) + 1/(T-t) )
# end
