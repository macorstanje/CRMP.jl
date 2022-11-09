"""
    next_jump(::constant_rate, t::Real, x::Union{T, Array{T,1}}, P::ChemicalReactionProcess{T}) where {T<:Real}

If the process is of constant (in time) rate, simulates exponential jump times for all reactions
and returns the reaction with the shortest jump time along with the respective time.
"""
function next_jump(::constant_rate, t::Real, x::Union{T, Array{T,1}}, P::ChemicalReactionProcess{T}) where {T<:Real}
    ℛ = P.ℛ
    next_reaction = typeof(ℛ) == reaction ? ℛ : ℛ[1]
    time  = 1e10
    for ℓ in ℛ
        if ℓ.λ(t,x)>0
            new_time = gettime(constant_rate(), ℓ, t, x)
            if new_time < time
                next_reaction = ℓ
                time = new_time
            end
        end
    end
    return time, next_reaction
end

"""
    simulate_forward(::constant_rate, x₀::Union{T1, Array{T1,1}}, T::T2, P::ChemicalReactionProcess{T1}) where {T1<:Real, T2<:Real}

Forward simulation starting from x₀, untime time 'T' of the process specified in 'P' when jump rates are constant (in time).
"""
function simulate_forward(::constant_rate, x₀::Union{T1, Array{T1,1}}, T::T2, P::ChemicalReactionProcess{T1}) where {T1<:Real, T2<:Real}
    @assert T>0 "T must be positive"
    t = 0.; tt = [t]; x = x₀; xx = [x]
    while t<T
        time, next_reaction = next_jump(constant_rate(), t, x, P)
        t, x = t+time , x+next_reaction.ξ
        push!(tt, t) ; push!(xx, x)
    end
    return push!(tt[1:end-1], T), push!(xx[1:end-1], xx[end-1])
end

"""
    conditional <: method

For specifying that we want to simulate from the guided process
"""
struct conditional <: method end

"""
    simulate_forward(::conditional, x₀::Union{T1, Array{T1,1}}, xT, T::T2, P::ChemicalReactionProcess{T1}, d²::Function, η) where {T1<:Real, T2<:Real}

Forward simulation of the guided process specified in 'P', starting from 'x₀' and with desired end state '(T, xT)'.
"""
function simulate_forward(::conditional, x₀::Union{T1, Array{T1,1}}, xT, T::T2, P::ChemicalReactionProcess{T1}, d²::Function, η) where {T1<:Real, T2<:Real}
    ℛ = P.ℛ
    t, x = 0.0, x₀
    tt, xx = [t], [x]

    while t<T
        Δ = [10e6 for ℓ in ℛ]
        for k in 1:length(ℛ)
            if ℛ[k].λ(t,x) > 0
                if C(ℛ[k], x, xT, d²) >= 0
                    Δ[k] = gettime(decreasing_rate(), ℛ[k], t, x)
                else
                    Δ[k] = gettime(increasing_rate(), ℛ[k], t, x, setδ(η, T, xT, d²))
                end
            end
        end
        dt, μ = findmin(Δ)
        t = t+dt
        x = x + ℛ[μ].ξ
        push!(xx,x)
        push!(tt,t)
    end
    return push!(tt[1:end-1], T), push!(xx[1:end-1], xx[end-1])
end
