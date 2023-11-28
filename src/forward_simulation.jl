"""
    next_jump(::constant_rate, t::Real, x::Union{T, Array{T,1}}, P::ChemicalReactionProcess{T}) where {T<:Real}

If the process is of constant (in time) rate, simulates exponential jump times for all reactions
and returns the reaction with the shortest jump time along with the respective time.
"""
function next_jump(::constant_rate, t::Real, x::Union{T, Array{T,1}}, P::ChemicalReactionProcess{T}) where {T<:Real}
    ℛ = reaction_array(P)
    next_reaction = ℛ[1]
    time  = 1e10
    for ℓ in ℛ # For each reaction, compute reaction time and pick the reaction with the shortest time
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


function next_jump(::Gillespie, t::Real, x::Union{T, Array{T,1}}, P::ChemicalReactionProcess{T}) where {T<:Real}
    ℛ = reaction_array(P)
    λ₀ = sum([ℓ.λ(t,x) for ℓ in ℛ])
    if λ₀ == 0.0
        return 1e10, ℛ[1]
    end
    if isinf(λ₀)
        time = 0.0
        next_reaction = ℛ[findall(a -> isinf(a), map(ℓ -> ℓ.λ(t,x), ℛ))][1]
        return time, next_reaction
    end
    #println("t = $t, x = $x, λ₀ = $λ₀")
    time = -log(rand())/λ₀
    # println("t = $t, x = $x, weights = $([ℓ.λ(t,x)/λ₀ for ℓ in ℛ])")
    next_reaction = ℛ[sample(1:length(ℛ), Weights([ℓ.λ(t,x)/λ₀ for ℓ in ℛ]))]
    return time, next_reaction
end
"""
    simulate_forward(::constant_rate, x₀::Union{T1, Array{T1,1}}, T::T2, P::ChemicalReactionProcess{T1}) where {T1<:Real, T2<:Real}

Forward simulation starting from `x₀`, untime time `T` of the process specified in `P` when jump rates are constant (in time).
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
    simulate_forward(::Gillespie, x₀::Union{T1, Array{T1,1}}, T::T2, P::ChemicalReactionProcess{T1}) where {T1<:Real, T2<:Real}

Forward simulation starting from `x₀`, untime time `T` of the process specified in `P` when jump rates are constant (in time).
"""
function simulate_forward(::Gillespie, x₀::Union{T1, Array{T1,1}}, T::T2, P::ChemicalReactionProcess{T1}) where {T1<:Real, T2<:Real}
    @assert T>0 "T must be positive"
    t = 0.; tt = [t]; x = x₀; xx = [x]
    while t<T
        time, next_reaction = next_jump(Gillespie(), t, x, P)
        t = t+time ; x =  x+next_reaction.ξ
        push!(tt, t) ; push!(xx, x)
    end
    return push!(tt[1:end-1], T), push!(xx[1:end-1], xx[end-1])
end



"""
    simulate_forward(x₀,GP::Guided_Process, info)

Upper bounds too large, does not work very well.
"""
function simulate_forward(x₀, GP::Guided_Process, info)
    ℛ = GP.P.ℛ ; n = getn(GP) ; d = getd(GP) ; times = gett(GP)
    t,x = 0.0, x₀
    tt, xx = [t], [x]
    for k in 1:n
        t,x = tt[end], xx[end]
        while t < (n == 1 ? times : times[k])
            Δ = [10e6 for ℓ in ℛ]
            for (i,ℓ) in enumerate(ℛ)
                if ℓ.λ(t,x) > 0
                    Δ[i] = gettime(thinning(), ℓ, t, x, log_guiding_term(info,GP), GP)
                end
            end
            dt, μ = findmin(Δ)
        t = t+dt
        x = x + ℛ[μ].ξ
        push!(xx,x)
        push!(tt,t)
        end
        tt = vcat(tt[1:end-1], times[k]) ; xx[end] = xx[end-1]
    end
    return tt, xx
end

"""
    simulate_forward_monotone(x₀, GP::Guided_Process, info)

Currently the main simulation method. uitinizes the δ
"""
function simulate_forward_monotone(x₀, GP::diffusion_guiding_term{T}, info) where {T}
    ℛ = reaction_array(GP.P) ; n = getn(GP) ; d = getd(GP) ; times = gett(GP)
    t,x = 0.0, x₀
    tt, xx = [t], [x]
    H, F, LaL⁻¹, LC⁻¹ = info
    
    for k in 1:n
        o = n == 1 ? GP.obs : GP.obs[k]
        t,x = tt[end], xx[end]
        v, L = getv(o), getL(o)
        LaLinv = n == 1 ? LaL⁻¹ : LaL⁻¹[k]
        dist(x,y) = dot( y-x , LaLinv*(y-x))
        while t < gett(o)
            Δ = [10e6 for ℓ in ℛ] # initialized reaction times for all reactions
            for (i,ℓ) in enumerate(ℛ)
                if ℓ.λ(t,x) > 0
                    diff = dist(v, L*(x+ℓ.ξ)) - dist(v, L*x)
                    if diff < 0 # ℓ is a reaction that takes X closer to v
                        Δ[i] = gettime(increasing_rate(), ℓ, t, x, info,GP, setδ(0.5, diff, o))
                    else
                        Δ[i] = gettime(decreasing_rate(), ℓ, t, x, info, GP)
                    end
                end
            end
            dt, μ = findmin(Δ)
            t = t+dt
            x = x + ℛ[μ].ξ
            push!(xx,x)
            push!(tt,t)
        end
        tt = vcat(tt[1:end-1], gett(o)) ; xx[end] = xx[end-1]
    end
    return tt, xx
end

function simulate_forward_monotone(x₀, GP::poisson_guiding_term{T}, info) where {T}
    ℛ = reaction_array(GP.P) ; n = getn(GP) ; d = getd(GP) ; times = gett(GP) ; Y = poisson_terms(GP) ; Z = d-Y
    t,x = 0.0, x₀
    tt, xx = [t], [x]
    H, F, LaL⁻¹, LC⁻¹ = info
    
    for k in 1:n
        o = n == 1 ? GP.obs : GP.obs[k]
        t,x = tt[end], xx[end]
        v, L, m = getv(o), getL(o), getm(o)
        LaLinv = n == 1 ? LaL⁻¹ : LaL⁻¹[k]
        θ = n == 1 ? getθ(GP) : getθ(GP)[k]
        function dist(x,y)
            if Y == 1 && Z == 0
                return θ*(y-x)^2
            else
                return dot( y[1:m-Y]-x[1:m-Y] , LaLinv*(y[1:m-Y]-x[1:m-Y])) + dot(y[m-Y+1:m] - x[m-Y+1:m], θ*(y[m-Y+1:m] - x[m-Y+1:m]))
            end
        end
            while t < gett(o)
                Δ = [10e6 for ℓ in ℛ] # initialized reaction times for all reactions
                for (i,ℓ) in enumerate(ℛ)
                    if ℓ.λ(t,x) > 0
                        #ℓᵒ = condition_reaction(ℓ, guiding_term(info, GP))
                        diff = dist(v, L*(x+ℓ.ξ)) - dist(v, L*x)
                        if diff < 0 # ℓ is a reaction that takes X closer to v
                            Δ[i] = gettime(increasing_rate(), ℓ, t, x, info,GP, setδ(0.5, diff, o))
                        else
                            Δ[i] = gettime(decreasing_rate(), ℓ, t, x, info, GP)
                        end
                    end
                end
                dt, μ = findmin(Δ)
                t = t+dt
                x = x + ℛ[μ].ξ
                push!(xx,x)
                push!(tt,t)
            end
        tt = vcat(tt[1:end-1], gett(o)) ; xx[end] = xx[end-1]
    end
    return tt, xx
end
# function simulate_forward_monotone_1obs(x₀, GP::Guided_Process,info)
#     ℛ = GP.P.ℛ ; d = getd(GP) ; T = gett(GP)
#     t,x = 0.0, x₀
#     tt, xx = [t], [x]
#     H, F, LaL⁻¹, LC⁻¹ = info

#     t,x = tt[end], xx[end]
#     v, L = getv(o), getL(o)

#     dist(x,y) = dot( y-x , LaL⁻¹*(y-x))
#     while t < T
#         Δ = [10e6 for ℓ in ℛ] # initialized reaction times for all reactions
#         for (i,ℓ) in enumerate(ℛ)
#             if ℓ.λ(t,x) > 0
#                 ℓᵒ = condition_reaction(ℓ, guiding_term(info, GP))
#                 diff = dist(v, L*(x+ℓᵒ.ξ)) - dist(v, L*x)
#                 if diff < 0 # ℓ is a reaction that takes X closer to v
#                     Δ[i] = gettime(increasing_rate(), ℓᵒ, t, x, info,GP, setδ(0.5, diff, GP.obs))
#                 else
#                     Δ[i] = gettime(decreasing_rate(), ℓᵒ, t, x)
#                 end
#             end
#         end
#         dt, μ = findmin(Δ)
#         t = t+dt
#         x = x + ℛ[μ].ξ
#         push!(xx,x)
#         push!(tt,t)
#     end
#     tt = vcat(tt[1:end-1], T) ; xx[end] = xx[end-1]
#     return tt, xx
# end    


# make new function that does not check all reactions using δ