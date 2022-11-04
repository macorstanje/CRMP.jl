include("/Users/marc/Documents/GitHub/CRMP.jl/src/CRMP.jl")
extractcomp(a, i) = map(x -> x[i], a)
κ₁ = 200
κ₂ = 10
dₘ = 25
dₚ = 1

T = 1.0
x₀ = [1, 10, 25]

P = GTT(κ₁, κ₂, dₘ, dₚ)
tt, xx = simulate_forward(constant_rate(), x₀, T, P)
plot(tt, extractcomp(xx,2), label = P.𝒮[2])
plot!(tt, extractcomp(xx,3), label = P.𝒮[3])

xT = xx[end]
ϕ = 3.5
CP = condition_process(P, xT, T, dist²(ϕ))
tto, xxo = simulate_forward(conditional(), x₀, xT, T, CP, dist²(ϕ), 0.5)
plot(tto, extractcomp(xxo,2), label = P.𝒮[2])
plot!(tto, extractcomp(xxo,3), label = P.𝒮[3])
