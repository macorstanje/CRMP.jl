include("/Users/marc/Documents/GitHub/CRMP.jl/src/CRMP.jl")
extractcomp(a, i) = map(x -> x[i], a)
κ₁ = 200
κ₂ = 10
dₘ = 25
dₚ = 1

T = 1.0
x₀ = [1, 10, 25]

## Gene transcription, translation
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

## Virus kinetics
κ₁, κ₂, κ₃, κ₄, κ₅, κ₆ = 1., .025, 1000., .25, 2., 7.5e-6
P = viral_infection(κ₁, κ₂, κ₃, κ₄, κ₅, κ₆)

x₀ = [1, 100, 1, 1]
tt, xx = simulate_forward(constant_rate(), x₀, 200, P)

# Add 1 to counts to fix log-scale.
fig = plot(tt, extractcomp(xx, 1).+1, label = P.𝒮[1], xlabel = "t", ylabel = "counts", yaxis = :log)
for i in [2,3,4]
    plot!(fig, tt, extractcomp(xx, i).+1, label = P.𝒮[i], xlabel = "t", ylabel = "counts", yaxis = :log)
end
fig
