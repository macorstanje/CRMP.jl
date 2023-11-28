cd("/Users/marc/Documents/GitHub/CRMP.jl/")
using CRMP
using Plots
using LinearAlgebra
using BlockArrays
using ProgressMeter
using Statistics
using Bridge
extractcomp(a, i) = map(x -> x[i], a)
Plots.scalefontsizes(2)

"""
    Bridge, one (partial) observation
"""
# Forward simulation
κ₁ = 200.
κ₂ = 10.
dₘ = 25.
dₚ = 1.

T = 1.0
x₀ = [1, 50, 10]
P = GTT(κ₁, κ₂, dₘ, dₚ) # Original process
tt, xx = simulate_forward(constant_rate(), x₀, T, P)
plotprocess(tt, xx, P)

# CLE approximation
using StaticArrays
using Bridge
const ℝ{N} = SVector{N, Float64}

struct CLE <: ContinuousTimeProcess{ℝ{3}} 
    P::ChemicalReactionProcess
end
Bridge.b(t, x, M::CLE) = sum([ℓ.λ(0.0, x)*ℓ.ξ for ℓ in M.P.ℛ])
Bridge.σ(t, x, M::CLE) = transpose(mapreduce(permutedims, vcat, [sqrt(max(ℓ.λ(0.0,x),0.))*ℓ.ξ for ℓ in M.P.ℛ]))
β(t,x,P::ChemicalReactionProcess) = sum([ℓ.λ(t,x)*ℓ.ξ*ℓ.ξ' for ℓ in P.ℛ])

W = sample(0.0:0.01:1.0, Wiener{ℝ{4}}())
X = solve(Euler(),Float64.(x₀), W, CLE(P))

fig = plot(X.tt, extractcomp(X.yy,2), label = P.𝒮[2], 
            linewidth = 2.0, 
            margin = 10Plots.mm, 
            xlabel = "t", 
            ylabel = "Counts", 
            xguidefontsize = 16, yguidefontsize = 16,
            xtickfontsize = 16, ytickfontsize = 16,
            legend=:topleft, legendfotsize = 16, 
            size = (1800,900), 
            dpi = 300)
plot!(fig, X.tt, extractcomp(X.yy,3), label = P.𝒮[3], 
            linewidth = 2.0, 
            margin = 10Plots.mm, 
            xlabel = "t", 
            ylabel = "Counts",
            xguidefontsize = 16, yguidefontsize = 16, 
            xtickfontsize = 16, ytickfontsize = 16,
            legend=:topleft, 
            legendfontsize = 16, 
            size= (1800,900), 
            dpi = 300)
fig

# Save (partial)observation at T=1.0, C = 0.005LaL'
L = [ 0 1 0 ; 0 0 1]
obs = partial_observation(T, L*xx[end], L, 1e-5)
a = β(T,xx[end],P)# Bridge.σ(T, xx[end], CLE(P))*Bridge.σ(T,xx[end],CLE(P))'
#tto, xxo = simulate_forward(x₀, Guided_Process(obs, a, P))
GP = diffusion_guiding_term(obs,a,P)
info = filter_backward(GP)
@time tto, xxo = simulate_forward_monotone(x₀, GP, info)

plt = plotprocess(tto, xxo, P)
#plot!(plt,[gett(v)] , [getx(v)[2]], seriestype=:scatter,  markersize = 12, label = "Observation", color = :black)
ttm, xxm = gett(obs), getv(obs)
# plot!(plt, ttm, extractcomp(xxm,2), seriestype = :scatter, markersize = 15, label = "Observation of MRNA")
plot!(plt, [ttm], [xxm[1]], seriestype = :scatter, markersize = 15, label = "Observation of $(P.𝒮[2])")
plot!(plt, [ttm], [xxm[2]], seriestype = :scatter, markersize = 15, label = "Observation of $(P.𝒮[3])")
#savefig(plt, "GTT_15Short.png")


ttoo, xxoo = [tto], [xxo]
K = 500
p = Progress(K)
for k in 1:K
    tto, xxo = simulate_forward_monotone(x₀, GP, info)
    push!(ttoo, tto) ; push!(xxoo, xxo)
    next!(p)
end

println("$(round(100*sum([iscorrect_1obs(GP,ttoo[k], xxoo[k]) for k in 1:K]/K), digits=7))% correct")

plt = plot(tt, map(x -> x[1], xx), label = P.𝒮[1], 
        linetype=:steppost, 
        linewidth = 5.0, 
        margin = 10Plots.mm, 
        xlabel = "t", 
        ylabel = "Counts", 
        xguidefontsize = 16, yguidefontsize = 16,
        xtickfontsize = 16, ytickfontsize = 16,
        legend=:topleft, legendfotsize = 16, 
        size = (1800, 900), 
        dpi = 300)
for j in 2:nr_species(P)
    plot!(plt, tt, map(x -> x[j], xx), label = P.𝒮[j], 
            linetype=:steppost, 
            linewidth = 5.0, 
            margin = 10Plots.mm, 
            xlabel = "t", 
            ylabel = "Counts", 
            xguidefontsize = 16, yguidefontsize = 16,
            xtickfontsize = 16, ytickfontsize = 16,
            legend=:topleft, legendfotsize = 16, 
            size = (1800, 900), 
            dpi = 300)
end
for k in (0*K+1):100:1*K
    plot!(plt, ttoo[k], extractcomp(xxoo[k], 2), label = false, linetype=:steppost, color = theme_palette(:auto)[2], alpha = 0.7)
    plot!(plt, ttoo[k], extractcomp(xxoo[k], 3), label = false, linetype=:steppost, color = theme_palette(:auto)[3], alpha = 0.7)
end
plot!(plt, [ttm], [xxm[1]], seriestype = :scatter, markersize = 15, label = "Observation of $(P.𝒮[2])", color = theme_palette(:auto)[2])
plot!(plt, [ttm], [xxm[2]], seriestype = :scatter, markersize = 15, label = "Observation of $(P.𝒮[3])", color = theme_palette(:auto)[3])
plt
savefig(plt, "GTT-obs-multipletracectories.png")

a2 = collect(10:5:100)
a3 = collect(30:5:50)
n2 = length(a2); n3 = length(a3)
mat = zeros(n2,n3)
p = Progress(n2*n3*K)
for i in 1:n2
    for j in 1:n3
        a = diagm([150., a2[i], a3[j]])
        GP = Guided_Process(obs,a,P)
        info = filter_backward(GP)
        for k in 1:K
            tto, xxo = simulate_forward_monotone(x₀, GP, info)
            mat[i,j] = loglikelihood(tto, xxo, GP, info)
            next!(p)
        end
    end
end

mat2 = mat
exp.(mat2)
heatmap(a2, a3, mat, xlabel = "a₂", ylabel = "a₃")

findmax(mat)



##############################################################################
# Multiple observations
##############################################################################
κ₁ = 200.
κ₂ = 10.
dₘ = 25.
dₚ = 1.

T = 1.0
x₀ = [1, 50, 10]
P = GTT(κ₁, κ₂, dₘ, dₚ) # Original process
tt, xx = simulate_forward(constant_rate(), x₀, T, P)

n = 15
LL = [rand([[0 1 0], [0 0 1]]) for k in 1:n]
tint = collect(0:0.001:T)
indices = sort(rand(2:length(tint), n))
times = tint[indices]
xobs = map(t -> xx[searchsortedfirst(tt, t)-1], times)
partial_obs = [partial_observation(times[i], LL[i]*xobs[i], LL[i], 1e-5) for i in 1:n]


label = [true for k in eachindex(LL)]
for k in eachindex(LL)
    if LL[k] == [0 1 0]
        label[k] = true
    else
        label[k] = false
    end
end

plt = plotprocess(tt, xx, P)
plot!(plt, times[label], extractcomp(xobs[label],2), seriestype = :scatter, markersize = 15, label = "Observation of MRNA")
plot!(plt, times[.!label], extractcomp(xobs[.!label],3), seriestype = :scatter, markersize = 15, label = "Observation of Protein")
#savefig(plt, "GTT-multiple-partial-obs")


#a = [210*Matrix(I,3,3) for i in 1:n]
a = [sum([ ℓ.λ(gett(o), getL(o)\getv(o) .+ [1,0,0])*ℓ.ξ*ℓ.ξ' for ℓ in P.ℛ ]) for o in partial_obs] 
GP = diffusion_guiding_term(partial_obs,a ,GTT(κ₁,κ₂,dₘ,dₚ))
info = filter_backward(GP)
@time tto, xxo = simulate_forward_monotone(x₀, GP, info)
ttoo, xxoo = [tto], [xxo]
K = 1000
p = Progress(K)
for k in 1:K
    tto, xxo = simulate_forward_monotone(x₀, GP, info)
    push!(ttoo, tto) ; push!(xxoo, xxo)
    next!(p)
end

plt = plot(tt, map(x -> x[1], xx), label = P.𝒮[1], 
        linetype=:steppost, 
        linewidth = 5.0, 
        margin = 10Plots.mm, 
        xlabel = "t", 
        ylabel = "Counts", 
        xguidefontsize = 16, yguidefontsize = 16,
        xtickfontsize = 16, ytickfontsize = 16,
        legend=:topleft, legendfotsize = 16, 
        size = (1800, 900), 
        dpi = 300)
for j in 2:nr_species(P)
    plot!(plt, tt, map(x -> x[j], xx), label = P.𝒮[j], 
            linetype=:steppost, 
            linewidth = 5.0, 
            margin = 10Plots.mm, 
            xlabel = "t", 
            ylabel = "Counts", 
            xguidefontsize = 16, yguidefontsize = 16,
            xtickfontsize = 16, ytickfontsize = 16,
            legend=:topleft, legendfotsize = 16, 
            size = (1800, 900), 
            dpi = 300)
end
for k in (0*K+1):100:1*K
    plot!(plt, ttoo[k], extractcomp(xxoo[k], 2), label = false, linetype=:steppost, color = theme_palette(:auto)[2], alpha = 0.7)
    plot!(plt, ttoo[k], extractcomp(xxoo[k], 3), label = false, linetype=:steppost, color = theme_palette(:auto)[3], alpha = 0.7)
end
plot!(plt, times[label], extractcomp(xobs[label],2), seriestype = :scatter, markersize = 15, label = "Observation of MRNA")
plot!(plt, times[.!label], extractcomp(xobs[.!label],3), seriestype = :scatter, markersize = 15, label = "Observation of Protein")
plt
savefig(plt, "guided_process-15_partial_obs.png")

function iscorrect(GPP, tto, xxo )
    times = gett(GPP) ; obs = GPP.obs ; LL = getL(GPP)
    indices = [findfirst(t -> t==times[k], tto) for k in 1:n]
    simulated_values = map(k -> LL[k]*xxo[indices[k]], 1:n)
    out = [simulated_values[k] .- getv(obs[k]) == [0 for j in 1:getm(obs[k])] for k in 1:n]
    return all(out)
end

println("$(round(100*sum([iscorrect(GP,ttoo[k], xxoo[k]) for k in 1:K]/K), digits=7))% correct")



# Parameter inference
κ₂_range = collect(1.0:1.0:60.)
n2 = length(κ₂_range)
N = 10

K=1
mat = zeros(n2)
p = Progress(n2*K*N)
tta = [tt]
xxa = [xx]
a = [sum([ ℓ.λ(gett(o), getL(o)\getv(o) .+ [1,0,0])*ℓ.ξ*ℓ.ξ' for ℓ in GTT(κ₁, κ₂, dₘ,dₚ).ℛ ]) for o in partial_obs] 
for i in 1:n2
    P = GTT(κ₁, κ₂_range[i], dₘ, dₚ)
    GP = Guided_Process(partial_obs,a,P)
    info = filter_backward(GP)
    for k in 1:K
            tto, xxo = simulate_forward_monotone(x₀, GP, info)
            mat[i] += loglikelihood(tto, xxo, GP, info)/K
            next!(p)
            push!(tta,tto) ; push!(xxa, xxo)
    end
end
plt = plot(κ₂_range, mat, xlabel ="κ₂", ylabel = "log-likelihood", 
    title = "Estimated likelihood using $K samples with varying values of κ₂",
    label = false,
    margin = 12Plots.mm, size = (1800, 900))
for l in 2:N
    mat = zeros(n2)
    tta = [tt]
    xxa = [xx]
    a = [sum([ ℓ.λ(gett(o), getL(o)\getv(o) .+ [1,0,0])*ℓ.ξ*ℓ.ξ' for ℓ in GTT(κ₁, κ₂, dₘ,dₚ).ℛ ]) for o in partial_obs] 
    for i in 1:n2
        P = GTT(κ₁, κ₂_range[i], dₘ, dₚ)
        GP = Guided_Process(partial_obs,a,P)
        info = filter_backward(GP)
        for k in 1:K
                tto, xxo = simulate_forward_monotone(x₀, GP, info)
                mat[i] += loglikelihood(tto, xxo, GP, info)/K
                next!(p)
                push!(tta,tto) ; push!(xxa, xxo)
        end
    end
    plot!(plt, κ₂_range, mat, xlabel ="κ₂", ylabel = "log-likelihood", 
        title = "Estimated likelihood using $K samples with varying values of κ₂",
        label = false,
        margin = 12Plots.mm, size = (1800, 900))
end
plot!(plt,[κ₂], seriestype=:vline, label = "True value", linewidth = 5.0)
plt

function MH(obs, a₀, κ₀, σ, N)
    trace_κ = [κ₀] ; trace_a = [a₀]
    κ = κ₀ ; a = a₀
    ll = -1e10
    p = Progress(N)
    for j in 1:N
        κᵒ = κ + σ*randn()
        while κᵒ < 0.3
            κᵒ = κ + σ*randn()
        end
        Pᵒ = GTT(κ₁,κᵒ,dₚ,dₘ)
        # aᵒ = a₀
        aᵒ = [sum([ ℓ.λ(gett(o), getL(o)\getv(o)+[1.,0.,0.])*ℓ.ξ*ℓ.ξ' for ℓ in Pᵒ.ℛ ]) for o in obs]
        GPᵒ = Guided_Process(obs, aᵒ, Pᵒ)
        infoᵒ = filter_backward(GPᵒ)
        ttoᵒ, xxoᵒ = simulate_forward_monotone(x₀, GPᵒ, infoᵒ)
        llᵒ = loglikelihood(ttoᵒ, xxoᵒ, GPᵒ, infoᵒ)
        if log(rand()) <= llᵒ - ll
            ll = llᵒ
            tto = ttoᵒ ; xxo = xxoᵒ
            a = aᵒ ; κ = κᵒ
            P = Pᵒ ; GP  = GPᵒ ; info = infoᵒ
        end
        push!(trace_κ, κ)
        push!(trace_a , a)
        next!(p)
    end
    return trace_κ, trace_a
end

κκ, aa  = MH(partial_obs, a, 25.0, 5.0, 200)

plot(1:length(κκ), κκ , xlabel = "Iteration", ylabel = "Value", 
            label = "Trace of κ", margin = 12Plots.mm, 
            size = (1800, 900))
plot!(1:length(κκ), [κ₂ for i in 1:length(κκ)], label = "True value")