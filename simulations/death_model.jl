cd("/Users/marc/Documents/GitHub/CRMP.jl/")
using CRMP
using Plots
using ForwardDiff
using LinearAlgebra
using Bridge
using ProgressMeter
using Statistics
using Distributions
using Random
extractcomp(a, i) = map(x -> x[i], a)
Plots.scalefontsizes(2)

function plota(aa, πa1, cap, lab)
    plt = plot(aa[isless.(πa1, cap)], πa1[isless.(πa1, cap)], title = lab, label = false,
        color = theme_palette(:auto)[1],
        linewidth = 2.0, 
        margin = 10Plots.mm, 
        xlabel = "a", 
        ylabel = "Likelihood", 
        xguidefontsize = 16, yguidefontsize = 16,
        xtickfontsize = 16, ytickfontsize = 16,
        legend=:topright, legendfotsize = 16, 
        size = (1800, 1600), 
        dpi = 300)
    plt
end

"""
    Bridge, one (partial) observation
"""





# Forward simulation
c = 0.5


T = 1.0
x₀ = 50
c = 0.5
P = PureDeathProcess_constantrate(c) # Original process
tt, xx = simulate_forward(constant_rate(), x₀, T, P)
# plotprocess(tt, xx, P)

### Get quantiles and median
samples = [xx[end]]
K = 5000
prog = Progress(K)
for k in 1:K
    tt, xx = simulate_forward(constant_rate(), x₀, T, P)
    push!(samples, xx[end])
    next!(prog)
end

# histogram(samples, label = false)
L = 1.0
C = 0.0
x1 = Int64(quantile(samples, 0.01)) ; LNAobs1 = partial_observation(T, x1, L, C)
x50 = Int64(quantile(samples, 0.5)) ; LNAobs50 = partial_observation(T,x50,L,C)
x99 = Int64(quantile(samples, 0.99)) ; LNAobs99 = partial_observation(T,x99,L,C)

obs1 = partial_observation(T, x1, L, 1e-6)
obs50 = partial_observation(T, x50, L, 1e-6)
obs99 = partial_observation(T, x99, L, 1e-6)


# Checking correctness using differential equations LNAR forward guiding term
info =(nothing, nothing, nothing, nothing)

GP1_LNAR = LNAR_death(obs1, P)
GP50_LNAR = LNAR_death(obs50,P)
GP99_LNAR = LNAR_death(obs99,P)

GP1_diff = diff_death(obs1, 170, P)
GP50_diff = diff_death(obs50, 170, P)
GP99_diff = diff_death(obs99, 170, P)

tto1, xxo1 = simulate_forward_monotone(x₀, GP1_LNAR, info)
tto50, xxo50 = simulate_forward_monotone(x₀, GP50_LNAR, info)
tto99, xxo99 = simulate_forward_monotone(x₀, GP99_LNAR, info)

tto1_d, xxo1_d = simulate_forward_monotone(x₀, GP1_diff, info)
tto50_d, xxo50_d = simulate_forward_monotone(x₀, GP50_diff, info)
tto99_d, xxo99_d = simulate_forward_monotone(x₀, GP99_diff, info)   

ttoo1, xxoo1 = [tto1], [xxo1]
ttoo50, xxoo50 = [tto50], [xxo50]
ttoo99, xxoo99 = [tto99], [xxo99]

ttoo1_d, xxoo1_d = [tto1_d], [xxo1_d]
ttoo50_d, xxoo50_d = [tto50_d], [xxo50_d]
ttoo99_d, xxoo99_d = [tto99_d], [xxo99_d]

K = 1000
prog = Progress(K)
for k in 1:K
    tto1, xxo1 = simulate_forward_monotone(x₀, GP1_LNAR, info)
    tto50, xxo50 = simulate_forward_monotone(x₀, GP50_LNAR, info)
    tto99, xxo99 = simulate_forward_monotone(x₀, GP99_LNAR, info)

    tto1_d, xxo1_d = simulate_forward_monotone(x₀, GP1_diff, info)
    tto50_d, xxo50_d = simulate_forward_monotone(x₀, GP50_diff, info)
    tto99_d, xxo99_d = simulate_forward_monotone(x₀, GP99_diff, info)   

    push!(ttoo1_d, tto1_d) ; push!(xxoo1_d, xxo1_d)
    push!(ttoo50_d, tto50_d) ; push!(xxoo50_d, xxo50_d)
    push!(ttoo99_d, tto99_d) ; push!(xxoo99_d, xxo99_d)

    push!(ttoo1, tto1) ; push!(xxoo1, xxo1)
    push!(ttoo50, tto50) ; push!(xxoo50, xxo50)
    push!(ttoo99, tto99) ; push!(xxoo99, xxo99)

    next!(prog)
end

println("LNA guiding term:")
println("1% quantile:  $(round(100*sum([iscorrect_1obs(GP1_LNAR,ttoo1[k], xxoo1[k]) for k in 1:K]/K), digits=3))% correct. Expected: $(round(100*mean(samples.==x1), digits=3))%")
println("50% quantile: $(round(100*sum([iscorrect_1obs(GP50_LNAR,ttoo50[k], xxoo50[k]) for k in 1:K]/K), digits=3))% correct. Expected: $(round(100*mean(samples.==x50), digits=3))%")
println("99% quantile: $(round(100*sum([iscorrect_1obs(GP99_LNAR,ttoo99[k], xxoo99[k]) for k in 1:K]/K), digits=3))% correct. Expected: $(round(100*mean(samples.==x99), digits=3))%")
println("Diffusion guiding term:")
println("1% quantile:  $(round(100*sum([iscorrect_1obs(GP1_diff,ttoo1_d[k], xxoo1_d[k]) for k in 1:K]/K), digits=3))% correct. Expected: $(round(100*mean(samples.==x1), digits=3))%")
println("50% quantile: $(round(100*sum([iscorrect_1obs(GP50_diff,ttoo50_d[k], xxoo50_d[k]) for k in 1:K]/K), digits=3))% correct. Expected: $(round(100*mean(samples.==x50), digits=3))%")
println("99% quantile: $(round(100*sum([iscorrect_1obs(GP99_diff,ttoo99_d[k], xxoo99_d[k]) for k in 1:K]/K), digits=3))% correct. Expected: $(round(100*mean(samples.==x99), digits=3))%")


# Some plots
p1 = plot(ttoo1_d[1], xxoo1_d[1], label = "1% quantile",
    color = theme_palette(:auto)[1],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮, 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p2 = plot(ttoo50_d[1], xxoo50_d[1], label = "50% quantile",
    color = theme_palette(:auto)[2],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮, 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p3 = plot(ttoo99_d[1], xxoo99_d[1], label = "99% quantile",
    color = theme_palette(:auto)[3],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮, 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
for k in 1:100:K
    plot!(p1, ttoo1_d[k], xxoo1_d[k], color = theme_palette(:auto)[1], linetype = :steppost, label = false)
    plot!(p2, ttoo50_d[k], xxoo50_d[k], color = theme_palette(:auto)[2],linetype = :steppost, label = false)
    plot!(p3, ttoo99_d[k], xxoo99_d[k], color = theme_palette(:auto)[3],linetype = :steppost, label = false)
end
plot!(p1, [T], [x1], seriestype=:scatter, markersize = 10, label = false, color = theme_palette(:auto)[1])
plot!(p2, [T], [x50], seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[2])
plot!(p3, [T], [x99], seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[3])
plot(p1,p2,p3, layout = (3,1))




""" 
Choosing a tuning parameter in a diffusion guiding term
"""

aa = 5:1.0:500
m = length(aa)
N = 100
πa1 = zeros(m,N) ; correct1 = zeros(m)
πa50 = zeros(m,N) ; correct50 = zeros(m)
πa99 = zeros(m,N) ; correct99 = zeros(m)
p = Progress(m*N)
for i in 1:m
    for j in 1:N        
        GP1 = diff_death(obs1, aa[i], P) ; info1 = (0.,0.,0.,0.) #filter_backward(GP1)
        tto1, xxo1 = simulate_forward_monotone(x₀, GP1, info1)
        πa1[i,j] = likelihood_general_1obs(tto1,xxo1,GP1,info1)
        correct1[i] += iscorrect_1obs(GP1, tto1, xxo1)/N
        GP50 = diff_death(obs50, aa[i], P) ; info50 = info1 = (0.,0.,0.,0.) #ffilter_backward(GP50)
        tto50, xxo50 = simulate_forward_monotone(x₀, GP50, info50)
        πa50[i,j] = likelihood_general_1obs(tto50,xxo50,GP50,info50)
        correct50[i] += iscorrect_1obs(GP50, tto50, xxo50)/N
        GP99 = diff_death(obs99, aa[i], P) ; info99 = info1 = (0.,0.,0.,0.) #ffilter_backward(GP99)
        tto99, xxo99 = simulate_forward_monotone(x₀, GP99, info99)
        πa99[i,j] = likelihood_general_1obs(tto99,xxo99,GP99,info99)
        correct99[i] += iscorrect_1obs(GP99, tto99, xxo99)/N
        next!(p)
    end
end

πa1_avg = [mean(πa1[i,:]) for i in 1:m]
πa50_avg = [mean(πa50[i,:]) for i in 1:m]
πa99_avg = [mean(πa99[i,:]) for i in 1:m]

πa1_med = [median(πa1[i,:]) for i in 1:m]
πa50_med = [median(πa50[i,:]) for i in 1:m]
πa99_med = [median(πa99[i,:]) for i in 1:m]

plt1 = plota(aa, πa1_avg, 1e10, "1% quantile, T = $T, Avarege")
plt2 = plota(aa, πa1_med, 1e10, "1% quantile, T = $T, Median")
plot!(plt1, [c*x1], seriestype = :vline, label = false)
plot!(plt2, [c*x1], seriestype = :vline, label = false)
# plot(plt1,plt2,layout = (2,1))
plt3 = plota(aa, πa50_avg, 1e10, "50% quantile, T = $T, Avarege")
plt4 = plota(aa, πa50_med, 1e10, "50% quantile, T = $T, Median")
plot!(plt3, [c*x50], seriestype = :vline, label = false)
plot!(plt4, [c*x50], seriestype = :vline, label = false)
# plot(plt1,plt2,layout = (2,1))
plt5 = plota(aa, πa99_avg, 1e10, "99% quantile, T = $T, Avarege")
plt6 = plota(aa, πa99_med, 1e10, "99% quantile, T = $T, Median")
plot!(plt5, [c*x99], seriestype = :vline, label = false)
plot!(plt6, [c*x99], seriestype = :vline, label = false)
plot(plt1,plt2,plt3,plt4,plt5,plt6,layout = (3,2))
# savefig(fig, "death_model_choice_a.png")


plt1 = plot(aa, correct1, color = theme_palette(:auto)[1], linewidth = 2.0, 
        margin = 10Plots.mm, label = "1% quantile",
        xlabel = "a", 
        ylabel = "percentage correct", 
        xguidefontsize = 16, yguidefontsize = 16,
        xtickfontsize = 16, ytickfontsize = 16,
        legend=:topright, legendfotsize = 16, 
        size = (1800, 1600), 
        dpi = 300)
plt2 = plot(aa, correct50, color = theme_palette(:auto)[1], linewidth = 2.0, 
        margin = 10Plots.mm, label = "50% quantile",
        xlabel = "a", 
        ylabel = "percentage correct", 
        xguidefontsize = 16, yguidefontsize = 16,
        xtickfontsize = 16, ytickfontsize = 16,
        legend=:topright, legendfotsize = 16, 
        size = (1800, 1600), 
        dpi = 300)
plt3 = plot(aa, correct99, color = theme_palette(:auto)[1], linewidth = 2.0, 
        margin = 10Plots.mm, label = "99% quantile",
        xlabel = "a", 
        ylabel = "percentage correct", 
        xguidefontsize = 16, yguidefontsize = 16,
        xtickfontsize = 16, ytickfontsize = 16,
        legend=:topright, legendfotsize = 16, 
        size = (1800, 1600), 
        dpi = 300)
plot(plt1,plt2,plt3, layout = (3,1))






θθ = 5:0.1:40
m = length(θθ)
N = 100
πθ1 = zeros(m,N) ; correct1 = zeros(m)
πθ50 = zeros(m,N) ; correct50 = zeros(m)
πθ99 = zeros(m,N) ; correct99 = zeros(m)
p = Progress(m*N)
for i in 1:m
    for j in 1:N        
        obs1 = partial_observation_poisson(T,x1,[nothing, L], θθ[i]) ; GP1 = pois_death(obs1, P)
        tto1, xxo1 = simulate_forward_monotone(x₀, GP1, info)
        πθ1[i,j] = likelihood_general_1obs(tto1,xxo1,GP1,info)
        correct1[i] += iscorrect_1obs(GP1, tto1, xxo1)/N
        obs50 = partial_observation_poisson(T,x50,[nothing, L], θθ[i]) ; GP50 = pois_death(obs50, P)
        tto50, xxo50 = simulate_forward_monotone(x₀, GP50, info)
        πθ50[i,j] = likelihood_general_1obs(tto50,xxo50,GP50,info)
        correct50[i] += iscorrect_1obs(GP50, tto50, xxo50)/N
        obs99 = partial_observation_poisson(T,x99,[nothing, L], θθ[i]) ; GP99 = pois_death(obs99, P)
        tto99, xxo99 = simulate_forward_monotone(x₀, GP99, info)
        πθ99[i,j] = likelihood_general_1obs(tto99,xxo99,GP99,info)
        correct99[i] += iscorrect_1obs(GP99, tto99, xxo99)/N
        next!(p)
    end
end

πθ1_avg = [mean(πθ1[i,:]) for i in 1:m]
πθ50_avg = [mean(πθ50[i,:]) for i in 1:m]
πθ99_avg = [mean(πθ99[i,:]) for i in 1:m]

πθ1_med = [median(πθ1[i,:]) for i in 1:m]
πθ50_med = [median(πθ50[i,:]) for i in 1:m]
πθ99_med = [median(πθ99[i,:]) for i in 1:m]

plt1 = plota(θθ, πθ1_avg, 1e10, "1% quantile, T = $T, Avarege")
plt2 = plota(θθ, πθ1_med, 1e10, "1% quantile, T = $T, Median")
plot!(plt1, [c*x1], seriestype = :vline, label = false)
plot!(plt2, [c*x1], seriestype = :vline, label = false)
# plot(plt1,plt2,layout = (2,1))
plt3 = plota(θθ, πθ50_avg, 1e10, "50% quantile, T = $T, Avarege")
plt4 = plota(θθ, πθ50_med, 1e10, "50% quantile, T = $T, Median")
plot!(plt3, [c*x50], seriestype = :vline, label = false)
plot!(plt4, [c*x50], seriestype = :vline, label = false)
# plot(plt1,plt2,layout = (2,1))
plt5 = plota(θθ, πθ99_avg, 1e10, "99% quantile, T = $T, Avarege")
plt6 = plota(θθ, πθ99_med, 1e10, "99% quantile, T = $T, Median")
plot!(plt5, [c*x99], seriestype = :vline, label = false)
plot!(plt6, [c*x99], seriestype = :vline, label = false)
plot(plt1,plt2,plt3,plt4,plt5,plt6,layout = (3,2))
# savefig(fig, "death_model_choice_a.png")


plt1 = plot(θθ, correct1, color = theme_palette(:auto)[1], linewidth = 2.0, 
        margin = 10Plots.mm, label = "1% quantile",
        xlabel = "θ", 
        ylabel = "percentage correct", 
        xguidefontsize = 16, yguidefontsize = 16,
        xtickfontsize = 16, ytickfontsize = 16,
        legend=:topright, legendfotsize = 16, 
        size = (1800, 1600), 
        dpi = 300)
plt2 = plot(θθ, correct50, color = theme_palette(:auto)[1], linewidth = 2.0, 
        margin = 10Plots.mm, label = "50% quantile",
        xlabel = "θ", 
        ylabel = "percentage correct", 
        xguidefontsize = 16, yguidefontsize = 16,
        xtickfontsize = 16, ytickfontsize = 16,
        legend=:topright, legendfotsize = 16, 
        size = (1800, 1600), 
        dpi = 300)
plt3 = plot(θθ, correct99, color = theme_palette(:auto)[1], linewidth = 2.0, 
        margin = 10Plots.mm, label = "99% quantile",
        xlabel = "θ", 
        ylabel = "percentage correct", 
        xguidefontsize = 16, yguidefontsize = 16,
        xtickfontsize = 16, ytickfontsize = 16,
        legend=:topright, legendfotsize = 16, 
        size = (1800, 1600), 
        dpi = 300)
plot(plt1,plt2,plt3, layout = (3,1))





"""
Sherlock & Golightly forward simulation algorithm comparison
"""

# function simulate_forward_SG(x₀, type::TP) where {TP<:Union{LNA, pois}}
#     t, x = 0.0, x₀ 
#     tt, xx = [t], [x]
#     ℓ = type.P.ℛ

#     while t < gett(type)
#         dt = -log(rand())/(ℓ.λ(t,x)*guiding_term(type)(ℓ,t,x))
#         t = t + dt
#         x = x - 1
#         push!(xx, x)
#         push!(tt, t)
#     end
#     tt = vcat(tt[1:end-1], T)
#     xx[end] = xx[end-1]
#     return tt, xx
# end

# C = 0.0
# P1_SG = condition_process(P, guiding_term(LNAR(L,x1,T,C,P)))
# P50_SG = condition_process(P, guiding_term(LNAR(L,x50,T,C,P)))
# P99_SG = condition_process(P, guiding_term(LNAR(L,x99,T,C,P)))


# tto1, xxo1 = simulate_forward(constant_rate(), x₀, T, P1_SG)
# tto50, xxo50 = simulate_forward(constant_rate(), x₀, T, P50_SG)
# tto99, xxo99 = simulate_forward(constant_rate(), x₀, T, P99_SG)

# ttoo1, xxoo1 = [tto1], [xxo1]
# ttoo50, xxoo50 = [tto50], [xxo50]
# ttoo99, xxoo99 = [tto99], [xxo99]
# K = 1000
# prog = Progress(K)
# for k in 1:K
#     tto1, xxo1 = simulate_forward(constant_rate(), x₀, T, P1_SG)
#     tto50, xxo50 = simulate_forward(constant_rate(), x₀, T, P50_SG)
#     tto99, xxo99 = simulate_forward(constant_rate(), x₀, T, P99_SG) 
#     push!(ttoo1, tto1) ; push!(xxoo1, xxo1)
#     push!(ttoo50, tto50) ; push!(xxoo50, xxo50)
#     push!(ttoo99, tto99) ; push!(xxoo99, xxo99)
#     next!(prog)
# end


# println("$(round(100*sum([iscorrect_1obs(P1,ttoo1[k], xxoo1[k]) for k in 1:K]/K), digits=3))% correct. Expected: $(round(100*mean(samples.==x1), digits=3))%")
# println("$(round(100*sum([iscorrect_1obs(P50,ttoo50[k], xxoo50[k]) for k in 1:K]/K), digits=3))% correct. Expected: $(round(100*mean(samples.==x50), digits=3))%")
# println("$(round(100*sum([iscorrect_1obs(P99,ttoo99[k], xxoo99[k]) for k in 1:K]/K), digits=3))% correct. Expected: $(round(100*mean(samples.==x99), digits=3))%")

# p1 = plot(ttoo1[1], xxoo1[1], label = "1% quantile",
#     color = theme_palette(:auto)[1],
#     linetype=:steppost, 
#     linewidth = 2.0, 
#     margin = 10Plots.mm, 
#     xlabel = "t", 
#     ylabel = P.𝒮, 
#     xguidefontsize = 16, yguidefontsize = 16,
#     xtickfontsize = 16, ytickfontsize = 16,
#     legend=:topright, legendfotsize = 16, 
#     size = (1800, 1600), 
#     dpi = 300) # Original process
# p2 = plot(ttoo50[1], xxoo50[1], label = "50% quantile",
#     color = theme_palette(:auto)[2],
#     linetype=:steppost, 
#     linewidth = 2.0, 
#     margin = 10Plots.mm, 
#     xlabel = "t", 
#     ylabel = P.𝒮, 
#     xguidefontsize = 16, yguidefontsize = 16,
#     xtickfontsize = 16, ytickfontsize = 16,
#     legend=:topright, legendfotsize = 16, 
#     size = (1800, 1600), 
#     dpi = 300) # Original process
# p3 = plot(ttoo99[1], xxoo99[1], label = "99% quantile",
#     color = theme_palette(:auto)[3],
#     linetype=:steppost, 
#     linewidth = 2.0, 
#     margin = 10Plots.mm, 
#     xlabel = "t", 
#     ylabel = P.𝒮, 
#     xguidefontsize = 16, yguidefontsize = 16,
#     xtickfontsize = 16, ytickfontsize = 16,
#     legend=:topright, legendfotsize = 16, 
#     size = (1800, 1600), 
#     dpi = 300) # Original process
# for k in 1:100:K
#     plot!(p1, ttoo1[k], xxoo1[k], color = theme_palette(:auto)[1], linetype = :steppost, label = false)
#     plot!(p2, ttoo50[k], xxoo50[k], color = theme_palette(:auto)[2],linetype = :steppost, label = false)
#     plot!(p3, ttoo99[k], xxoo99[k], color = theme_palette(:auto)[3],linetype = :steppost, label = false)
# end
# plot!(p1, [T], [x1], seriestype=:scatter, markersize = 10, label = false, color = theme_palette(:auto)[1])
# plot!(p2, [T], [x50], seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[2])
# plot!(p3, [T], [x99], seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[3])
# plot(p1,p2,p3, layout = (3,1))





obs50 = partial_observation(T, x50, L, 1e-5)
a50 = (x₀-x50)/T
GP50 = Guided_Process(obs50, a50, P)
info50 = filter_backward(GP50)



val = x50+5
taxis = collect(0:0.1:0.95*T)
p1 = plot(taxis, map(t->guiding_term(P50)(P.ℛ,t,val), taxis), 
    label = "x = $val", ylim = (0.0, 2.5),
    color = theme_palette(:auto)[1],
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = "LNA guiding term", 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300)
plot!(p1, taxis, map(t->guiding_term(P50)(P.ℛ,t,val-1),taxis), label = "x = $(val-1)")
plot!(p1, taxis, map(t->guiding_term(P50)(P.ℛ,t,val-2), taxis), label = "x = $(val-2)")
plot!(p1, taxis, map(t->guiding_term(P50)(P.ℛ,t,val-3), taxis), label = "x = $(val-3)")
plot!(p1, taxis, map(t->guiding_term(P50)(P.ℛ,t,val-4), taxis), label = "x = $(val-4)")
plot!(p1, taxis, map(t->guiding_term(P50)(P.ℛ,t,val-5), taxis), label = "x = $(val-5)")

p2 = plot(taxis, map(t->CRMP.guiding_term(info50,GP50)(P.ℛ,t,val), taxis), 
    label = "x = $val", ylim = (0.0, 1.5),
    color = theme_palette(:auto)[1],
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = "Diffusion guiding term", 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300)
plot!(p2, taxis, map(t->CRMP.guiding_term(info50,GP50)(P.ℛ,t,val-1), taxis), label = "x = $(val-1)")
plot!(p2, taxis, map(t->CRMP.guiding_term(info50,GP50)(P.ℛ,t,val-2), taxis), label = "x = $(val-2)")
plot!(p2, taxis, map(t->CRMP.guiding_term(info50,GP50)(P.ℛ,t,val-3), taxis), label = "x = $(val-3)")
plot!(p2, taxis, map(t->CRMP.guiding_term(info50,GP50)(P.ℛ,t,val-4), taxis), label = "x = $(val-4)")
plot!(p2, taxis, map(t->CRMP.guiding_term(info50,GP50)(P.ℛ,t,val-5), taxis), label = "x = $(val-5)")

obs50 = partial_observation_poisson(T, x50, [nothing,L], 1e-5)
GP50 = Guided_Process_Poisson(obs50, a50, P)
a50 = (x₀-x50)/T

p3 = plot(taxis, map(t->CRMP.guiding_term(info50,GP50)(P.ℛ,t,val), taxis), 
    label = "x = $val", ylim = (0.0, 1.0),
    color = theme_palette(:auto)[1],
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = "Poisson guiding term", 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300)
plot!(p3, taxis, map(t->CRMP.guiding_term(info50,GP50)(P.ℛ,t,val-1), taxis), label = "x = $(val-1)")
plot!(p3, taxis, map(t->CRMP.guiding_term(info50,GP50)(P.ℛ,t,val-2), taxis), label = "x = $(val-2)")
plot!(p3, taxis, map(t->CRMP.guiding_term(info50,GP50)(P.ℛ,t,val-3), taxis), label = "x = $(val-3)")
plot!(p3, taxis, map(t->CRMP.guiding_term(info50,GP50)(P.ℛ,t,val-4), taxis), label = "x = $(val-4)")
plot!(p3, taxis, map(t->CRMP.guiding_term(info50,GP50)(P.ℛ,t,val-5), taxis), label = "x = $(val-5)")


plt = plot(p1,p2,p3, layout = (3,1))
plt
# savefig(plt, "guiding_terms.png")


"""
    Comparing guiding terms
"""

val = x50+5
taxis = collect(0:0.01:0.95*T)
p2 = plot(taxis, map(t->guiding_term(info, GP50_LNAR)(P.ℛ,t,val), taxis), 
    label = "x = $val", ylim = (0.0, 2.5),
    color = theme_palette(:auto)[1],
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = "LNA guiding term", 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300)
plot!(p2, taxis, map(t->guiding_term(info, GP50_LNAR)(P.ℛ,t,val-1),taxis), label = "x = $(val-1)")
plot!(p2, taxis, map(t->guiding_term(info, GP50_LNAR)(P.ℛ,t,val-2), taxis), label = "x = $(val-2)")
plot!(p2, taxis, map(t->guiding_term(info, GP50_LNAR)(P.ℛ,t,val-3), taxis), label = "x = $(val-3)")
plot!(p2, taxis, map(t->guiding_term(info, GP50_LNAR)(P.ℛ,t,val-4), taxis), label = "x = $(val-4)")
plot!(p2, taxis, map(t->guiding_term(info, GP50_LNAR)(P.ℛ,t,val-5), taxis), label = "x = $(val-5)")
p2

































"""
Proof of concept for death process
"""
# Given a death process, x₀ - X(T) ∣ X(0) = x₀ ~ Pois(cT). 
# Let q(v) = (cT)^{x₀-v}/(x₀-v)! exp(-cT) be the distribution of X(T)∣ X(0)=x₀
Random.seed!(61)
T = 1.0
x₀ = 50
c = 0.5
P = PureDeathProcess_constantrate(c) # Original process
info = (nothing, nothing, nothing, nothing)
tt, xx = simulate_forward(constant_rate(), x₀, T, P)
q(v) = pdf(Binomial(x₀, exp(-c*T)), v)
draw_q() = rand(Binomial(x₀, exp(-c*T)))

# plotprocess(tt, xx, P)

L = 1.0
C = 1e-5

nr_draws_q = 2000           # nr of times we draw an endpoint
nr_processes = 500          # nr of processes we sample per endpoint to compute likelihood
v_samples = [draw_q() for i in 1:50000]
vals = sort(unique(v_samples)) # should be all possible endpoints


guided_estimates_LNA, guided_estimates_diff, guided_estimates_P = [], [], []
SG_estimates_δ, SG_estimates_g = [], []
true_vals = []
estimates_δ_LNA, estimates_δ_diff, estimates_δ_P = [], [], []
N = length(unique(v_samples))

ttoo_δ_LNA, xxoo_δ_LNA = [], []
# ttoo_g_LNA, xxoo_g_LNA = [], []
ttoo_δ_diff, xxoo_δ_diff = [], []
# ttoo_g_diff, xxoo_g_diff = [], []
ttoo_δ_P, xxoo_δ_P = [], []
# ttoo_g_P, xxoo_g_P = [], []
v_values = []
proposal_samples = []

prog = Progress(Int64(ceil(nr_processes*N*nr_draws_q*0.040975)))

# A likelihood for each draw in each endpoint
likelihood_δ_LNA = zeros(nr_draws_q, N)
# likelihood_g_LNA = zeros(nr_draws_q, N)
likelihood_δ_LNA_SG = zeros(nr_draws_q,N)
# likelihood_g_LNA_SG = zeros(nr_draws_q,N)

likelihood_δ_diff = zeros(nr_draws_q, N)
# likelihood_g_diff  = zeros(nr_draws_q, N)
likelihood_δ_P = zeros(nr_draws_q, N)
# likelihood_g_P = zeros(nr_draws_q, N)

total_trajs = zeros(Int64, N)
correct_LNA = zeros(Int64,N)
correct_diff = zeros(Int64, N)
correct_P = zeros(Int64,N)
# correct_g = zeros(Int64, N)

for (i,n) in enumerate(vals)
    # sample_q_prop() # draw_q()
    obs = partial_observation(T, n, L, C) ; obsP = partial_observation_poisson(T,n,[nothing, L], c*n)
    GP_LNAR = LNAR_death(obs, P)
    GP_diff = diff_death(obs, 2.5*(x₀-n), P)
    GP_P = pois_death(obsP, P)
    # info_diff = filter_backward(GP_diff)
    # info_P = (nothing,nothing,nothing,nothing) #filter_backward(GP_P)

    for k in 1:nr_draws_q
        V = draw_q()
        if V == n
            for j in 1:nr_processes
                tto_δ_LNA, xxo_δ_LNA = simulate_forward_monotone(x₀, GP_LNAR, info)
                # tto_g_LNA, xxo_g_LNA = CRMP.simulate_forward(Gillespie(), x₀, T, condition_process(P, guiding_term(GP_LNAR)))
                
                tto_δ_diff, xxo_δ_diff = simulate_forward_monotone(x₀, GP_diff, info)
                # tto_g_diff, xxo_g_diff = CRMP.simulate_forward(Gillespie(), x₀,T,condition_process(P, guiding_term( GP_diff)))

                tto_δ_P, xxo_δ_P = simulate_forward_monotone(x₀, GP_P, info)
                # tto_g_P, xxo_g_P = CRMP.simulate_forward(Gillespie(), x₀,T,condition_process(P, guiding_term(GP_P)))

                likelihood_δ_LNA[k,i] += likelihood_general_1obs(tto_δ_LNA, xxo_δ_LNA, GP_LNAR, info)/nr_processes
                # likelihood_g_LNA[k,i] += exp(loglikelihood_direct(tto_g_LNA, xxo_g_LNA, GP_LNAR))/nr_processes
                
                # likelihood_δ_LNA_SG[k,i] += exp(loglikelihood_SG(tto_δ_LNA, xxo_δ_LNA, GP_LNAR))/nr_processes
                # likelihood_g_LNA_SG[k,i] += exp(loglikelihood_SG(tto_g_LNA, xxo_g_LNA, GP_LNAR))/nr_processes            

                likelihood_δ_diff[k,i] += likelihood_general_1obs(tto_δ_diff, xxo_δ_diff, GP_diff, info)/nr_processes
                # likelihood_g_diff[k,i] += exp(loglikelihood_direct(tto_g_diff, xxo_g_diff, GP_diff))/nr_processes

                likelihood_δ_P[k,i] += likelihood_general_1obs(tto_δ_P, xxo_δ_P, GP_P, info)/nr_processes
                # likelihood_g_P[k,i] += exp(loglikelihood_direct(tto_g_P, xxo_g_P, GP_P))/nr_processes
                    
                # push!(ttoo_g_LNA, tto_g_LNA) ; push!(xxoo_g_LNA, xxo_g_LNA)
                push!(ttoo_δ_LNA, tto_δ_LNA) ; push!(xxoo_δ_LNA, xxo_δ_LNA)
                # push!(ttoo_g_diff, tto_g_diff) ; push!(xxoo_g_diff, xxo_g_diff)
                push!(ttoo_δ_diff, tto_δ_diff) ; push!(xxoo_δ_diff, xxo_δ_diff)
                # push!(ttoo_g_P, tto_g_P) ; push!(xxoo_g_P, xxo_g_P)
                push!(ttoo_δ_P, tto_δ_P) ; push!(xxoo_δ_P, xxo_δ_P)
                push!(v_values, V)
                next!(prog)
                correct_LNA[i] += iscorrect_1obs(GP_LNAR, tto_δ_LNA, xxo_δ_LNA)
                correct_diff[i] += iscorrect_1obs(GP_diff, tto_δ_diff, xxo_δ_diff)
                correct_P[i] += iscorrect_1obs(GP_P, tto_δ_P, xxo_δ_P)
                # correct_g[i] += iscorrect_1obs(GP_LNAR, tto_g_LNA, xxo_g_LNA)
                total_trajs[i] += 1
            end
        end
    end
    # push!(guided_estimates_LNA, mean(likelihood_g_LNA[:,i])/q(n))
    push!(estimates_δ_LNA, mean(likelihood_δ_LNA[:,i])/q(n))
    # push!(SG_estimates_δ, mean(likelihood_δ_LNA_SG[:,i])/q(n))
    #  push!(SG_estimates_g, mean(likelihood_g_LNA_SG[:,i])/q(n))

    # push!(guided_estimates_diff, mean(likelihood_g_diff[:,i])/q(n))
    push!(estimates_δ_diff, mean(likelihood_δ_diff[:,i])/q(n))
    # push!(guided_estimates_P, mean(likelihood_g_P[:,i])/q(n))
    push!(estimates_δ_P, mean(likelihood_δ_P[:,i])/q(n))
end


# guided_estimates_LNA, estimates_δ_LNA, SG_estimates_δ, SG_estimates_g, likelihood_g_LNA, ttoo_δ_LNA, xxoo_δ_LNA, ttoo_g_LNA, xxoo_g_LNA, v_values= do_stuff(nr_draws_q,nr_processes)

# map(xx -> xx[end], xxoo_g)

# # V = draw_q()
# # tto_δ, xxo_δ = simulate_forward_monotone(x₀, LNAR_death(L,V,T,C,P))
# # tto_g, xxo_g = CRMP.simulate_forward(Gillespie(), x₀, T, condition_process(P, guiding_term(GP_LNAR)))
# m_g, m_δ = zeros(1000), zeros(1000)
# kk = []
# for i in 1:1000
#     k = rand(1:1:length(21:1:40)*nr_draws_q)
#     m_g[i] += exp(loglikelihood_direct(ttoo_g_LNA[k],xxoo_g_LNA[k], LNAR_death(v = v_values[k], P = P)))/1000
#     m_δ[i] += exp(loglikelihood_direct(ttoo_δ_LNA[k],xxoo_δ_LNA[k], LNAR_death(v = v_values[k], P = P)))/1000
#     push!(kk,k)
# end
# println(mean(m_g)) ; println(mean(m_δ))

# k = kk[findall(x -> isinf(x), m_g)[11]]#rand(1:1:length(21:1:40)*nr_draws_q)
# p1 = plotprocess(ttoo_g_LNA[k],xxoo_g_LNA[k],P)
# p2 = plotprocess(ttoo_δ_LNA[k],xxoo_δ_LNA[k],P)
# plot!(p1, [T], [v_values[k]], seriestype = :scatter, markersize = 10, label = "v", legend = :topright)
# plot!(p2, [T], [v_values[k]], seriestype=:scatter, markersize = 10, label = "v", legend = :topright)
# println(loglikelihood_direct(ttoo_g_LNA[k],xxoo_g_LNA[k], LNAR_death(v = v_values[k],P=P)))
# println(loglikelihood_direct(ttoo_δ_LNA[k],xxoo_δ_LNA[k], LNAR_death(v = v_values[k], P=P)))
# plot(p1,p2,layout=(2,1), plot_title = "Upper: SG, lower: δ")




# loglikelihood_SG(ttoo_g[2], xxoo_g[2], GP_LNAR)

# p1 = bar(vals, q.(vals), size = (1800,1600), normalize = true, label = "q", legend = :topleft)
# # p2 = bar(vals, guided_estimates_LNA, size = (1800,1600),label ="guided estimates, SG forward simulation", legend=:topleft)
# p3 = bar(vals,estimates_δ_LNA, size = (1800,1600), label = "guided estimates, δ forward simulation",  legend = :topleft)
# # p4 = bar(vals, q.(vals), size = (1800,1600), normalize = true, label = "q", legend = :topleft)
# # p5 = bar(vals,SG_estimates_g, size = (1800,1600), label = "guided estimates, SG forward simulation" ,legend = :topleft)
# p6 = bar(vals,SG_estimates_δ, size = (1800,1600), label = "guided estimates, δ forward simulation, SG",  legend = :topleft)
# plot(p1,p2,p5,p3,p6,layout = (3,2), plot_title = "LNA guiding term")

p1 = bar(vals[1:end], correct_LNA[1:end]./total_trajs[1:end], 
            size = (1800,1600), label ="LNA", legend=:topright, yaxis = (0.0,1.0),
            xlabel = "k", ylabel = "percentage correct", margin = 10Plots.mm)
p2 = bar(vals[1:end], correct_diff[1:end]./total_trajs[1:end], yaxis = (0.0,1.0),
            size = (1800,1600), label ="Diffusion guiding term", legend=:topright, 
            xlabel = "k", ylabel = "percentage correct", margin = 10Plots.mm)
p3 = bar(vals[1:end], correct_P[1:end]./total_trajs[1:end], yaxis = (0.0,1.0),
            size = (1800,1600), label ="Poisson guidng term", legend=:topright, 
            xlabel = "k", ylabel = "percentage correct", margin = 10Plots.mm)
plt = plot(p1,p2, p3, layout = (3,1))
savefig(plt, "perc_correct.png")



# p1 = bar(vals, q.(vals), size = (1800,1600), normalize = true, label = "q", legend = :topleft)
# p2 = bar(vals, guided_estimates_diff, label ="guided estimates, SG forward simulation", legend=:topleft)
# p3 = bar(vals, estimates_δ_diff, size = (1800,1600), label = "guided estimates, δ forward simulation", legend = :topleft)
# plot(p1,p2,p3,layout = (3,1), plot_title = "Diffusion guiding term")

# p1 = bar(vals, q.(vals), size = (1800,1600), normalize = true, label = "q", legend = :topleft)
# p2 = bar(vals, guided_estimates_P, label ="guided estimates, SG forward simulation", bar_width=1.0, legend=:topleft)
# p3 = bar(vals, estimates_δ_P, size = (1800,1600), label = "guided estimates, δ forward simulation", bar_width=1.0, legend = :topleft)
# plot(p1,p2,p3,layout = (3,1), plot_title = "Poisson guiding term")


p1 = bar(vals, q.(vals), size = (1800,1600), normalize = true, label = "q", legend = :topleft, xlabel = "k", ylabel = "q(k)")
p2 = bar(vals,estimates_δ_LNA, size = (1800,1600), label = "LNA guiding term",  legend = :topleft, xlabel = "k", ylabel = "p̂(k)")
p3 = bar(vals, estimates_δ_diff, size = (1800,1600), label = "Diffusion guiding term", legend = :topleft, xlabel = "k", ylabel = "p̂(k)")
p4 = bar(vals, estimates_δ_P, size = (1800,1600), label = "Poisson guiding term", legend = :topleft, xlabel = "k", ylabel = "p̂(k)")
plt = plot(p1,p2,p3,p4, layout = (4,1))
savefig(plt, "density_estimates.png")

p1 = plot(vals, estimates_δ_LNA, linetype = :steppost, 
            size = (1800,1600), label = "LNA guiding term",  margin = 10Plots.mm,
            legend = :topleft, xlabel = "k", ylabel = "p̂(k)", linewidth = 5.5)
plot!(p1, vals, q.(vals), linetype  = :steppost, label = "q", linestyle = :dash, linewidth = 2.5)
plot!(p1, vals, estimates_δ_LNA .- q.(vals), label = "difference" , linetype = :steppost, fillrange = 0.0, fillalpha = 0.7)
p2 = plot(vals, estimates_δ_diff, linetype = :steppost, 
            size = (1800,1600), label = "Diffusion guiding term",  margin = 10Plots.mm,
            legend = :topleft, xlabel = "k", ylabel = "p̂(k)", linewidth = 5.5)
plot!(p2, vals, q.(vals), linetype  = :steppost, label = "q", linestyle = :dash, linewidth = 2.5)
plot!(p2, vals, estimates_δ_diff .- q.(vals), label = "difference" , linetype = :steppost, fillrange = 0.0, fillalpha = 0.7)
p3 = plot(vals, estimates_δ_P, linetype = :steppost, 
            size = (1800,1600), label = "Poisson guiding term",  margin = 10Plots.mm,
            legend = :topleft, xlabel = "k", ylabel = "p̂(k)", linewidth = 5.5)
plot!(p3, vals, q.(vals), linetype  = :steppost, label = "q", linestyle = :dash, linewidth = 2.5)
plot!(p3, vals, estimates_δ_P .- q.(vals), label = "difference" , linetype = :steppost, fillrange = 0.0, fillalpha = 0.7)
plt = plot(p1,p2,p3,layout = (3,1))
savefig(plt, "density_estimates.png")


ssq_error_LNA = sum((q.(vals) .- estimates_δ_LNA).^2)
ssq_error_diff = sum((q.(vals) .- estimates_δ_diff).^2)
ssq_error_P = sum((q.(vals) .- estimates_δ_P).^2)
println("squared error LNA: $(round(sum((q.(vals) .- estimates_δ_LNA).^2), digits = 5))")
println("squared error diff: $(round(sum((q.(vals) .- estimates_δ_diff).^2), digits = 5))")
println("squared error poisson: $(round(sum((q.(vals) .- estimates_δ_P).^2), digits = 5))")

sum(map(x -> x == 0.0, likelihood_g_LNA[:,1]))



sum(guided_estimates_LNA)






"""
    Computing effective sample sizes
"""


GP_LNAR1 = LNAR_death(v = x1, P=P)
GP_LNAR50 = LNAR_death(v = x50, P=P)
GP_LNAR99 = LNAR_death(v = x99, P=P)


GP_diff1 = diff_death(v = x1, a =  β(1.0,x1,P), C = 1e-8, P = P)
GP_diff50 = diff_death(v = x50, a =  β(1.0, x50, P), C = 1e-8, P = P)
GP_diff99 = diff_death(v = x99, a =  β(1.0,x99,P), C = 1e-8, P = P)

GP_pois1 = pois_death(1.0 ,x1, T, 0.5(x₀-x1)/T , P )
GP_pois50 = pois_death(1.0 ,x50, T, 0.5(x₀-x50)/T , P )
GP_pois99 = pois_death(1.0 ,x99, T, 0.5(x₀-x99)/T , P )






function get_likelihoods(m,N, type)
    hhat_δ = zeros(m,N)
    hhat_SG = zeros(m,N)

    prog = Progress(m*N)
    for i in 1:m
        for j in 1:N
            tto_δ, xxo_δ = simulate_forward_monotone(x₀, type)
            tto_SG, xxo_SG = simulate_forward_SG(x₀, type)
            hhat_δ[i,j] = exp(loglikelihood_direct(tto_δ, xxo_δ, type))
            hhat_SG[i,j] = exp(loglikelihood_SG(tto_SG, xxo_SG, type)) 
            next!(prog)       
        end
    end
    return hhat_δ, hhat_SG
end

N = 10
m = 5000

hhat_δ_LNA1, hhat_SG_LNA1 = get_likelihoods(m,N,GP_LNAR1)
hhat_δ_LNA50, hhat_SG_LNA50 = get_likelihoods(m,N,GP_LNAR50)
hhat_δ_LNA99, hhat_SG_LNA99 = get_likelihoods(m,N,GP_LNAR99)

hhat_δ_diff1, hhat_SG_diff1 = get_likelihoods(m,N,GP_diff1)
hhat_δ_diff50, hhat_SG_diff50 = get_likelihoods(m,N,GP_diff50)
hhat_δ_diff99, hhat_SG_diff99 = get_likelihoods(m,N,GP_diff99)

hhat_δ_pois1, hhat_SG_pois1 = get_likelihoods(m,N,GP_pois1)
hhat_δ_pois50, hhat_SG_pois50 = get_likelihoods(m,N,GP_pois50)
hhat_δ_pois99, hhat_SG_pois99 = get_likelihoods(m,N,GP_pois99)

getll(ll) = [maximum(ll[i,:]) + log(mean(exp.(ll[i,:] .- maximum(ll[i,:])))) for i in eachindex(ll[:,1])]
# sum(isnan.([maximum(hhat_SG_LNA1[k,:]) + log(mean(exp.(hhat_SG_LNA1[k,:] .- maximum(hhat_SG_LNA1[k,:])))) for k in 1:m]))


hhat_δ_LNA1_means = [mean(hhat_δ_LNA1[i,:]) for i in 1:m]
hhat_SG_LNA1_means = [mean(hhat_SG_LNA1[i,:]) for i in 1:m]
hhat_δ_LNA50_means = [mean(hhat_δ_LNA50[i,:]) for i in 1:m]
hhat_SG_LNA50_means = [mean(hhat_SG_LNA50[i,:]) for i in 1:m]
hhat_δ_LNA99_means = [mean(hhat_δ_LNA99[i,:]) for i in 1:m]
hhat_SG_LNA99_means = [mean(hhat_SG_LNA99[i,:]) for i in 1:m]

hhat_δ_LNA1_ll = getll(hhat_δ_LNA1)
hhat_SG_LNA1_ll = getll(hhat_SG_LNA1)
hhat_δ_LNA50_ll = getll(hhat_δ_LNA50)
hhat_SG_LNA50_ll = getll(hhat_SG_LNA50)
hhat_δ_LNA99_ll = getll(hhat_δ_LNA99)
hhat_SG_LNA99_ll = getll(hhat_SG_LNA99)

hhat_δ_LNA1_means = exp.(hhat_δ_LNA1_ll)
hhat_SG_LNA1_means = exp.(hhat_SG_LNA1_ll)
hhat_δ_LNA50_means = exp.(hhat_δ_LNA50_ll)
hhat_SG_LNA50_means = exp.(hhat_SG_LNA50_ll)
hhat_δ_LNA99_means = exp.(hhat_δ_LNA99_ll)
hhat_SG_LNA99_means = exp.(hhat_SG_LNA99_ll)

plt1 = histogram(hhat_δ_LNA1_ll, xguidefontsize = 16, yguidefontsize = 16, label = "1%, δ",
                xtickfontsize = 16, ytickfontsize = 16,size = (1800, 1600), dpi = 300)
plt2 = histogram(hhat_SG_LNA1_ll[hhat_SG_LNA1_ll .> -10], xguidefontsize = 16, yguidefontsize = 16, label = "1%, SG",
                xtickfontsize = 16, ytickfontsize = 16, size = (1800, 1600), dpi = 300)
plt3 = histogram(hhat_δ_LNA50_ll, xguidefontsize = 16, yguidefontsize = 16, label = "50%, δ",
                xtickfontsize = 16, ytickfontsize = 16, size = (1800, 1600), dpi = 300)
plt4 = histogram(hhat_SG_LNA50_ll[hhat_SG_LNA50_ll .> -10], xguidefontsize = 16, yguidefontsize = 16, label = "50%, SG",
                xtickfontsize = 16, ytickfontsize = 16, size = (1800, 1600), dpi = 300)
plt5 = histogram(hhat_δ_LNA99_ll, xguidefontsize = 16, yguidefontsize = 16, label = "99%, δ",
                xtickfontsize = 16, ytickfontsize = 16, size = (1800, 1600), dpi = 300)
plt6 = histogram(hhat_SG_LNA99_ll[hhat_SG_LNA99_ll .> -10], xguidefontsize = 16, yguidefontsize = 16, label = "99%, SG",
                xtickfontsize = 16, ytickfontsize = 16, size = (1800, 1600), dpi = 300)
plot(plt1,plt2,plt3,plt4,plt5,plt6, layout = (3,2))


hhat_δ_diff1_means = [mean(hhat_δ_diff1[i,:]) for i in 1:m]
hhat_SG_diff1_means = [mean(hhat_SG_diff1[i,:]) for i in 1:m]
hhat_δ_diff50_means = [mean(hhat_δ_diff50[i,:]) for i in 1:m]
hhat_SG_diff50_means = [mean(hhat_SG_diff50[i,:]) for i in 1:m]
hhat_δ_diff99_means = [mean(hhat_δ_diff99[i,:]) for i in 1:m]
hhat_SG_diff99_means = [mean(hhat_SG_diff99[i,:]) for i in 1:m]

hhat_δ_pois1_means = [mean(hhat_δ_pois1[i,:]) for i in 1:m]
hhat_SG_pois1_means = [mean(hhat_SG_pois1[i,:]) for i in 1:m]
hhat_δ_pois50_means = [mean(hhat_δ_pois50[i,:]) for i in 1:m]
hhat_SG_pois50_means = [mean(hhat_SG_pois50[i,:]) for i in 1:m]
hhat_δ_pois99_means = [mean(hhat_δ_pois99[i,:]) for i in 1:m]
hhat_SG_pois99_means = [mean(hhat_SG_pois99[i,:]) for i in 1:m]


get_ess(x) = isnan(round(sum(x)^2/sum(x.^2))) ? NaN : Int64(round(sum(x)^2/sum(x.^2)))

ess_δ_LNA1 = get_ess(hhat_δ_LNA1_means)
ess_SG_LNA1 = get_ess(hhat_SG_LNA1_means)
ess_δ_LNA50 = get_ess(hhat_δ_LNA50_means)
ess_SG_LNA50 = get_ess(hhat_SG_LNA50_means)
ess_δ_LNA99 = get_ess(hhat_δ_LNA99_means)
ess_SG_LNA99 = get_ess(hhat_SG_LNA99_means)

ess_δ_diff1 = get_ess(hhat_δ_diff1_means)
ess_SG_diff1 = get_ess(hhat_SG_diff1_means)
ess_δ_diff50 = get_ess(hhat_δ_diff50_means)
ess_SG_diff50 = get_ess(hhat_SG_diff50_means)
ess_δ_diff99 = get_ess(hhat_δ_diff99_means)
ess_SG_diff99 = get_ess(hhat_SG_diff99_means)

ess_δ_pois1 = get_ess(hhat_δ_pois1_means)
ess_SG_pois1 = get_ess(hhat_SG_pois1_means)
ess_δ_pois50 = get_ess(hhat_δ_pois50_means)
ess_SG_pois50 = get_ess(hhat_SG_pois50_means)
ess_δ_pois99 = get_ess(hhat_δ_pois99_means)
ess_SG_pois99 = get_ess(hhat_SG_pois99_means)












a1 = 2.0*(x₀-x1)/T
a50 = 3.5*(x₀-x50)/T
a99 = 1e20 #7*(x₀-x99)/T
   
GP1 = Guided_Process(obs1, a1, P)
GP50 = Guided_Process(obs50, a50, P)
GP99 = Guided_Process(obs99, a99, P)
   
info1 = filter_backward(GP1)
info50 = filter_backward(GP50)
info99 = filter_backward(GP99)
   
tto1, xxo1 = simulate_forward_monotone(x₀, GP1, info1)
tto50, xxo50 = simulate_forward_monotone(x₀, GP50, info50)
tto99, xxo99 = simulate_forward_monotone(x₀, GP99, info99)

plot(tto99, xxo99, label = "99% quantile",
    color = theme_palette(:auto)[3],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮, 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) 
plot!([T],[x99], markersize = 15, seriestype = :scatter, label = false)

m = 5000
N = 10
π1 = zeros(m)
π50 = zeros(m)
π99 = zeros(m)
p = Progress(m*N)
for i in 1:m
    for j in 1:N
        tto1, xxo1 = simulate_forward_monotone(x₀, GP1, info1)
        tto50, xxo50 = simulate_forward_monotone(x₀, GP50, info50)
        tto99, xxo99 = simulate_forward_monotone(x₀, GP99, info99)
        π1[i] += exp(loglikelihood_direct(tto1,xxo1,GP1,info1))/N
        π50[i] += exp(loglikelihood_direct(tto50,xxo50,GP50,info50))/N
        π99[i] += exp(loglikelihood_direct(tto99,xxo99,GP99,info99))/N
        next!(p)
    end
end

ESS1 = round(sum([p for p in π1])^2/(sum([p^2 for p in π1])))
ESS50 = round(sum([p for p in π50])^2/(sum([p^2 for p in π50])))
ESS99 = round(sum([p for p in π99])^2/(sum([p^2 for p in π99])))

p1 = binomial(x₀,x1)*exp(-c*T*x1)*(1-exp(-c*T))^(x₀-x1)
p50 = binomial(x₀,x50)*exp(-c*T*x50)*(1-exp(-c*T))^(x₀-x50)
p99 = binomial(x₀,x99)*exp(-c*T*x99)*(1-exp(-c*T))^(x₀-x99)
ReMSE1 = sum([ (p-p1)^2/p1 for p in π1 ])/m
ReMSE50 = sum([ (p-p50)^2/p50 for p in π50 ])/m
ReMSE99 = sum([ (p-p99)^2/p99 for p in π99 ])/m



GP14 = Guided_Process(obs50, 14., P)
info14 = filter_backward(GP50)
tto14, xxo14 = simulate_forward_monotone(x₀, GP14, info14)

GP140 = Guided_Process(obs50, 140., P)
info140 = filter_backward(GP140)
tto140, xxo140 = simulate_forward_monotone(x₀, GP140, info140)

GP800 = Guided_Process(obs50, 800., P)
info800 = filter_backward(GP800)
tto800, xxo800 = simulate_forward_monotone(x₀, GP800, info800)

plt = plot(tto14, xxo14, label = "14",
    color = theme_palette(:auto)[1],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮, 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300)
plot!(plt, tto140, xxo140, label = "140", linetype = :steppost, linewidth = 2.0)
plot!(plt, tto800, xxo800, label = "800", linetype = :steppost, linewidth = 2.0)
plot!(plt, [T], [x50], seriestype = :scatter, markersize = 15, label = "observation")
plt


cc = 0.1:0.01:1.5
m = length(cc)
N = 100
πcB = zeros(m,N)
p = Progress(m*N)
for i in 1:m
    for j in 1:N
        GP50 = Guided_Process(obs50, a50, PureDeathProcess_constantrate(cc[i]))
        info50 = filter_backward(GP50)
        tto50, xxo50 = simulate_forward_monotone(x₀, GP50, info50)
        πcB[i,j] += exp(loglikelihood_direct(tto50,xxo50,GP50,info50))
        next!(p)
    end
end

πcB_avg = [mean(πcB[i,:]) for i in 1:m]
πcB_med = [median(πcB[i,:]) for i in 1:m]

p1 = plot(cc, πcB_avg, label = "Estimated likelihood",
    color = theme_palette(:auto)[1],
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "c", 
    ylabel = "Likelihood", 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) 
p2 = plot(cc, πcB_med, label = "Estimated likelihood",
    color = theme_palette(:auto)[1],
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "c", 
    ylabel = "Likelihood", 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) 
plot!(p1,[c], label = "True value", seriestype = :vline)
plot!(p2,[c], label = "True value", seriestype = :vline)
plot(p1,p2,layout = (2,1))






### Poisson approximations
L = 1
obs1 = partial_observation_poisson(T, x1, [nothing,L], 1e-5)
obs50 = partial_observation_poisson(T, x50, [nothing,L], 1e-5)
obs99 = partial_observation_poisson(T, x99, [nothing,L], 1e-5)

a1 = 0.65*(x₀-x1)/T
a50 = 0.9*(x₀-x50)/T
a99 = 1.25*(x₀-x99)/T

GP1 = Guided_Process_Poisson(obs1, a1, P)
GP50 = Guided_Process_Poisson(obs50, a50, P)
GP99 = Guided_Process_Poisson(obs99, a99, P)

info1 = (nothing, nothing,nothing,nothing) # filter_backward(GP1)
info50 = (nothing, nothing,nothing,nothing) # filter_backward(GP50)
info99 = (nothing, nothing,nothing,nothing) #filter_backward(GP99)

tto1, xxo1 = simulate_forward_monotone(x₀, GP1, info1)
tto50, xxo50 = simulate_forward_monotone(x₀, GP50, info50)
tto99, xxo99 = simulate_forward_monotone(x₀, GP99, info99)



ttoo1, xxoo1 = [tto1], [xxo1]
ttoo50, xxoo50 = [tto50], [xxo50]
ttoo99, xxoo99 = [tto99], [xxo99]
K = 1000
p = Progress(K)
for k in 1:K
    tto1, xxo1 = simulate_forward_monotone(x₀, GP1, info1)
    tto50, xxo50 = simulate_forward_monotone(x₀, GP50, info50)
    tto99, xxo99 = simulate_forward_monotone(x₀, GP99, info99)
    push!(ttoo1, tto1) ; push!(xxoo1, xxo1)
    push!(ttoo50, tto50) ; push!(xxoo50, xxo50)
    push!(ttoo99, tto99) ; push!(xxoo99, xxo99)
    next!(p)
end

p1 = plot(ttoo1[1], xxoo1[1], label = "1% quantile",
    color = theme_palette(:auto)[1],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮, 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p2 = plot(ttoo50[1], xxoo50[1], label = "50% quantile",
    color = theme_palette(:auto)[2],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮, 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p3 = plot(ttoo99[1], xxoo99[1], label = "99% quantile",
    color = theme_palette(:auto)[3],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮, 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
for k in 1:100:K
    plot!(p1, ttoo1[k], xxoo1[k], color = theme_palette(:auto)[1], linetype = :steppost, label = false)
    plot!(p2, ttoo50[k], xxoo50[k], color = theme_palette(:auto)[2],linetype = :steppost, label = false)
    plot!(p3, ttoo99[k], xxoo99[k], color = theme_palette(:auto)[3],linetype = :steppost, label = false)
end
plot!(p1, [T], [x1], seriestype=:scatter, markersize = 10, label = false, color = theme_palette(:auto)[1])
plot!(p2, [T], [x50], seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[2])
plot!(p3, [T], [x99], seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[3])
plot(p1,p2,p3, layout = (3,1))   

iscorrect_1obs(GP, tt, xx) = ( getL(GP.obs)*xx[end] == getv(GP.obs))

println("$(round(100*sum([iscorrect_1obs(GP1,ttoo1[k], xxoo1[k]) for k in 1:K]/K), digits=3))% correct")
println("$(round(100*sum([iscorrect_1obs(GP50,ttoo50[k], xxoo50[k]) for k in 1:K]/K), digits=3))% correct")
println("$(round(100*sum([iscorrect_1obs(GP99,ttoo99[k], xxoo99[k]) for k in 1:K]/K), digits=3))% correct")

function loglikelihood_direct(tt,xx, GPP::Guided_Process_Poisson{T},info) where {T}
    t₁, a, xT  = gett(GPP), getθ(GPP), getv(GPP.obs)
    # Array of size n with the indices of the first reaction after the observation times 
    # in the vector tt of the realization (or if n=1 the final time )
    if !iscorrect_1obs(GPP,tt,xx)
        return -Inf
    end
    # out = abs(xT-xx[1])*log(a*t₁)-sum([log(abs(xT-xx[1])-j) for j in 0:1:abs(xT-xx[1])-1]) # log h̃(0,x₀)
    out = logpdf(Poisson(a*t₁), abs(xT-xx[1])) + a*t₁
    for j in 1:length(tt)-2
        t,x = tt[j], xx[j]
        out += abs(xT-x)*log((t₁-tt[j+1])/(t₁-t))*(1-GPP.P.ℛ.λ(t,x)/a) - GPP.P.ℛ.λ(t,x)*(tt[j+1]-t)
    end
    # use process between t_k-1 and t_k 
    # reactions = typeof(GPP.P.ℛ) == reaction{T} ? [GPP.P.ℛ] : GPP.P.ℛ
    # for i in 1:length(tt)-1
    #     t, x = tt[i], xx[i]
    #     # out += t == tt[end-1] ? 0.0 : abs(xT-x)*(log(a*(t₁-tt[i+1]))-log(a*(t₁-t)))
    #     out += t == tt[end-1] ? 0.0 : abs(xT-x)*log((t₁-tt[i+1])/(t₁-t))
    #     ℒh̃(s,p) = sum([ ℓ.λ(s,x)*(guiding_term(info, GPP)(ℓ,s,x) - 1.0) for ℓ in reactions ]) #ℒh̃/h̃
    #     out += solve(IntegralProblem(ℒh̃,tt[i], tt[i+1]) , HCubatureJL() ; reltol = 1e-3, abstol = 1e-3).u
    # end
    return out
end

m = 5000
N = 10
π1 = zeros(m)
π50 = zeros(m)
π99 = zeros(m)
p = Progress(m*N)
for i in 1:m
    for j in 1:N
        tto1, xxo1 = simulate_forward_monotone(x₀, GP1, info1)
        tto50, xxo50 = simulate_forward_monotone(x₀, GP50, info50)
        tto99, xxo99 = simulate_forward_monotone(x₀, GP99, info99)
        π1[i] += exp(loglikelihood_1obs_1dim(tto1,xxo1,GP1,info1))/N
        π50[i] += exp(loglikelihood_1obs_1dim(tto50,xxo50,GP50,info50))/N
        π99[i] += exp(loglikelihood_1obs_1dim(tto99,xxo99,GP99,info99))/N
        next!(p)
    end
end

ESS1 = round(sum([p for p in π1])^2/(sum([p^2 for p in π1])))
ESS50 = round(sum([p for p in π50])^2/(sum([p^2 for p in π50])))
ESS99 = round(sum([p for p in π99])^2/(sum([p^2 for p in π99])))

p1 = binomial(x₀,x1)*exp(-c*T*x1)*(1-exp(-c*T))^(x₀-x1)
p50 = binomial(x₀,x50)*exp(-c*T*x50)*(1-exp(-c*T))^(x₀-x50)
p99 = binomial(x₀,x99)*exp(-c*T*x99)*(1-exp(-c*T))^(x₀-x99)
ReMSE1 = sum([ (p-p1)^2/p1 for p in π1 ])/m
ReMSE50 = sum([ (p-p50)^2/p50 for p in π50 ])/m
ReMSE99 = sum([ (p-p99)^2/p99 for p in π99 ])/m

plt1 = histogram(π1, label = "1% quantile", margin = 10Plots.mm, xlims = (0.0,0.5), xlabel = "probability", ylabel = "Frequency", xguidefontsize = 16, yguidefontsize = 16,
xtickfontsize = 16, ytickfontsize = 16, legend=:topright, legendfotsize = 16, 
size = (1800, 1600), dpi = 300) 
plot!(plt1, [p1], seriestype = :vline, color = :red, label = "True value")
plt2 = histogram(π50, label = "50% quantile", margin = 10Plots.mm, xlims = (0.0,0.5),xlabel = "probability", ylabel = "Frequency", xguidefontsize = 16, yguidefontsize = 16,
xtickfontsize = 16, ytickfontsize = 16, legend=:topright, legendfotsize = 16, 
size = (1800, 1600), dpi = 300) 
plot!(plt2, [p50], seriestype = :vline, color = :red, label = "True value")
plt3 = histogram(π99, label = "99% quantile", margin = 10Plots.mm, xlims = (0.0,0.5),xlabel = "probability", ylabel = "Frequency", xguidefontsize = 16, yguidefontsize = 16,
xtickfontsize = 16, ytickfontsize = 16, legend=:topright, legendfotsize = 16, 
size = (1800, 1600), dpi = 300) 
plot!(plt3, [p99], seriestype = :vline, color = :red, label = "True value")

plot(plt1,plt2,plt3, layout = (3,1))

using Random

aseq=10:0.02:20.0
vec = zeros(50, length(aseq))
p = Progress(50*length(aseq))
for j in 1:length(aseq)
    Random.seed!(4)
    for i in 1:50
    GP = Guided_Process_Poisson(obs50, aseq[j], PureDeathProcess_constantrate(c))
    tto, xxo = simulate_forward_monotone(x₀, GP, info50)
    vec[i,j] = exp(loglikelihood_direct(tto,xxo,GP,info50))
    next!(p)
    end
end

vec_avg = [mean(vec[:,j]) for j in 1:length(aseq)]
vec_med = [median(vec[:,j]) for j in 1:length(aseq)]
plot(aseq,vec_med)


plot(aseq,vec_avg)
plot!(aseq, vec[2,:])
plot!(aseq, vec[20,:])


aa = 1:0.2:30
m = length(aa)
N = 1
πa1 = zeros(m,N) ; πa50 = zeros(m,N) ; πa99 = zeros(m,N)
p = Progress(m*N)
# ttoo50, xxoo50 = [tto50], [xxo50]
# maxtto50, maxxxo50 = [tto50], [xxo50]
for i in 1:m
    for j in 1:N        
        GP1 = Guided_Process_Poisson(obs1, aa[i], P)
        #tto1, xxo1 = simulate_forward_monotone(x₀, GP1, info1)
        πa1[i,j] = exp(loglikelihood_direct(tto1,xxo1,GP1,info1))
        GP50 = Guided_Process_Poisson(obs50, aa[i], P)
        # tto50, xxo50 = simulate_forward_monotone(x₀, GP50, info50)
        πa50[i,j] = exp(loglikelihood_direct(tto50,xxo50,GP50,info50))
        GP99 = Guided_Process_Poisson(obs99, aa[i], P)
        # tto99, xxo99 = simulate_forward_monotone(x₀, GP99, info99)
        πa99[i,j] = exp(loglikelihood_direct(tto99,xxo99,GP99,info99))
        push!(ttoo50,tto50) ; push!(xxoo50,xxo50)
        next!(p)
    end
    # val,ind = findmax(πa50[i,:])
    # push!(maxtto50, ttoo50[ind]) ; push!(maxxxo50,xxoo50[ind])
    # ttoo50, xxoo50 = [tto50], [xxo50]
end

πa1_avg = [mean(πa1[i,:]) for i in 1:m]
πa50_avg = [mean(πa50[i,:]) for i in 1:m]
πa99_avg = [mean(πa99[i,:]) for i in 1:m]

πa1_med = [median(πa1[i,:]) for i in 1:m]
πa50_med = [median(πa50[i,:]) for i in 1:m]
πa99_med = [median(πa99[i,:]) for i in 1:m]


plt1 = plota(aa, πa1_avg, 1.0, "1% quantile, T = $T, Avarege")
plt2 = plota(aa, πa1_med, 1.0, "1% quantile, T = $T, Median")
plot!(plt1, [c*x1], seriestype = :vline, label = false)
plot!(plt2, [c*x1], seriestype = :vline, label = false)
# plot(plt1,plt2,layout = (2,1))
plt3 = plota(aa, πa50_avg, 1.0, "50% quantile, T = $T, Avarege")
plt4 = plota(aa, πa50_med, 1.0, "50% quantile, T = $T, Median")
plot!(plt3, [c*x50], seriestype = :vline, label = false)
plot!(plt4, [c*x50], seriestype = :vline, label = false)
# plot(plt1,plt2,layout = (2,1))
plt5 = plota(aa, πa99_avg, 1.0, "99% quantile, T = $T, Avarege")
plt6 = plota(aa, πa99_med, 1.0, "99% quantile, T = $T, Median")
plot!(plt5, [c*x99], seriestype = :vline, label = false)
plot!(plt6, [c*x99], seriestype = :vline, label = false)
plot(plt1,plt2,plt3,plt4,plt5,plt6,layout = (3,2))

val,ind = findmax(πa50_avg)
val,ind2 = findmax(πa50[ind,:])
plot(maxtto50[ind2], maxxxo50[ind2], label = "50% quantile, a=$(aa[ind])",
    color = theme_palette(:auto)[1],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮, 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300)


cc = 0.1:0.05:1.5
m = length(cc)
N = 100
πc = zeros(m)
p = Progress(m*N)
for i in 1:m
    for j in 1:N
        GP50 = Guided_Process_Poisson(obs50, 17.0, PureDeathProcess_constantrate(cc[i]))
        tto50, xxo50 = simulate_forward_monotone(x₀, GP50, info50)
        πc[i] += exp(loglikelihood(tto50,xxo50,GP50,info50))/N
        next!(p)
    end
end

plot(cc, πc, label = "Estimated likelihood",
    color = theme_palette(:auto)[1],
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "c", 
    ylabel = "Likelihood", 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) 
plot!([c], label = "True value", seriestype = :vline)


c₀ = 5.0
cc = [c₀]
c₀ᵒ = c₀
m = 1000
N = 20
πc = zeros(m)
p = Progress(m*N)
acc = 0
for i in 2:m
    for j in 1:N
        GP50 = Guided_Process_Poisson(obs50, 17.0, PureDeathProcess_constantrate(c₀ᵒ))
        tto50, xxo50 = simulate_forward_monotone(x₀, GP50, info50)
        πc[i] += exp(loglikelihood_1obs_1dim(tto50,xxo50,GP50,info50))/N
        next!(p)
    end
    if rand() <= πc[i]/πc[i-1]
        c₀ = c₀ᵒ
        acc += 1
    end
    c₀ᵒ = c₀ + randn()
    c₀ᵒ = c₀ᵒ < 0 ? c₀ : c₀ᵒ
    push!(cc, c₀)
end

plot(1:m, cc, label = "Trace",
    color = theme_palette(:auto)[1],
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "Iteration", 
    ylabel = "c", 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topright, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) 
plot!(1:m, [c for i in 1:m], label = "True value")


histogram(cc[50:1000], label = false, margin = 10Plots.mm, xlabel = "c", ylabel = "Frequency", xguidefontsize = 16, yguidefontsize = 16,
xtickfontsize = 16, ytickfontsize = 16, legend=:topright, legendfotsize = 16, 
size = (1800, 1600), dpi = 300)
plot!([c], seriestype = :vline, label = "True value")


















