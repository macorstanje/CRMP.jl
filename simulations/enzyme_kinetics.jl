cd("/Users/marc/Documents/GitHub/CRMP.jl/")
using CRMP
using Plots
using LinearAlgebra
using BlockArrays
using ProgressMeter
using Statistics
using StaticArrays
using Random
extractcomp(a, i) = map(x -> x[i], a)
Plots.scalefontsizes(2)

"""
    Bridge, one (partial) observation
"""
# Forward simulation
κ₁ = 5.
κ₂ = 5.
κ₃ = 3.


T = 1.0
x₀ = [12,10,10,10]
P = enzyme_kinetics(κ₁,κ₂,κ₃) # Original process
β(t,x,P) = sum([ℓ.λ(t,x)*ℓ.ξ*ℓ.ξ' for ℓ in P.ℛ])
tt, xx = simulate_forward(constant_rate(), x₀, T, P)

function plot_enzyme_kinetics(ttoo, xxoo, nr_plots::Int64, GP; title = false)
    if nr_plots == 1
        legends = [:topright, :bottomright, :topright,:bottomright]
        plots = [ plot(ttoo, extractcomp(xxoo, i), label = GP.P.𝒮[i],
                    color = theme_palette(:auto)[i],
                    linetype=:steppost, 
                    linewidth = 4.0, 
                    margin = 10Plots.mm, 
                    xlabel = "t", 
                    ylabel = "counts", 
                    xguidefontsize = 16, yguidefontsize = 16,
                    xtickfontsize = 16, ytickfontsize = 16,
                    legend= legends[i], legendfotsize = 16, 
                    size = (1800, 1600), 
                    dpi = 300) # Original process
                    for i in eachindex(GP.P.𝒮) ]
        for i in eachindex(GP.P.𝒮)
            plot!(plots[i], [gett(GP)], [Linv(getv(GP), x₀)[i]], seriestype=:scatter, markersize = 10, label = false, color = theme_palette(:auto)[i])
        end
        plot(plots..., layout = (4,1), plot_title = title)
    else
        legends = [:topright, :bottomright, :topright,:bottomright]
        plots = [ plot(ttoo[1], extractcomp(xxoo[1], i), label = GP.P.𝒮[i],
            color = theme_palette(:auto)[i],
            linetype=:steppost, 
            linewidth = 4.0, 
            margin = 10Plots.mm, 
            xlabel = "t", 
            ylabel = "counts", 
            xguidefontsize = 16, yguidefontsize = 16,
            xtickfontsize = 16, ytickfontsize = 16,
            legend= legends[i], legendfotsize = 16, 
            size = (1800, 1600), 
            dpi = 300) # Original process
            for i in eachindex(GP.P.𝒮) ]
        for i in eachindex(GP.P.𝒮)
            for k in rand(1:length(ttoo), nr_plots)
                plot!(plots[i], ttoo[k], extractcomp(xxoo[k],i), color = theme_palette(:auto)[i], linetype = :steppost, label = false)
            end
            plot!(plots[i], [gett(GP)], [Linv(getv(GP), x₀)[i]], seriestype=:scatter, markersize = 10, label = false, color = theme_palette(:auto)[i])
        end
        plot(plots..., layout = (4,1), plot_title = title)
    end
end



##############################################################################
# Single full observation
##############################################################################
L = Matrix{Float64}(I,4,4)
obs = partial_observation(T, L*xx[end], L, 1e-5)
obsP = partial_observation_poisson(T, xx[end], [Matrix{Float64}(I,3,3), Matrix{Float64}(I,1,1)], 1e-5)
a = diagm([50.,50.,50.,30.])
# a = Bridge.σ(T, xx[end],CLE(P))*Bridge.σ(T, xx[end], CLE(P))'
GP = diffusion_guiding_term(obs,a,P)
GPP = poisson_guiding_term(obsP,a,P)

info = filter_backward(GP)
infoP = filter_backward(GPP)

@time tto, xxo = simulate_forward_monotone(x₀, GP, info)
@time ttoP, xxoP = simulate_forward_monotone(x₀, GPP, infoP)

ttoo, xxoo = [tto], [xxo]
ttooP, xxooP = [ttoP], [xxoP]

K = 100
p = Progress(K)
for k in 1:K
    tto, xxo = simulate_forward_monotone(x₀, GP, info)
    ttoP, xxoP = simulate_forward_monotone(x₀, GPP, infoP)
    push!(ttoo, tto) ; push!(xxoo, xxo)
    push!(ttooP, ttoP) ; push!(xxooP , xxoP)
    next!(p)
end

println("$(round(100*sum([iscorrect_1obs(GP,ttoo[k], xxoo[k]) for k in 1:K]/K), digits=3))% correct")
println("$(round(100*sum([iscorrect_1obs(GPP,ttooP[k], xxooP[k]) for k in 1:K]/K), digits=3))% correct")


# Seperate plots
p1 = plot(ttooP[1], map(x -> x[1], xxooP[1]), label = false,
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[1], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p2 = plot(ttooP[1], map(x -> x[2], xxooP[1]), label = false,
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[2], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p3 = plot(ttooP[1], map(x -> x[3], xxooP[1]), label = false,
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[3], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p4 = plot(ttooP[1], map(x -> x[4], xxooP[1]), label = false,
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[4], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original proces
for k in 2:10   
    plot!(p1, ttooP[k], map(x -> x[1], xxooP[k]), linetype = :steppost, label = false)
    plot!(p2, ttooP[k], map(x -> x[2], xxooP[k]), linetype = :steppost, label = false)
    plot!(p3, ttooP[k], map(x -> x[3], xxooP[k]), linetype = :steppost, label = false)
    plot!(p4, ttooP[k], map(x -> x[4], xxooP[k]), linetype = :steppost, label = false)
end
plot!(p1, [T], [xx[end][1]], seriestype=:scatter, markersize = 10, label = false, color = theme_palette(:auto)[1])
plot!(p2, [T], [xx[end][2]], seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[2])
plot!(p3, [T], [xx[end][3]], seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[3])
plot!(p4, [T], [xx[end][4]], seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[4])
plot(p1,p2,p3,p4, layout = (4,1))   

p1 = plot(ttoo[1], map(x -> x[1], xxoo[1]), label = false,
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[1], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p2 = plot(ttoo[1], map(x -> x[2], xxoo[1]), label = false,
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[2], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p3 = plot(ttoo[1], map(x -> x[3], xxoo[1]), label = false,
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[3], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p4 = plot(ttoo[1], map(x -> x[4], xxoo[1]), label = false,
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[4], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original proces
for k in 2:10   
    plot!(p1, ttoo[k], map(x -> x[1], xxoo[k]), linetype = :steppost, label = false)
    plot!(p2, ttoo[k], map(x -> x[2], xxoo[k]), linetype = :steppost, label = false)
    plot!(p3, ttoo[k], map(x -> x[3], xxoo[k]), linetype = :steppost, label = false)
    plot!(p4, ttoo[k], map(x -> x[4], xxoo[k]), linetype = :steppost, label = false)
end
plot!(p1, [T], [xx[end][1]], seriestype=:scatter, markersize = 10, label = false, color = theme_palette(:auto)[1])
plot!(p2, [T], [xx[end][2]], seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[2])
plot!(p3, [T], [xx[end][3]], seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[3])
plot!(p4, [T], [xx[end][4]], seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[4])
plot(p1,p2,p3,p4, layout = (4,1)) 


πk = zeros(K)
πkP = zeros(K)
p = Progress(K)
for k in 1:K
    πk[k] = likelihood_general_1obs(ttoo[k], xxoo[k], GP, info)
    πkP[k] = likelihood_general_1obs(ttooP[k], xxooP[k], GPP, infoP)
    next!(p)
end
p1 = plot(1:K, πk, seriestype = :scatter, margin = 10Plots.mm, label = false, 
                ylabel = "probability", xguidefontsize = 16, yguidefontsize = 16,
                xtickfontsize = 16, ytickfontsize = 16, legend=:topright, legendfotsize = 16, 
                size = (1800, 1600), dpi = 300)
p2 = plot(1:K, πkP, seriestype=:scatter,margin = 10Plots.mm, label = false, 
                ylabel = "probability",  xguidefontsize = 16, yguidefontsize = 16,
                xtickfontsize = 16, ytickfontsize = 16, legend=:topright, legendfotsize = 16, 
                size = (1800, 1600), dpi = 300) 
plot(p1,p2,layout = (2,1))

# Inference
κ_arr = 2.0:1.0:10
m = length(κ_arr)
N = 40
πκ = zeros(m,N)
πκP = zeros(m,N)
p = Progress(m*N)
for j in 1:m
    for k in 1:N
        GP = diffusion_guiding_term(obs,a,enzyme_kinetics(κ₁,κ_arr[j],κ₃))
        GPP = poisson_guiding_term(obsP,a,enzyme_kinetics(κ₁,κ_arr[j],κ₃))
        info = filter_backward(GP)
        infoP = filter_backward(GPP)
        tto, xxo = simulate_forward_monotone(x₀, GP, info)
        ttoP, xxoP = simulate_forward_monotone(x₀, GPP, infoP)
        πκ[j,k] = likelihood_general_1obs(tto, xxo, GP, info)
        πκP[j,k] = likelihood_general_1obs(ttoP, xxoP, GPP, infoP)
        next!(p)
    end
end

πκ_avg = [mean(πκ[j,:]) for j in 1:m]
πκ_med = [median(πκ[j,:]) for j in 1:m]
πκP_avg = [mean(πκP[j,:]) for j in 1:m]
πκP_med = [median(πκP[j,:]) for j in 1:m]

p1 = plot(κ_arr, πκ_avg, title = "No poisson, average", label = false,
            color = theme_palette(:auto)[1],
            linewidth = 2.0, 
            margin = 10Plots.mm, 
            xlabel = "κ₂", 
            ylabel = "Likelihood", 
            xguidefontsize = 16, yguidefontsize = 16,
            xtickfontsize = 16, ytickfontsize = 16,
            legend=:topright, legendfotsize = 16, 
            size = (1800, 1600), 
            dpi = 300)
p2 = plot(κ_arr, πκ_med, title = "No poisson, median", label = false,
            color = theme_palette(:auto)[1],
            linewidth = 2.0, 
            margin = 10Plots.mm, 
            xlabel = "κ₂", 
            ylabel = "Likelihood", 
            xguidefontsize = 16, yguidefontsize = 16,
            xtickfontsize = 16, ytickfontsize = 16,
            legend=:topright, legendfotsize = 16, 
            size = (1800, 1600), 
            dpi = 300)
p3 = plot(κ_arr, πκP_avg, title = "Poisson, average", label = false,
            color = theme_palette(:auto)[1],
            linewidth = 2.0, 
            margin = 10Plots.mm, 
            xlabel = "κ₂", 
            ylabel = "Likelihood", 
            xguidefontsize = 16, yguidefontsize = 16,
            xtickfontsize = 16, ytickfontsize = 16,
            legend=:topright, legendfotsize = 16, 
            size = (1800, 1600), 
            dpi = 300)
p4 = plot(κ_arr, πκP_med, title = "Poisson, median", label = false,
            color = theme_palette(:auto)[1],
            linewidth = 2.0, 
            margin = 10Plots.mm, 
            xlabel = "κ₂", 
            ylabel = "Likelihood", 
            xguidefontsize = 16, yguidefontsize = 16,
            xtickfontsize = 16, ytickfontsize = 16,
            legend=:topright, legendfotsize = 16, 
            size = (1800, 1600), 
            dpi = 300)
plot!(p1, [κ₂], seriestype = :vline, color = :red, label = false)
plot!(p2, [κ₂], seriestype = :vline, color = :red, label = false)
plot!(p3, [κ₂], seriestype = :vline, color = :red, label = false)
plot!(p4, [κ₂], seriestype = :vline, color = :red, label = false)
plot(p1,p2,p3,p4, layout = (2,2))



# Save (partial)observation at T=1.0, C = 0.005LaL'

L = [1 0 0 0 ; 0 0 0 1]
obs = partial_observation_poisson(T, L*xx[end], [[1 0 0],Matrix{Int64}(I,1,1)], 1e-5)
a = diagm([150.,150.,150.,15.])
#tto, xxo = simulate_forward(x₀, diffusion_guiding_term(obs, a, P))
GPP = poisson_guiding_term(obs,a,P)
info = filter_backward(GPP)
@time tto, xxo = simulate_forward_monotone(x₀, GPP, info)

plt = plotprocess(tto, xxo, P)
#plot!(plt,[gett(v)] , [getx(v)[2]], seriestype=:scatter,  markersize = 12, label = "Observation", color = :black)
ttm, xxm = gett(obs), getv(obs)
# plot!(plt, ttm, extractcomp(xxm,2), seriestype = :scatter, markersize = 15, label = "Observation of MRNA")
plot!(plt, [ttm], [xxm[1]], seriestype = :scatter, markersize = 15, label = "Observation of $(P.𝒮[1])", color = theme_palette(:auto)[1])
# plot!(plt, [ttm], [xxm[2]], seriestype = :scatter, markersize = 15, label = "Observation of $(P.𝒮[2])", color = theme_palette(:auto)[2])
# plot!(plt, [ttm], [xxm[3]], seriestype = :scatter, markersize = 15, label = "Observation of $(P.𝒮[3])", color = theme_palette(:auto)[3])
plot!(plt, [ttm], [xxm[2]], seriestype = :scatter, markersize = 15, label = "Observation of $(P.𝒮[4])", color = theme_palette(:auto)[4])
#savefig(plt, "GTT_15Short.png")

##############################################################################
# Multiple observations
##############################################################################
n = 10
_LL = [rand([[1 0 0], [0 1 0] , [0 0 1]]) for k in 1:n]
LL = [vcat(hcat(L, 0),[0 0 0 1]) for L in _LL]
tint = collect(0:0.001:T)
indices = sort(rand(2:length(tint), n))
times = tint[indices]
xobs = map(t -> xx[searchsortedfirst(tt, t)-1], times)
partial_obs = [
    partial_observation_poisson(times[i], LL[i]*xobs[i], [_LL[i],Matrix{Int64}(I,1,1)], 1e-5) for i in 1:n]


label = [0 for k in eachindex(_LL)]
for k in eachindex(_LL)
    if _LL[k] == [1 0 0]
        label[k] = 1
    elseif _LL[k] == [0 1 0]
        label[k] = 2
    elseif _LL[k] == [0 0 1]
        label[k] = 3
    end
end


plt = plotprocess(tt, xx, P)
plot!(plt, times[label.==1], extractcomp(xobs[label.==1],1), seriestype=:scatter, markersize = 15, label = false, color = theme_palette(:auto)[1])
plot!(plt, times[label.==2], extractcomp(xobs[label.==2],2), seriestype = :scatter, markersize = 15, label = false, color = theme_palette(:auto)[2])
plot!(plt, times[label.==3], extractcomp(xobs[label.==3],3), seriestype = :scatter, markersize = 15, label = false, color = theme_palette(:auto)[3])
plot!(plt, times, extractcomp(xobs,4), seriestype = :scatter, markersize = 15, label = false, color = theme_palette(:auto)[4])
plt

Linv(v, x₀) = [1 0 ; 1 1 ; -1 -1 ; 0 1]*v + [0, x₀[2]-x₀[1]-x₀[4], x₀[3]+x₀[1]+x₀[4],0]

# Using the poisson framework
# θ_array = vcat((getv(partial_obs[1])[end]-x₀[end])/gett(partial_obs[1]) +5.0 , 
    # [5.0 + (getv(partial_obs[i])[end]-getv(partial_obs[i-1])[end])/(gett(partial_obs[i])[end]-gett(partial_obs[i-1])) for i in 2:n])
θ_array = [P.ℛ[end].λ(times[i], xobs[i]) for i in 1:n]
    # a = [sum([ ℓ.λ(gett(o), Linv(getv(o),x₀))*ℓ.ξ*ℓ.ξ' for ℓ in P.ℛ ]) for o in partial_obs] 
a = pushfirst!([25.0*β(times[i], xobs[i],P)/(times[i]-times[i-1]) for i in 2:n], 75*β(times[1], xobs[1],P)/times[1])
aθ = [a - [0 0 0 a[1,4] ; 0 0 0 a[2,4] ; 0 0 0 a[3,4] ; a[4,1] a[4,2] a[4,3] a[4,4]-θ_array[i]] for (i,a) in enumerate(a)]
GP_P = poisson_guiding_term(partial_obs,aθ ,enzyme_kinetics(κ₁,κ₂,κ₃))
info_P = filter_backward(GP_P)
@time tto_P, xxo_P = simulate_forward_monotone(x₀, GP_P, info_P)


# Brownian framework
a = [diagm([50.,50.,50.,7.5]) for i in 1:n]
pobs = [partial_observation(times[i], LL[i]*xobs[i], LL[i], 1e-5) for i in 1:n]
GP_diff = diffusion_guiding_term(pobs, a, enzyme_kinetics(κ₁,κ₂,κ₃))
info_diff = filter_backward(GP_diff)
@time tto_diff, xxo_diff = simulate_forward_monotone(x₀, GP_diff, info_diff)


ttoo_diff, xxoo_diff = [tto_diff], [xxo_diff]
ttoo_P, xxoo_P = [tto_P], [xxo_P]
K = 50
p = Progress(K)
for k in 1:K
    tto_diff, xxo_diff = simulate_forward_monotone(x₀, GP_diff, info_diff)
    # tto_P, xxo_P = simulate_forward_monotone(x₀, GP_P, info_P)
    push!(ttoo_diff, tto_diff) ; push!(xxoo_diff, xxo_diff)
    # push!(ttoo_P, tto_P) ; push!(xxoo_P,xxo_P)
    next!(p)
end

"""
    Checking correctness
"""
# Check if the observations match
function iscorrect(GPP, tto, xxo )
    times = gett(GPP) ; obs = GPP.obs ; LL = getL(GPP)
    indices = [findfirst(t -> t==times[k], tto) for k in 1:n]
    simulated_values = map(k -> LL[k]*xxo[indices[k]], 1:n)
    out = [simulated_values[k] .- getv(obs[k]) == [0 for j in 1:getm(obs[k])] for k in 1:n]
    return all(out)
end

println("$(round(100*sum([iscorrect(GP_diff,ttoo_diff[k], xxoo_diff[k]) for k in 1:K]/K), digits=3))% correct")
println("$(round(100*sum([iscorrect(GP_P,ttoo_P[k], xxoo_P[k]) for k in 1:K]/K), digits=3))% correct")


for k in 1:K
    println(iscorrect(GP_P, ttoo_P[k], xxoo_P[k]))
end

"""
    Plotting
"""
# All in 1 plot 
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
for k in (0*K+1):1*K
    plot!(plt, ttoo[k], extractcomp(xxoo[k], 1), label = false, linetype=:steppost, color = theme_palette(:auto)[1], alpha = 0.7)
    plot!(plt, ttoo[k], extractcomp(xxoo[k], 2), label = false, linetype=:steppost, color = theme_palette(:auto)[2], alpha = 0.7)
    plot!(plt, ttoo[k], extractcomp(xxoo[k], 3), label = false, linetype=:steppost, color = theme_palette(:auto)[3], alpha = 0.7)
    plot!(plt, ttoo[k], extractcomp(xxoo[k], 4), label = false, linetype=:steppost, color = theme_palette(:auto)[4], alpha = 0.7)
end
plot!(plt, times[label.==1], extractcomp(xobs[label.==1],1), seriestype=:scatter, markersize = 15, label = false, color = theme_palette(:auto)[1])
plot!(plt, times[label.==2], extractcomp(xobs[label.==2],2), seriestype = :scatter, markersize = 15, label = false, color = theme_palette(:auto)[2])
plot!(plt, times[label.==3], extractcomp(xobs[label.==3],3), seriestype = :scatter, markersize = 15, label = false, color = theme_palette(:auto)[3])
plot!(plt, times, extractcomp(xobs,4), seriestype = :scatter, markersize = 15, label = false, color = theme_palette(:auto)[4])
plt

# Seperate plots
p1 = plot(ttooP[1], map(x -> x[1], xxooP[1]), label = false, color = theme_palette(:auto)[1],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[1], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p2 = plot(ttooP[1], map(x -> x[2], xxooP[1]), label = false,color = theme_palette(:auto)[2],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[2], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p3 = plot(ttooP[1], map(x -> x[3], xxooP[1]), label = false,color = theme_palette(:auto)[3],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[3], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p4 = plot(ttooP[1], map(x -> x[4], xxooP[1]), label = false,color = theme_palette(:auto)[4],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[4], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original proces
for k in 2:10   
    plot!(p1, ttooP[k], map(x -> x[1], xxooP[k]), linetype = :steppost, label = false, color = theme_palette(:auto)[1])
    plot!(p2, ttooP[k], map(x -> x[2], xxooP[k]), linetype = :steppost, label = false, color = theme_palette(:auto)[2])
    plot!(p3, ttooP[k], map(x -> x[3], xxooP[k]), linetype = :steppost, label = false, color = theme_palette(:auto)[3])
    plot!(p4, ttooP[k], map(x -> x[4], xxooP[k]), linetype = :steppost, label = false, color = theme_palette(:auto)[4])
end
plot!(p1, times[label.==1], extractcomp(xobs[label.==1],1), seriestype=:scatter, markersize = 10, label = false, color = theme_palette(:auto)[1])
plot!(p2, times[label.==2], extractcomp(xobs[label.==2],2), seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[2])
plot!(p3, times[label.==3], extractcomp(xobs[label.==3],3), seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[3])
plot!(p4, times, extractcomp(xobs,4), seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[4])
plot(p1,p2,p3,p4, layout = (4,1))   

p1 = plot(ttoo_diff[1], map(x -> x[1], xxoo_diff[1]), label = false,color = theme_palette(:auto)[1],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[1], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p2 = plot(ttoo_diff[1], map(x -> x[2], xxoo_diff[1]), label = false,color = theme_palette(:auto)[2],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[2], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p3 = plot(ttoo_diff[1], map(x -> x[3], xxoo_diff[1]), label = false,color = theme_palette(:auto)[3],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[3], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original process
p4 = plot(ttoo_diff[1], map(x -> x[4], xxoo_diff[1]), label = false,color = theme_palette(:auto)[4],
    linetype=:steppost, 
    linewidth = 2.0, 
    margin = 10Plots.mm, 
    xlabel = "t", 
    ylabel = P.𝒮[4], 
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=:topleft, legendfotsize = 16, 
    size = (1800, 1600), 
    dpi = 300) # Original proces
for k in 2:10   
    plot!(p1, ttoo_diff[k], map(x -> x[1], xxoo_diff[k]), linetype = :steppost, label = false,color = theme_palette(:auto)[1])
    plot!(p2, ttoo_diff[k], map(x -> x[2], xxoo_diff[k]), linetype = :steppost, label = false,color = theme_palette(:auto)[2])
    plot!(p3, ttoo_diff[k], map(x -> x[3], xxoo_diff[k]), linetype = :steppost, label = false,color = theme_palette(:auto)[3])
    plot!(p4, ttoo_diff[k], map(x -> x[4], xxoo_diff[k]), linetype = :steppost, label = false,color = theme_palette(:auto)[4])
end
plot!(p1, times[label.==1], extractcomp(xobs[label.==1],1), seriestype=:scatter, markersize = 10, label = false, color = theme_palette(:auto)[1])
plot!(p2, times[label.==2], extractcomp(xobs[label.==2],2), seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[2])
plot!(p3, times[label.==3], extractcomp(xobs[label.==3],3), seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[3])
plot!(p4, times, extractcomp(xobs,4), seriestype = :scatter, markersize = 10, label = false, color = theme_palette(:auto)[4])
plot(p1,p2,p3,p4, layout = (4,1))   



indices = [findfirst(t -> t==times[k], ttoo[4]) for k in 1:n]
simulated_values = map(k -> LL[k]*xxoo[4][indices[k]], 1:n)
out = [simulated_values[k] .- getv(partial_obs[k]) == [0 for j in 1:getm(partial_obs[k])] for k in 1:n]

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
    plot!(plt, ttoo[4], extractcomp(xxoo[4], 1), label = false, linetype=:steppost, color = theme_palette(:auto)[1], alpha = 0.7)
    plot!(plt, ttoo[4], extractcomp(xxoo[4], 2), label = false, linetype=:steppost, color = theme_palette(:auto)[2], alpha = 0.7)
    plot!(plt, ttoo[4], extractcomp(xxoo[4], 3), label = false, linetype=:steppost, color = theme_palette(:auto)[3], alpha = 0.7)
    plot!(plt, ttoo[4], extractcomp(xxoo[4], 4), label = false, linetype=:steppost, color = theme_palette(:auto)[4], alpha = 0.7)
plot!(plt, times[label.==1], extractcomp(xobs[label.==1],1), seriestype=:scatter, markersize = 15, label = false, color = theme_palette(:auto)[1])
plot!(plt, times[label.==2], extractcomp(xobs[label.==2],2), seriestype = :scatter, markersize = 15, label = false, color = theme_palette(:auto)[2])
plot!(plt, times[label.==3], extractcomp(xobs[label.==3],3), seriestype = :scatter, markersize = 15, label = false, color = theme_palette(:auto)[3])
plot!(plt, times, extractcomp(xobs,4), seriestype = :scatter, markersize = 15, label = false, color = theme_palette(:auto)[4])
plt


 K = 10
κκ = collect(0.1: 0.01:50.0)
n2 = length(κκ)

mat = zeros(n2)
p = Progress(n2*K)
tta = [tt]
xxa = [xx]
for i in 1:n2
    P = enzyme_kinetics(κκ[i], κ₂,κ₃)
    GP = diffusion_guiding_term(partial_obs,a,P)
    info = filter_backward(GP)
    tto, xxo = simulate_forward_monotone(x₀, GP, info)
    mat[i] += loglikelihood(tto, xxo, GP, info)
    next!(p)
end
plt = plot(κκ, mat, xlabel ="\$ \\κ_1 \$", ylabel = "log-likelihood", 
    title = "Likelihood using 1 sample with varying values of κ₁",
    label = false,
    margin = 12Plots.mm, size = (1800, 900))
for l in 1:(K-1)
    mat = zeros(n2)
    tta = [tt]
    xxa = [xx]
    for i in 1:n2
        P = enzyme_kinetics(κκ[i], κ₂,κ₃)
        GP = diffusion_guiding_term(partial_obs,a,P)
        info = filter_backward(GP)
        tto, xxo = simulate_forward_monotone(x₀, GP, info)
        mat[i] += loglikelihood(tto, xxo, GP, info)
        next!(p)
    end
    plot!(plt,κκ, mat, margin = 12Plots.mm, size = (1800, 900), label = false)
end
plot!(plt, [κ₁], seriestype=:vline, label = "True value", linewidth = 5.0)
plt


























"""
    Comparison of three methods LNA method
"""

T = 1.0
x₀ = [12,10,10,10]
κ₁ = 5.
κ₂ = 5.
κ₃ = 3.
P = enzyme_kinetics(κ₁,κ₂,κ₃) # Original process
tt, xx = simulate_forward(constant_rate(), x₀, T, P)

samples = [xx[end]]
prog = Progress(5000)
for k in 1:5000
    tt,xx = simulate_forward(constant_rate(), x₀,T,P)
    push!(samples, xx[end])
    next!(prog)
end

x1 = rand(samples[extractcomp(samples,4) .== quantile(extractcomp(samples,4), 0.01)])
x50 = rand(samples[extractcomp(samples,4) .== quantile(extractcomp(samples,4), 0.5)])
x99 = rand(samples[extractcomp(samples,4) .== quantile(extractcomp(samples,4), 0.99)])

# plotprocess(tt, xx, P)

### Get quantiles and median
# samples = [xx[end]]
# K = 5000
# prog = Progress(K)
# for k in 1:K
#     tt, xx = simulate_forward(constant_rate(), x₀, T, P)
#     push!(samples, xx[end])
#     next!(prog)
# end

# histogram(samples, label = false)
# x1 = Int64(quantile(samples, 0.01))
# x50 = Int64(quantile(samples, 0.5))
# x99 = Int64(quantile(samples, 0.99))
# xT = xx[end]

# FULL OBSERVATION
L = SMatrix{2,4, Float64}([1 0 0 0 ; 0 0 0 1])
ϵ = 1e-5
C = SMatrix{2,2,Float64}(ϵ*I)
obs1 = partial_observation(T, L*x1, L, ϵ)
obs1_P = partial_observation_poisson(T, L*x1,[[1 0 0],Matrix{Int64}(I,1,1)], ϵ)
obs50 = partial_observation(T, L*x50, L, ϵ)
obs50_P = partial_observation_poisson(T, L*x50,[[1 0 0],Matrix{Int64}(I,1,1)], ϵ)
obs99 = partial_observation(T, L*x99, L, ϵ)
obs99_P = partial_observation_poisson(T, L*x99, [[1 0 0],Matrix{Int64}(I,1,1)], ϵ)

# P1 = LNAR(L,x1,T,C,P)
# P50 = LNAR(L,x50,T,C,P)
# P99 = LNAR(L,x99,T,C,P)

# GP_LNAR = LNAR(L,L*xT,T,C,P)
GP1_LNA_nR = LNA_nR(obs1, 0.0:0.001:T , P)
@time fill_grid!(GP1_LNA_nR, x₀)


GP1_LNAR = LNAR(obs1,P)
GP50_LNAR = LNAR(obs50,P)
GP99_LNAR = LNAR(obs99,P)

a = β(0,x₀, P)
GP1_diff = diffusion_guiding_term(obs1, 75*β(T, x1, P), P)
GP50_diff = diffusion_guiding_term(obs50, 100*β(T, x50, P), P)
GP99_diff = diffusion_guiding_term(obs50,100*β(T, [0,19,1,31], P), P)

aθ1, aθ50, aθ99 = [a - [0 0 0 a[1,4] ; 0 0 0 a[2,4] ; 0 0 0 a[3,4] ; a[4,1] a[4,2] a[4,3] 0] for a in [β(T, x1, P),  β(T, x50, P), β(T, [0,19,1,31], P)]]
GP1_P = poisson_guiding_term(obs1_P, 1.5*aθ1, P)
GP50_P = poisson_guiding_term(obs50_P, 1.5*aθ50, P)
GP99_P = poisson_guiding_term(obs99_P, 1.5*aθ99, P)
# tto1, xxo1 = simulate_forward_monotone(x₀, P1)
# tto50, xxo50 = simulate_forward_monotone(x₀, P50)
# tto99, xxo99 = simulate_forward_monotone(x₀, P99)


# plot_enzyme_kinetics(ttoo,xxoo,10,GP_LNAR)


# L = [1. 0. 0. 0. ; 0. 0. 0. 1.]


info1 = filter_backward(GP1_diff)
info50 = filter_backward(GP50_diff)
info99 = filter_backward(GP99_diff)

info1_P = filter_backward(GP1_P)
info50_P = filter_backward(GP50_P)
info99_P = filter_backward(GP99_P)


@time tto_LNAR, xxo_LNAR = simulate_forward_monotone(x₀, GP50_LNAR, info50)
@time tto_diff, xxo_diff = CRMP.simulate_forward_monotone(x₀, GP50_diff, info50)
@time tto_P, xxo_P = CRMP.simulate_forward_monotone(x₀, GP50_P, info50_P)



# xT_array = []
# ttoo, xxoo , ttooP, xxooP, ttoo_LNAR, xxoo_LNAR = [],[],[],[],[],[]
# prog = Progress(10*K)
# GP_LNAR = [] ; GP = [] ; GPP = []
# for i in 1:10
#     tt, xx = simulate_forward(constant_rate(), x₀, T, P)
#     xT = xx[end]
#     push!(xT_array, xT)

#     L = [1. 0. 0. 0. ; 0. 0. 0. 1.]
#     obs = partial_observation(T, L*xx[end], L, 1e-5)
#     obsP = partial_observation_poisson(T, L*xT, [[1. 0. 0.], Matrix{Float64}(I,1,1)], 1e-5)
#     a = β(0.0,x₀,P)
#     # a = Bridge.σ(T, xx[end],CLE(P))*Bridge.σ(T, xx[end], CLE(P))'
#     GP_LNAR = LNAR(L,L*xT,T,C,P)
#     GP = diffusion_guiding_term(obs,a,P)
#     GPP = poisson_guiding_term(obsP,[a[1,1] a[1,2] a[1,3] 0.0 ; a[2,1] a[2,2] a[2,3] 0.0 ; a[3,1] a[3,2] a[3,3] 0.0 ; 0.0 0.0 0.0 10.0],P)

#     info = filter_backward(GP)
#     infoP = filter_backward(GPP)


#     push!(ttoo_LNAR, [tt]) ; push!(xxoo_LNAR, [xx])
#     push!(ttoo, [tt]) ; push!(xxoo, [xx])
#     push!(ttooP, [tt]) ; push!(xxooP, [xx])
#     K = 50
#     for k in 1:K
#         tto_LNAR, xxo_LNAR = simulate_forward_monotone(x₀, GP_LNAR)
#         tto, xxo = CRMP.simulate_forward_monotone(x₀, GP, info)
#         ttoP, xxoP = CRMP.simulate_forward_monotone(x₀, GPP, infoP)
#         push!(ttoo_LNAR[i],tto_LNAR) ; push!(xxoo_LNAR[i], xxo_LNAR)
#         push!(ttoo[i], tto) ; push!(xxoo[i], xxo)
#         push!(ttooP[i], ttoP) ; push!(xxooP[i] , xxoP)
#         next!(prog)
#     end
# end

Random.seed!(61)
count1_LNA = 0
count1_diff = 0
count1_P = 0
count50_LNA = 0
count50_diff = 0
count50_P = 0
count99_LNA = 0
count99_diff = 0
count99_P = 0

ttoo_LNAR , xxoo_LNAR = [tt], [xx]
ttoo, xxoo = [tt], [xx]
ttooP, xxooP = [tt], [xx]
K = 100
prog = Progress(K)

# TAKES LONG WHEN LNA IS INCLUDED
for k in 1:K
    tto_LNAR, xxo_LNAR = simulate_forward_monotone(x₀, GP1_LNAR, info1)
    tto_diff, xxo_diff = simulate_forward_monotone(x₀, GP1_diff, info1)
    tto_P, xxo_P = simulate_forward_monotone(x₀, GP1_P, info1_P)

    count1_LNA += Int64(iscorrect_1obs(GP1_LNAR, tto_LNAR, xxo_LNAR))
    count1_diff += Int64(iscorrect_1obs(GP1_diff, tto_diff, xxo_diff))
    count1_P += Int64(iscorrect_1obs(GP1_P, tto_P, xxo_P))

    tto_LNAR, xxo_LNAR = simulate_forward_monotone(x₀, GP50_LNAR, info50)
    tto_diff, xxo_diff = simulate_forward_monotone(x₀, GP50_diff, info50)
    tto_P, xxo_P = simulate_forward_monotone(x₀, GP50_P, info50_P)

    count50_LNA += Int64(iscorrect_1obs(GP50_LNAR, tto_LNAR, xxo_LNAR))
    count50_diff += Int64(iscorrect_1obs(GP50_diff, tto_diff, xxo_diff))
    count50_P += Int64(iscorrect_1obs(GP50_P, tto_P, xxo_P))    
    
    tto_LNAR, xxo_LNAR = simulate_forward_monotone(x₀, GP99_LNAR, info99)
    tto_diff, xxo_diff = simulate_forward_monotone(x₀, GP99_diff, info99)
    tto_P, xxo_P = simulate_forward_monotone(x₀, GP99_P, info99_P)

    count99_LNA += Int64(iscorrect_1obs(GP99_LNAR, tto_LNAR, xxo_LNAR))
    count99_diff += Int64(iscorrect_1obs(GP99_diff, tto_diff, xxo_diff))
    count99_P += Int64(iscorrect_1obs(GP99_P, tto_P, xxo_P))

    next!(prog)
end

p1 = bar(["LNA", "Diffusion", "Poisson"], 100.0.*[count1_LNA/K, count1_diff/K, count1_P/K], title = "Scenario A", margin = 10Plots.mm, ylabel = "Percentage correct", ylim = (0,100),
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=false, legendfotsize = 16, 
    size = (1800, 1000), 
    dpi = 300)
p50 = bar(["LNA", "Diffusion", "Poisson"], 100.0.*[count50_LNA/K, count50_diff/K, count50_P/K], title = "Scenario B", margin = 10Plots.mm, ylabel = "Percentage correct",ylim = (0,100),
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=false, legendfotsize = 16, 
    size = (1800, 1000), 
    dpi = 300)
p99 = bar(["LNA", "Diffusion", "Poisson"], 100.0.*[count99_LNA/K, count99_diff/K, count99_P/K], title = "Scenario C", margin = 10Plots.mm, ylabel = "Percentage correct",ylim = (0,100),
    xguidefontsize = 16, yguidefontsize = 16,
    xtickfontsize = 16, ytickfontsize = 16,
    legend=false, legendfotsize = 16, 
    size = (1800, 1000), 
    dpi = 300)
plt = plot(p1,p50,p99,layout = (1,3))
plt
savefig(plt, "enzyme_percentages.png")



plot_enzyme_kinetics(tto_LNAR, xxo_LNAR, 1, GP50_LNAR, title = "LNA")
plot_enzyme_kinetics(tto_diff,xxo_diff,1, GP50_diff, title = "Diffusion")
plot_enzyme_kinetics(tto_P,xxo_P,1,GP50_P, title = "Poisson")


for i in 1:10
    xT = xT_array[i]
    obs = partial_observation(T, L*xT, L, 1e-5)
    obsP = partial_observation_poisson(T, L*xT, [[1. 0. 0.], Matrix{Float64}(I,1,1)], 1e-5)
    a = β(0.0,x₀,P)
    # a = Bridge.σ(T, xx[end],CLE(P))*Bridge.σ(T, xx[end], CLE(P))'
    GP_LNAR = LNAR(L,L*xT,T,C,P)
    GP = diffusion_guiding_term(obs,a,P)
    GPP = poisson_guiding_term(obsP,a,P)
    println("At xT = $(xT_array[i])")
    println("LNA guiding term : $(round(100*mean([iscorrect_1obs(GP_LNAR,ttoo_LNAR[i][k], xxoo_LNAR[i][k]) for k in eachindex(ttoo_LNAR[i])]), digits=3))% correct")
    println("Diffusion guiding term: $(round(100*mean([iscorrect_1obs(GP,ttoo[i][k], xxoo[i][k]) for k in eachindex(ttoo[i])]), digits=3))% correct")
    println("Poisson guiding term : $(round(100*mean([iscorrect_1obs(GPP,ttooP[i][k], xxooP[i][k]) for k in eachindex(ttooP[i])]), digits=3))% correct")
    println("")
end

k = 5
plot_enzyme_kinetics(ttoo_LNAR[k], xxoo_LNAR[k], 10, GP_LNAR, title = "LNA")
plot_enzyme_kinetics(ttoo[k],xxoo[k],10, GP, title = "Diffusion")
plot_enzyme_kinetics(ttooP[k],xxooP[k],10,GPP, title = "Poisson")

