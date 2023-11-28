function plotprocess(tt, xx, P::ChemicalReactionProcess)
    if nr_species(P) == 1
        fig = plot(tt, xx, label = P.𝒮, 
            linetype=:steppost, 
            linewidth = 2.0, 
            margin = 10Plots.mm, 
            xlabel = "t", 
            ylabel = "Counts", 
            xguidefontsize = 16, yguidefontsize = 16,
            xtickfontsize = 16, ytickfontsize = 16,
            legend=:topleft, legendfotsize = 16, 
            size = (1800, 900), 
            dpi = 300)
    else
        fig = plot(tt, map(x -> x[1], xx), label = P.𝒮[1], 
            linetype=:steppost, 
            linewidth = 2.0, 
            margin = 10Plots.mm, 
            xlabel = "t", 
            ylabel = "Counts", 
            xguidefontsize = 16, yguidefontsize = 16,
            xtickfontsize = 16, ytickfontsize = 16,
            legend=:topleft, legendfotsize = 16, 
            size = (1800, 900), 
            dpi = 300)
        for j in 2:nr_species(P)
            plot!(fig, tt, map(x -> x[j], xx), label = P.𝒮[j], 
            linetype=:steppost, 
            linewidth = 2.0, 
            margin = 10Plots.mm, 
            xlabel = "t", 
            ylabel = "Counts", 
            xguidefontsize = 16, yguidefontsize = 16,
            xtickfontsize = 16, ytickfontsize = 16,
            legend=:topleft, legendfotsize = 16, 
            size = (1800, 900), 
            dpi = 300)
        end
    end
    return fig
end