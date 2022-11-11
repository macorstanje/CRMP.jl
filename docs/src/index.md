# Documentation for CRMP.jl

## Summary

Library with functions for forward simulation of continuous-time chemical reaction process and performing statistical inference on such process. Manual to be added later.

## Walkthrough for Birth-death process
The birth death-process consists of the simple species set `𝒮 = {"Individuals"}`
and the reactions `birth` and `death` with rates `birth_rate` and `death_rate`, respectively.
```@julia-repl
julia> using CRMP
julia> birth = reaction( (t,x) -> x + 1 , birth_rate*x)
julia> death = reaction( (t,x) -> x - 1 , death_rate*x)
julia> P = ChemicalReactionProcess(["Individuals"], [birth, death])
```

Acces the set of species or the set of reactions through `P.𝒮` and `P.ℛ` respectively.
The birth-death process is also a buit-in function of this library accessed through
```@julia-repl
julia> P = BirthDeathProcess(birth_rate, death_rate)
```
For other built-in reaction process, see [networks](@ref networks).

### Forward simulation

The rate functions are constant in time and thus we simulate forward given `x₀` and some final time `T` using
```@julia-repl
julia> tt, xx = simulate_forward(constant_rate(), x₀, T, P)
```
We can plot the components using (dedicated plot function will be added to the library later)
```@julia-repl
julia> using Plots
julia> plot(tt, map(x -> x[1], xx), label = P.𝒮[1], xlabel = "t", ylabel = "counts")
julia> plot(tt, map(x -> x[2], xx), label = P.𝒮[2], xlabel = "t", ylabel = "counts")
```

### Conditioned process
If we have a desired end-state `(T,xT)` and latent variable `ϕ`, in the guided proposal framework often the diffusivity of the auxiliary process, we simulate a guided process via
```@julia-repl
CP = condition_process(P, xT, T, dist²(ϕ))
tto, xxo = simulate_forward(conditional(), x₀, xT, T, CP, dist²(ϕ), 0.5)
```
Here, `dist²(ϕ)` is a distance measure on the chemical reaction networks, currently set as `dist²(ϕ)(x,y) = |y-x|²/ϕ`. The `0.5` can be adjusted, this is the minimal desired acceptance rate of the thinning process when sampling reaction times.

### Likelihood computation

This is currently still implemented locally, will follow soon ...
