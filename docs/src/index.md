# Documentation for CRMP.jl

## Summary

Library with functions for forward simulation of continuous-time chemical reaction process and performing statistical inference on such process. Manual to be added later.

## Walkthrough for Birth-death process
The birth death-process consists of the simple species set `𝒮 = {"Individuals"}`
and the reactions `birth` and `death` with rates `birth_rate` and `death_rate`, respectively.
```julia-repl
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
```julia-repl
julia> tt, xx = simulate_forward(constant_rate(), x₀, T, P)
```
We can plot the components using 
```julia-repl
julia> plotprocess(tt, xx, P)
```

### Conditioned process
If we have a desired end-state `(T,xT)`, we can employ various guiding terms to find a guided process. See [Conditional processes](@ref conditional_process) for all of them. We demostrate the diffusion guiding term.
```julia-repl
julia> L = 1.0
julia> eps = 1e-5
julia> obs = partial_observation(T, xT, L, eps)
julia> a = 10.0
julia> GP = diffusion_guiding_term(obs, a, P)
julia> info = filter_backward(GP)
julia> tto, xxo = simulate_forward_monotone(x₀, GP, info)
```


### Likelihood computation

For likelihood computation with 1 observation, the preferred method is the function `loglikelihood_general_1obs`. For multiple observations, LNA methods are not implemented yet. For poisson and diffusion guiding terms, use `loglikelihood(tto,xxo,GP,info)`

Continuing the example:
```julia-repl
julia> loglikelihood_general_1obs(tto, xxo, GP, info)
``` 

### Extension to multiple observations
For multiple observations, use the methods prescribed earlier, but then with an array of `partial_observation`s and an array of `a`. 
