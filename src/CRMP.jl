module CRMP

    using LinearAlgebra, ForwardDiff
    using Distributions
    using Plots
    using Integrals
    using SpecialFunctions
    using StaticArrays
    using StatsBase
    using DifferentialEquations
    using DiffEqCallbacks
    using FiniteDifferences

    # Network basics
    export reaction, ChemicalReactionNetwork, ChemicalReactionProcess, nr_species, nr_reactions, reaction_array
    export PoissonProcess_constantrate, BirthDeathProcess, GTT, viral_infection, enzyme_kinetics
    export Schlogl, lotka_volterra, AR, PureDeathProcess_constantrate

    # For conditional (guided) processes

    export partial_observation, Guided_Process, diffusion_guiding_term
    export partial_observation_poisson, poisson_guiding_term
    export poisson_terms, poisson_density, log_poisson_density
    export gett, getx, getL, getm, getd, getn, getϵ, getv, getk, getL₁, getL₂, geta, getθ
    export get_upper_bound, filter_backward
    export log_guiding_term, guiding_term
    export condition_reaction, uncondition_reaction, condition_process

    # LNA methods 
    export LNA, pois, LNAR, LNAR_death, LNA_nR, diff_death, pois_death, fill_grid!
    export loglikelihood_SG

    # Reaction times
    export method, constant_rate, decreasing_rate, increasing_rate, thinning, gettime, setδ

    # Forward simulation of the process
    export Gillespie, next_jump, simulate_forward, simulate_forward_monotone

    # Likelihood computations 
    export loglikelihood, likelihood, loglikelihood_general_1obs, likelihood_general_1obs, iscorrect_1obs

    # Plot process
    export plotprocess

    include("networks.jl")
    include("conditional_process.jl")
    include("LNA.jl")
    include("reaction_times.jl")
    include("forward_simulation.jl")
    include("likelihood.jl")
    include("plotprocess.jl")
end
