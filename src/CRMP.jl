module CRMP

    using LinearAlgebra, ForwardDiff
    using Distributions
    using Plots

    # Network basics
    export reaction, ChemicalReactionNetwork, ChemicalReactionProcess, nr_species, nr_reactions
    export PoissonProcess_consantrate, BirthDeathProcess, GTT, viral_infection, enzyme_kinetics

    # For conditional (guided) processes
    export dist², C, condition_reaction, condition_process, setδ

    # Reaction times
    export method, constant_rate, decreasing_rate, increasing_rate, gettime

    # Forward simulation of the process
    export next_jump, conditional, simulate_forward 

    include("networks.jl")
    include("conditional_process.jl")
    include("reaction_times.jl")
    include("forward_simulation.jl")
end
