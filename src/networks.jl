

struct reaction{T<:Real}
    λ::Function
    ξ::Union{T, Array{T,1}}
end

struct ChemicalReactionNetwork{T<:Real}
    𝒮::Union{String, Array{String, 1}}
    ℛ::Union{reaction{T}, Array{reaction{T},1}}
end

struct ChemicalReactionProcess{T<:Real}
    𝒮::Union{String, Array{String, 1}}
    ℛ::Union{reaction{T}, Array{reaction{T}, 1}}
end
nr_species(P::ChemicalReactionProcess) = typeof(P.𝒮) == String ? 1 : length(P.𝒮)
nr_reactions(P::ChemicalReactionProcess) = typeof(P.ℛ) == Reaction ? 1 : length(P.ℛ)

"""
    Poisson process and Birth-death process
"""
function PoissonProcess_constantrate(rate::Real)
    @assert rate > 0 "Rate must be positive"
    plus1 = reaction( (t,x) -> x*rate, 1)
    return ChemicalReactionProcess("Counts", plus1)
end

function BirthDeathProcess(birth_rate::Real , death_rate::Real)
    @assert min(birth_rate , death_rate) > 0 "All rates must be positive"
    plus1 = reaction( (t,x) -> x*birth_rate, 1)
    minus1 = reaction( (t,x) -> x*death_rate, -1)
    return ChemicalReactionProcess("Individuals", [plus1, minus1])
end

"""
    Gene transcription and translation (possibly with dimerization of protein)
"""
function GTT(κ₁::T ,κ₂::T, dₘ::T, dₚ::T) where {T<:Real}
    @assert min(κ₁,κ₂,dₘ,dₚ) > 0 "All rate parameters must be positive"
    Transcription = reaction( (t,x) -> κ₁*x[1] , [0, 1, 0])
    Translation = reaction( (t,x) -> κ₂*x[2], [0, 0, 1])
    Degradation_mRNA = reaction( (t,x) -> dₘ*x[2] , [0, -1, 0])
    Degradation_Protein = reaction( (t,x) -> dₚ*x[3] , [0, 0, -1])
    return ChemicalReactionProcess(["Gene", "mRNA", "Protein"], [Transcription, Translation, Degradation_mRNA, Degradation_Protein])
end

function GTT(κ₁::T ,κ₂::T, κ₃::T, dₘ::T, dₚ::T, dD::T) where {T<:Real}
    @assert min(κ₁,κ₂,κ₃,dₘ,dₚ,dD) > 0 "All rate parameters must be positive"
    Transcription = reaction( (t,x) -> κ₁*x[1] , [0, 1, 0])
    Translation = reaction( (t,x) -> κ₂*x[2], [0, 0, 1])
    Degradation_mRNA = reaction( (t,x) -> dₘ*x[2] , [0, -1, 0])
    Degradation_Protein = reaction( (t,x) -> dₚ*x[3] , [0, 0, -1])
    Dimerization = Reaction( (t,x) -> κ₃*x[3]*(x[3]-1) , [0,0,-2, 1])
    Degradation_Dimer = Reaction( (t,x) -> dD*x[4] , [0,0,0,-1])
    return ChemicalReactionProcess(["Gene", "mRNA", "Protein", "Dimer"],
        [Transcription, Translation, Degradation_mRNA, Degradation_Protein, Dimerization, Degradation_Dimer])
end

"""
    Virus kinetics

See e.g. section 2.1.2 of Anderson & Kurtz
"""
function viral_infection(κ₁::T, κ₂::T, κ₃::T, κ₄::T, κ₅::T, κ₆::T) where {T<:Real}
    @assert min(κ₁,κ₂,κ₃,κ₄,κ₅,κ₆) > 0 "All rate parameters must be positive"
    R1 = reaction( (t,x) -> κ₁*x[3] , [1,0,0,0])
    R2 = reaction( (t,x) -> κ₂*x[1] , [-1,0,1,0])
    R3 = reaction( (t,x) -> κ₃*x[3] , [0,1,0,0])
    R4 = reaction( (t,x) -> κ₄*x[3] , [0,0,-1,0])
    R5 = reaction( (t,x) -> κ₅*x[2] , [0,-1,0,0])
    R6 = reaction( (t,x) -> κ₆*x[1]*x[2] , [-1,-1,0,1])
    return ChemicalReactionProcess(["Genome", "Viral structural protein", "Viral template", "Virus"], [R1,R2,R3,R4,R5,R6])
end


"""
    Enzyme kinetics

See e.g. section 2.1.3 of Anderson & Kurtz
"""
function enzyme_kinetics(κ₁::T, κ₂::T, κ₃::T) where {T<:Real}
    @assert min(κ₁,κ₂,κ₃) > 0 "All rate parameters must be positive"
    R1 = reaction( (t,x) -> κ₁*x[1]*x[2] , [-1, -1, 1, 0])
    R2 = reaction( (t,x) -> κ₂*x[3] , [1, 1, -1, 0])
    R3 = reaction( (t,x) -> κ₃*x[3], [0, 1, -1, 1])
    return ChemicalReactionProcess(["Substrate", "Enzyme", "Enzyme-substrate", "Product"], [R1,R2,R3])
end
