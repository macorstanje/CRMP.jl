
"""
    reaction{T<:Real}

A reaction, specified through a rate function `λ` of time and space and a difference
vector `ξ`. For example, a unit rate poisson has one reaction, specified as
```julia-repl
julia> plus1 = reaction((t,x) -> x, 1)
```
"""
struct reaction{T<:Real}
    λ::Function
    ξ::Union{T, Array{T,1}}
end

"""
    ChemicalReactionNetwork{T<:Real}

A chemical reaction network is set-up as a tuple of a vector of species 𝒮 with `String`s as input
and a vector of `reaction{T}`s. I still need some way to distinct between process and network
"""
struct ChemicalReactionNetwork{T<:Real}
    𝒮::Union{String, Array{String, 1}}
    ℛ::Union{reaction{T}, Array{reaction{T},1}}
end

"""
    ChemicalReactionProcess{T<:Real}

A chemical reaction process is set-up as a tuple of a vector of species 𝒮 with `String`s as input
and a vector of `reaction{T}`s. I still need some way to distinct between process and network
For example, a poisson process is set-up as follows
```julia-repl
julia> plus1 = reaction((t,x) -> x, 1)
julia> PoissonProcecss = ChemicalReactionProcess(["Counts"], [plus1])
```
Alternatively, if there is just one reaction, or species, one could omit the `Array`.
"""
struct ChemicalReactionProcess{T<:Real}
    𝒮::Union{String, Array{String, 1}}
    ℛ::Union{reaction{T}, Array{reaction{T}, 1}}
end
reaction_array(P::ChemicalReactionProcess{T}) where {T} = typeof(P.ℛ) == reaction{T} ? [P.ℛ] : P.ℛ
getd(P::ChemicalReactionProcess{T}) where {T} = typeof(P.𝒮) == String ? 1 : length(P.𝒮)
"""
    nr_species(P::ChemicalReactionProcess)

Returns how much species a `ChemicalReactionProcess` contains
"""
nr_species(P::ChemicalReactionProcess) = typeof(P.𝒮) == String ? 1 : length(P.𝒮)

"""
    nr_reactions(P::ChemicalReactionProcess)

Returns how much reactions a `ChemicalReactionProcess` contains
"""
nr_reactions(P::ChemicalReactionProcess) = typeof(P.ℛ) == reaction ? 1 : length(P.ℛ)

"""
    PoissonProcess_constantrate(rate::Real)

Returns a ChemicalReactionProcess for the Poisson process with constant rate `rate`.
"""
function PoissonProcess_constantrate(rate::Real)
    @assert rate > 0 "Rate must be positive"
    plus1 = reaction( (t,x) -> rate*x , 1)
    return ChemicalReactionProcess("Counts", plus1)
end

function PureDeathProcess_constantrate(rate::Real)
    @assert rate > 0 "Rate must be positive"
    minus1 = reaction( (t,x) -> rate*x, -1)
    return ChemicalReactionProcess("Counts", minus1)
end
"""
    BirthDeathProcess(birth_rate::Real , death_rate::Real)

Returns a ChemicalReactionProcess for the birth-death process with parameters
`birth_rate` and `death_rate`
"""
function BirthDeathProcess(birth_rate::Real , death_rate::Real)
    @assert min(birth_rate , death_rate) > 0 "All rates must be positive"
    plus1 = reaction( (t,x) -> x*birth_rate, 1)
    minus1 = reaction( (t,x) -> x*death_rate, -1)
    return ChemicalReactionProcess("Individuals", [plus1, minus1])
end
BirthDeathProcess(θ::Array{T,1}) where {T<:Real} = BirthDeathProcess(θ[1],θ[2])

"""
    GTT(κ₁::T ,κ₂::T, dₘ::T, dₚ::T) where {T<:Real}

Returns a ChemicalReactionProcess for Gene transcriptiona and translation as described
in section 2.1.something of Anderson & Kurtz with
- transcription rate `κ₁`
- translation rate `κ₂`
- degradation rate of mRNA `dₘ`
- degradation rate of protein `dₚ`
"""
function GTT(κ₁::T ,κ₂::T, dₘ::T, dₚ::T) where {T<:Real}
    @assert min(κ₁,κ₂,dₘ,dₚ) > 0 "All rate parameters must be positive"
    Transcription = reaction( (t,x) -> κ₁*x[1] , [0, 1, 0])
    Translation = reaction( (t,x) -> κ₂*x[2], [0, 0, 1])
    Degradation_mRNA = reaction( (t,x) -> dₘ*x[2] , [0, -1, 0])
    Degradation_Protein = reaction( (t,x) -> dₚ*x[3] , [0, 0, -1])
    return ChemicalReactionProcess(["Gene", "mRNA", "Protein"], [Transcription, Translation, Degradation_mRNA, Degradation_Protein])
end

"""
    GTT(κ₁::T ,κ₂::T, κ₃::T, dₘ::T, dₚ::T, dD::T) where {T<:Real}

Similar to `GTT` but with the inclusion of dimerization with dimerization rate `κ₃`
and degradation rate of dimer `dD`
"""
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
function GTT(θ::Array{T,1}) where {T<:Real} 
    return length(θ)==4 ? GTT(θ[1],θ[2],θ[3],θ[4]) : GTT(θ[1],θ[2],θ[3],θ[4],θ[5],θ[6])
end

"""
    Schlogl(κ₁::T,κ₂::T,κ₃::T,κ₄:T) where {T<:Real}

Schlogl model (#X, #A,#B). 
"""
function Schlogl(κ₁::T,κ₂::T,κ₃::T,κ₄::T) where {T<:Real}
    R1 = reaction( (t,x) -> κ₁*x[2]*x[1]*(x[1]-1)/2, [1,-1,0])              # A + 2X -> 3X
    R2 = reaction( (t,x) -> κ₂*x[1]*(x[1]-1)*(x[1]-2)/6 , [-1,1,0])         # 3X -> A + 2X
    R3 = reaction( (t,x) -> κ₃*x[3], [1,0,-1])                              # B -> X 
    R4 = reaction( (t,x) -> κ₄*x[1], [-1,0,1])                              # X -> B
    return ChemicalReactionProcess(["X", "A", "B"], [R1,R2,R3,R4])
end


"""
    AR(G::Int64, κ₁::T,κ₂::T,κ₃::T,κ₄::T,κ₅::T,κ₆::T,κ₇::T,κ₈::T) where {T<:Real}

Can be used for the autoregulatory model, see e.g. example 3 of Sherlock & Golightly (2023)
"""

function AR(G::Int64, κ₁::T,κ₂::T,κ₃::T,κ₄::T,κ₅::T,κ₆::T,κ₇::T,κ₈::T) where {T<:Real}
    R1 = reaction( (t,x) -> κ₁*(G-x[4])*x[3], [0,0,-1, 1,-1])
    R2 = reaction( (t,x) -> κ₃*(G-x[3]), [0,1,0,0,0])
    R3 = reaction( (t,x) -> κ₂*x[4], [0,0,1,-1,1])
    R4 = reaction( (t,x) -> κ₄*x[1], [0,1,0,0,0])
    R5 = reaction( (t,x) -> κ₅*x[2]*(x[2]-1)/2, [0,-2,1,0,0])
    R6 = reaction( (t,x) -> κ₇*x[1], [-1,0,0,0,0])
    R7 = reaction( (t,x) -> κ₆*x[3], [0,2,-1,0,0])
    R8 = reaction( (t,x) -> κ₈*x[2], [0,-1,0,0,0] )
    return ChemicalReactionProcess(["RNA", "P", "P₂", "DNA⋅P₂", "DNA"], [R1,R2,R3,R4,R5,R6,R7,R8])
end

function AR(G::Int64,θ::Array{T,1}) where {T<:Real}
    @assert length(θ) == 8 "parameter vector must be of size 8"
    return AR(G, θ[1],θ[2],θ[3],θ[4],θ[5],θ[6],θ[7],θ[8])
end

"""
    viral_infection(κ₁::T, κ₂::T, κ₃::T, κ₄::T, κ₅::T, κ₆::T) where {T<:Real}

ChemicalReactionProcess for viral infection, See e.g. section 2.1.2 of Anderson & Kurtz
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

viral_infection(θ::Array{T,1}) where {T<:Real} = viral_infection(θ[1],θ[2],θ[3],θ[4],θ[5],θ[6])

"""
    enzyme_kinetics(κ₁::T, κ₂::T, κ₃::T) where {T<:Real}

See e.g. section 2.1.3 of Anderson & Kurtz
"""
function enzyme_kinetics(κ₁::T, κ₂::T, κ₃::T) where {T<:Real}
    @assert min(κ₁,κ₂,κ₃) > 0 "All rate parameters must be positive"
    R1 = reaction( (t,x) -> κ₁*x[1]*x[2] , [-1, -1, 1, 0])
    R2 = reaction( (t,x) -> κ₂*x[3] , [1, 1, -1, 0])
    R3 = reaction( (t,x) -> κ₃*x[3], [0, 1, -1, 1])
    return ChemicalReactionProcess(["Substrate", "Enzyme", "Enzyme-substrate", "Product"], [R1,R2,R3])
end
enzyme_kinetics(θ::Array{T,1}) where {T<:Real} = enzyme_kinetics(θ[1],θ[2],θ[3])

# """
#     enzyme_kinetics_degenrate(κ₁::T, κ₂::T, κ₃::T) where {T<:Real}

# Models only the first and fourth component of `enzyme_kinetics` to create an elliptic process. Utilizes computation rules
# `` x_2 = x_{0,2} + x_1+x_4`` and ``x_3 = x_{0,3} - (x_1+x_4)``
# """
# function enzyme_kinetics_degenrate(x₀, κ₁, κ₂, κ₃)
#     @assert min(κ₁,κ₂,κ₃) > 0 "All rate parameters must be positive"
#     R1 = reaction( (t,x) -> κ₁*x[1]*(x₀  x[1]+x[2]) , [-1, -1, 1, 0])
#     R2 = reaction( (t,x) -> κ₂*x[3] , [1, 1, -1, 0])
#     R3 = reaction( (t,x) -> κ₃*x[3], [0, 1, -1, 1])
# end


"""
    lotka_volterra(κ₁, κ₂, κ₃)

well known, pred -> ∅ , prey -> 2prey and pred + prey = 2pred
"""
function lotka_volterra(κ₁::T, κ₂::T, κ₃::T) where {T<:Real}
    @assert min(κ₁,κ₂,κ₃) > 0 "All rate parameters must be positive"
    R1 = reaction( (t,x) -> κ₁*x[1] , [-1, 0])
    R2 = reaction( (t,x) -> κ₂*x[2] , [0, 1])
    R3 = reaction( (t,x) -> κ₃*x[1]*x[2] , [1, -1])
    return ChemicalReactionProcess(["Predator", "Prey"], [R1,R2,R3])
end
