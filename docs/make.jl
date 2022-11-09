using Documenter, CRMP

makedocs(
    modules = [CRMP],
    sitename = "CRMP.jl",
    authors = "Marc Corstanje",
    doctest = false,
    pages = [ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        "Library" => [
            "Networks" => "networks.md",
            "Reaction times" => "reaction_times.md",
            "Forward simulation" => "forward_simulation.md",
            "Conditional process" => "conditional_process.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/macorstanje/CRMP.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)
