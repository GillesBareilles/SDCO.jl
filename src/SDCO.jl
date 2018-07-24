module SDCO

using SparseArrays, DataStructures, LinearAlgebra


include("types.jl")
include("NT.jl")
include("algebra_E.jl")
include("evaluate.jl")

include("input.jl")
include("io.jl")

include("test_cases.jl")

end # module
