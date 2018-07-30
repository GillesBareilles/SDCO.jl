module SDCO

using SparseArrays, DataStructures, LinearAlgebra, Printf


include("types.jl")
include("algebra_E.jl")
include("evaluate.jl")

include("input.jl")
include("io.jl")

include("NT.jl")

include("test_cases.jl")

end # module
