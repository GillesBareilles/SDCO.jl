module SDCO

using SparseArrays

export greet, greet2


greet() = print("Hello World!")
greet2() = print("Hello World2!")
# greet3() = print("Hello World3!")

SymMat{T} = SparseArrays.SparseMatrixCSC{T, Int64} where T<: Number


mutable struct PointE{T} where T<:Number
    mats::Vector{SymMat}     # Collection of symmetric matrices
    vec::Vector{T}           # R^m array
end


mutable struct PointPrimalDual{T} where T<:Number
    x::PointE{T}
    y:::Vector{T}
    s::PointE{T}
end


mutable struct SDCOcontext
    c::Array{T, 1}          # Objective: linear part
    A::Vector{SymMat}       # Ctr : linear operator
    b::Array{T, 1}          #       rhs

    nsdpvar::Vector{Int}
    nvar::Int
end


include("NT.jl")

end # module
