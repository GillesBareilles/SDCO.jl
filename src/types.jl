import Base.copy
export PointE, SDCOContext, copy, densify

# const SymMat{T} = SparseMatrixCSC{T, Int64}

const Dense{T} = Array{T, 2}
const Sparse{T} = SparseArrays.SparseMatrixCSC{T,Int64}

const SymSparse{T} = Symmetric{T, Sparse{T}}

mutable struct PointE{T, U}
    mats::Vector{U}             # Collection of symmetric matrices
    vec::Vector{T}              # R^m array
    dims::Vector{Int}           # Dimension of the sdp cones
end

function PointE(mats::Vector{Sparse{T}}, vec::AbstractArray{T}) where T<:Number
    dims = map(x->size(x, 1), mats)

    ## Check that sparse underlying matrix is lower triangular.
    for (ind, mat) in enumerate(mats)
        rows = SparseArrays.rowvals(mat)
        m, n = size(mat)
        for i = 1:n
            for j in nzrange(mat, i)
                if rows[j] < i
                    @error("PointE(): dropping non lower triangular term ($i, $(rows[j])), matrix $ind.")
                    mat[rows[j], i] = 0
                end
            end
        end

        dropzeros!(mat)
    end

    return PointE(mats, vec, dims)
end

function PointE(mats::Vector{Dense{T}}, vec::AbstractArray{T}) where T<:Number
    dims = map(x->size(x, 1), mats)

    return PointE(mats, vec, dims)
end

function PointE(dims::AbstractArray{Int}, vecdim::Int, T::DataType, U::DataType)
    mats = Vector{U}(undef, length(dims))
    
    if U<:Dense
        for (i, dimi) in enumerate(dims)
            mats[i] = zeros(T, dimi, dimi)
        end
    else
        for (i, dimi) in enumerate(dims)
            mats[i] = spzeros(T, dimi, dimi)
        end
    end
    
    return PointE(mats, zeros(T, vecdim), dims)
end


function copy(x::PointE{T, U}) where {T<:Number, U}
    mats = Vector{U}(undef, length(x.dims))

    for (i, mat) in enumerate(x.mats)
        mats[i] = copy(mat)
    end

    vec = copy(x.vec)
    return PointE(mats, vec)
end

function densify(x::PointE{T, Sparse{T}}) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])

    for (i, mat) in enumerate(x.mats)
        mats[i] = Matrix(Symmetric(mat, :L))
    end

    return PointE(mats, copy(x.vec))
end

mutable struct PointPrimalDual{T, U}
    x::PointE{T, U}
    y::Vector{T}
    s::PointE{T, U}
end

function PointPrimalDual(dims::AbstractArray{Int}, vecdim::Int, m::Int, T::DataType, U::DataType)
    return PointPrimalDual(PointE(mats, vec, dims), zeros(T, m), PointE(mats, vec, dims))
end


mutable struct SDCOContext{T, U}
    c::PointE{T, U}                 # Objective: linear part
    A::Vector{PointE{T, U}}         # Ctr : linear operator
    b::Array{T, 1}                  #       rhs

    nsdpvar::Vector{Int}            # Size of the several SDP cones
    nscalvar::Int                   # Number of scalar variables (of type T)
    m::Int                          # Nb of constraints
    nc::Int

    options::OrderedDict
end

function SDCOContext(c::PointE{T, U}, A::Vector{PointE{T, U}}, b::Array{T, 1}; options=OrderedDict()) where {T<:Number, U}
    @assert length(A) == length(b)
    m = length(A)

    nsdpvar = c.dims
    nscalvar = length(c.vec)

    # check all PointE elts are compatible
    for ai in A
        @assert nsdpvar == ai.dims
        @assert nscalvar == length(ai.vec)
    end

    nc = sum(nsdpvar) + nscalvar


    return SDCOContext(c, A, b, nsdpvar, nscalvar, m, nc, OrderedDict())
end
