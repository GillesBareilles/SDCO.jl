import Base.copy
export PointE, copy, densify, SDCOContext


### Matrix types
const Dense{T} = Array{T, 2} where T<:Number
const DenseSym{T} = Symmetric{T, Dense{T}} where {T<:Real}
const DenseHerm{T} = Hermitian{T, Dense{T}} where {T<:Complex}

const Sparse{T} = SparseArrays.SparseMatrixCSC{T,Int64} where T<:Number
const SparseSym{T} = Symmetric{T, Sparse{T}} where {T<:Real}
const SparseHerm{T} = Hermitian{T, Sparse{T}} where {T<:Complex}

mutable struct PointE{T, U}
    mats::Vector{U}             # Collection of symmetric matrices
    vec::Vector{T}              # R^m array
    dims::Vector{Int}           # Dimension of the sdp cones
end



function PointE(mats::Vector{U}, vec::AbstractArray{T}) where {U<:AbstractMatrix, T<:Number}
    dims = map(x->size(x, 1), mats)

    return PointE(mats, vec, dims)
end


function PointE(dims::AbstractArray{Int}, vecdim::Int, T::DataType, U::DataType)
    mats = Vector{U}(undef, length(dims))

    for (i, dimi) in enumerate(dims)
        if U<:Dense
            mats[i] = zeros(T, dimi, dimi)
        elseif U<: DenseSym
            mats[i] = Symmetric(zeros(T, dimi, dimi))
        elseif U<: DenseHerm
            mats[i] = Hermitian(zeros(T, dimi, dimi))
        elseif U<:Sparse
            mats[i] = spzeros(T, dimi, dimi)
        elseif U<: SparseSym
            mats[i] = Symmetric(spzeros(T, dimi, dimi))
        elseif U<: SparseHerm
            mats[i] = Hermitian(spzeros(T, dimi, dimi))
        else
            @error("PointE(): Unsupported matrix type $U")
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

function densify(x::PointE{T, U}) where {T<:Number, U<:Union{AbstractArray, Symmetric}}
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])

    for (i, mat) in enumerate(x.mats)
        mats[i] = Matrix(mat)
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


################################################################################
##### Context structure
################################################################################
mutable struct SDCOContext{T, matT}
    c::PointE{T, matT}                 # Objective: linear part
    A::Vector{PointE{T, matT}}         # Ctr : linear operator
    b::Array{T, 1}                  #       rhs

    mr::Int                         # Real valued constraints are indexed from 1 to mr
    mc::Int                         # Clpx valued constraints are indexed from mr+1 to mc

    nsdpvar::Vector{Int}            # Size of the several SDP cones
    nscalvar::Int                   # Number of scalar variables (of type T)
    m::Int                          # Nb of constraints
    nc::Int

    options::OrderedDict
end

function SDCOContext(c::PointE{T, matT}, A::Vector{PointE{T, matT}}, b::Array{T, 1}; options=OrderedDict()) where {T<:Number, matT}
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

    defoptions = get_defaultoption()

    for (opt, val) in options
        if haskey(defoptions, opt)
            defoptions[opt] = val
        else
            @warn "option $opt not supported"
        end
    end

    return SDCOContext(c, A, b, 0, 0, nsdpvar, nscalvar, m, nc, defoptions)
end

function get_defaultoption()
    return OrderedDict{Symbol, Any}(
                    :opt_ε => 1e-10,
                    :opt_maxit => 60,
                    :opt_outlev => 1,
    )
end
