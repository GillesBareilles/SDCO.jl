export SymMat, PointE, SDCOContext

const SymMat{T} = SparseMatrixCSC{T, Int64}


mutable struct PointE{T}
    mats::Vector{SymMat{T}}     # Collection of symmetric matrices
    vec::Vector{T}              # R^m array
    dims::Vector{Int}           # Dimension of the sdp cones
end

function PointE(ai::AbstractArray{Int}, aj::AbstractArray{Int}, ak::AbstractArray{Int}, aijk::AbstractArray{T}, dims::AbstractArray{Int}, vec::AbstractArray{T}) where T<:Number
    @assert length(ai) == length(aj)
    @assert length(ai) == length(ak)
    @assert length(ai) == length(aijk)

    vecmat = Vector{SymMat{T}}(undef, length(dims))

    for (j, nj) in enumerate(dims)
        vecmat[j] = spzeros(T, nj, nj)
    end

    for i=1:length(ai)
        vecmat[ai[i]][aj[i], ak[i]] = aijk[i]
    end

    for spmat in vecmat
        dropzeros(spmat)
    end

    return PointE(vecmat, vec, dims)
end

function PointE(dims::AbstractArray{Int}, vecdim::Int, T)
    mats = Vector{SymMat{T}}(undef, length(dims))
    
    for (i, dimi) in enumerate(dims)
        mats[i] = spzeros(T, dimi, dimi)
    end
    
    return PointE(mats, zeros(T, vecdim), dims)
end


# mutable struct PointPrimalDual{T}
#     x::PointE{T}
#     y:::Vector{T}
#     s::PointE{T}
# end

mutable struct SDCOContext{T}
    c::PointE{T}                # Objective: linear part
    A::Vector{PointE{T}}        # Ctr : linear operator
    b::Array{T, 1}              #       rhs

    nsdpvar::Vector{Int}        # Size of the several SDP cones
    nscalvar::Int               # Number of scalar variables (of type T)
    m::Int                      # Nb of constraints
end

function SDCOContext(ai::Vector{Int}, aj::Vector{Int}, ak::Vector{Int}, al::Vector{Int}, aijkl::Vector{T},  # SDP coefficients
                    veci::Vector{Int}, vecj::Vector{Int}, vecij::Vector{T},                                 # Vector coefficients
                    b::Vector{T},                                                                           # Constraints right hand side
                    ci::Vector{Int}, cj::Vector{Int}, ck::Vector{Int}, cijk::Vector{T},                     # Objective SDP part
                    cvecj::Vector{Int}, cvecval::Vector{T}) where T<:Number                                 # Objective vectorial part

    @assert length(ai) == length(aj)
    @assert length(ai) == length(ak)
    @assert length(ai) == length(al)
    @assert length(ai) == length(aijkl)

    @assert length(veci) == length(vecj)
    @assert length(veci) == length(vecij)

    @assert length(ci) == length(cj)
    @assert length(ci) == length(ck)
    @assert length(ci) == length(cijk)

    @assert length(cvecj) == length(cvecval)

    m = maximum(union(Set(ai), Set(veci)))
    nnz_sdpctr = length(ai)
    nnz_sdpobj = length(ci)

    # computing SDP cones dimensions
    blockid_to_var = SortedDict{Int, SortedSet{Int}}()
    scalvars = SortedDict{Int, Int}()
    for n=1:nnz_sdpctr
        blockid = aj[n]
        !haskey(blockid_to_var, blockid) && (blockid_to_var[blockid] = SortedSet{Int}())

        if aijkl[n] != 0                # Note: zero on floats ?
            push!(blockid_to_var[blockid], ak[n])
            push!(blockid_to_var[blockid], al[n])
        end
    end

    for n=1:nnz_sdpobj
        blockid = ci[n]

        !haskey(blockid_to_var, blockid) && (blockid_to_var[blockid] = SortedSet{Int}())

        if cijk[n] != 0                # Note: zero on floats ?
            push!(blockid_to_var[blockid], cj[n])
            push!(blockid_to_var[blockid], ck[n])
        end
    end

    
    nsdpvar = zeros(Int, maximum(collect(keys(blockid_to_var))))

    for (i, vars) in blockid_to_var
        nsdpvar[i] = maximum(collect(vars))          # Note: 'presolve' -> length(vars) ?
    end

    ## Computing nscalvar
    nscalvar = length(union(Set(vecj), Set(cvecj)))
    # nscalvar = maximum(collect(union(Set(vecj), Set(cvecj))))     # Note: check for nul coefficients

    ## Assembling constraints
    A = Vector{PointE{T}}(undef, m)
    for i=1:m
        sdpinds = findall(x->x==i, ai)
        vecinds = findall(x->x==i, veci)

        vec = zeros(T, nscalvar)
        for n in vecinds
            vec[vecj[n]] = vecij[n]
        end

        A[i] = PointE(view(aj, sdpinds), view(ak, sdpinds), view(al, sdpinds), view(aijkl, sdpinds), nsdpvar, vec)
    end

    vec = zeros(T, nscalvar)
    for (n, val) in zip(cvecj, cvecval)
        vec[n] = val
    end

    c = PointE(ci, cj, ck, cijk, nsdpvar, vec)
    
    pb = SDCOContext(c, A, b, nsdpvar, nscalvar, m)
end