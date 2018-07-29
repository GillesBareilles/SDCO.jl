import Base.min

export get_primobj, get_primslacks, get_primfeaserr
export get_dualobj, get_dualslacks, get_dualfeaserr
export evaluate
export min


function evaluate(A::Vector{PointE{T, U}}, x::PointE{T, V}) where {T<:Number, U, V}
    res = zeros(T, length(A))

    for (i, ai) in enumerate(A)
        res[i] = dot(ai, x)
    end

    return res
end

function evaluate(A::Vector{PointE{T, Sparse{T}}}, y::Vector{T}) where T<:Number
    @assert length(A) == length(y)

    dims = first(A).dims
    res = PointE(dims, length(first(A).vec), T, Dense{T})

    for (i, ai) in enumerate(A)
        add!(res, densify(product(A[i], y[i])))
    end

    return res
end

function evaluate(A::Vector{PointE{T, Dense{T}}}, y::Vector{T}; evaltype::DataType=Dense{T}) where T<:Number
    @assert length(A) == length(y)

    dims = first(A).dims
    res = PointE(dims, length(first(A).vec), T, evaltype)

    for (i, ai) in enumerate(A)
        add!(res, product(A[i], y[i]))
    end

    return res
end

function get_primobj(pb::SDCOContext{T}, x::PointE{T}) where T<:Number
    @assert pb.c.dims == x.dims
    return dot(pb.c, x)
end

function get_dualobj(pb::SDCOContext{T}, y::Vector{T}) where T<:Number
    @assert length(y) == length(pb.b)
    return dot(pb.b, y)
end



function get_primslacks(pb::SDCOContext{T}, x::PointE{T}) where T<:Number
    return evaluate(pb.A, x) .- pb.b
end

function get_dualslacks(pb::SDCOContext{T}, y::Vector{T}, s::PointE{T}) where T<:Number
    slacks = copy(s)
    add!(slacks, evaluate(pb.A, y))
    add!(slacks, -pb.c)
    return slacks
end


function get_primfeaserr(pb::SDCOContext{T}, x::PointE{T}) where T<:Number
    return norm(get_primslacks(pb, x))
end

function get_dualfeaserr(pb::SDCOContext{T}, y::Vector{T}, s::PointE{T}) where T<:Number
    return norm(get_dualslacks(pb, y, s))
end

function min(pb::SDCOContext{T}, x::PointE{T, Dense{T}}) where {T<:Number, U<:Dense}
    minx = Inf
    if length(x.vec) > 0
        minx = minimum(x.vec)
    end

    for (i, mat) in enumerate(x.mats)
        eigmin = LinearAlgebra.eigmin(mat)
        minx = min(minx, eigmin)
    end

    return minx
end