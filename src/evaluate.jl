import Base.min

export get_primobj, get_primslacks, get_primfeaserr
export get_dualobj, get_dualslacks, get_dualfeaserr
export evaluate
export min, mu, delta
export issol


function evaluate(A::Vector{PointE{T, U}}, x::PointE{T}) where {T<:Number, U<:AbstractArray}
    res = zeros(T, length(A))

    for (i, ai) in enumerate(A)
        res[i] = dot(ai, x)
    end

    return res
end

function evaluate(A::Vector{PointE{T, U}}, y::Vector{T}) where {T<:Number, U<:AbstractArray}
    @assert length(A) == length(y)

    dims = first(A).dims
    res = PointE(dims, length(first(A).vec), T, Dense{T})

    for (i, ai) in enumerate(A)
        add!(res, densify(product(A[i], y[i])))
    end

    return res
end

# function evaluate(A::Vector{PointE{T}}, y::Vector{T}; evaltype::DataType=Dense{T}) where T<:Number
#     @assert length(A) == length(y)
#
#     dims = first(A).dims
#     res = PointE(dims, length(first(A).vec), T, evaltype)
#
#     for (i, ai) in enumerate(A)
#         add!(res, product(A[i], y[i]))
#     end
#
#     return res
# end

function get_primobj(pb::SDCOContext{T}, x::PointE{T}) where T<:Number
    @assert pb.c.dims == x.dims
    return dot(pb.c, x)
end

get_primobj(pb::SDCOContext{T}, z::PointPrimalDual{T}) where T<:Number = get_primobj(pb, z.x)

function get_dualobj(pb::SDCOContext{T}, y::Vector{T}) where T<:Number
    @assert length(y) == length(pb.b)
    return dot(pb.b, y)
end

get_dualobj(pb::SDCOContext{T}, z::PointPrimalDual{T}) where T<:Number = get_dualobj(pb, z.y)





function get_primslacks(pb::SDCOContext{T}, x::PointE{T}) where T<:Number
    return evaluate(pb.A, x) .- pb.b
end

get_primslacks(pb::SDCOContext{T}, z::PointPrimalDual{T}) where T<:Number = get_primslacks(pb, z.x)

function get_dualslacks(pb::SDCOContext{T}, y::Vector{T}, s::PointE{T}) where T<:Number
    slacks = copy(s)
    add!(slacks, evaluate(pb.A, y))
    add!(slacks, -pb.c)
    return slacks
end

get_dualslacks(pb::SDCOContext{T}, z::PointPrimalDual{T}) where T<:Number = get_dualslacks(pb, z.y, z.s)





function get_primfeaserr(pb::SDCOContext{T}, x::PointE{T}) where T<:Number
    return norm(get_primslacks(pb, x))
end

get_primfeaserr(pb::SDCOContext{T}, z::PointPrimalDual{T}) where T<:Number = get_primfeaserr(pb, z.x)

function get_dualfeaserr(pb::SDCOContext{T}, y::Vector{T}, s::PointE{T}) where T<:Number
    return norm(get_dualslacks(pb, y, s))
end

get_dualfeaserr(pb::SDCOContext{T}, z::PointPrimalDual{T}) where T<:Number = get_dualfeaserr(pb, z.y, z.s)





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
