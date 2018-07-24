export get_primobj, get_primslacks, get_primfeaserr
export get_dualobj, get_dualslacks, get_dualfeaserr
export evaluate


function evaluate(A::Vector{PointE{T}}, x::PointE{T}) where T<:Number
    res = zeros(T, length(A))

    for (i, ai) in enumerate(A)
        @assert ai.dims == x.dims
        res[i] = dot(ai, x)
    end

    return res
end

function evaluate(A::Vector{PointE{T}}, y::Vector{T}) where T<:Number
    @assert length(A) == length(y)

    res = PointE(first(A).dims, length(first(A).vec), T)
    
    for (i, ai) in enumerate(A)
        @assert res.dims == ai.dims
        
        add!(res, product(A[i], y[i]))
    end

    return res
end


function get_primobj(pb::SDCOContext{T}, x::PointE{T}) where T<:Number
    @assert pb.c.dims == x.dims
    return dot(pb.c, x)
end

function get_dualobj(pb::SDCOContext{T}, y::Vector{T}) where T<:Number
    @assert length(y) == length(b)
    return dot(b, y)
end



function get_primslacks(pb::SDCOContext{T}, x::PointE{T}) where T<:Number
    return evaluate(pb.A, x) .- pb.b
end

function get_dualslacks(pb::SDCOContext{T}, y::Vector{T}, s::PointE{T}) where T<:Number
    slacks = evaluate(pb.A, y)
    add!(slacks, s)
    add!(slacks, -pb.c)
    return slacks
end


function get_primfeaserr(pb::SDCOContext{T}, x::PointE{T}) where T<:Number
    return norm(get_primslacks(pb, x))
end

function get_dualfeaserr(pb::SDCOContext{T}, y::Vector{T}, s::PointE{T}) where T<:Number
    return norm(get_dualslacks(pb, y, s))
end