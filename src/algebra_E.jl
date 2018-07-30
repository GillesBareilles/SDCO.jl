import LinearAlgebra: dot, norm, product
import Base: +, -, inv, *
export dot, norm, add!, product, -, +, *
export hadamard, inv

"""
Scalar product over E, primal space
"""
function dot(pt1::PointE{T, Sparse{T}}, pt2::PointE{T, Sparse{T}}) where T<:Number
    @assert pt1.dims == pt2.dims

    innerprod = dot(pt1.vec, pt2.vec)

    for n=1:length(pt1.dims)
        innerprod += dot(Symmetric(pt1.mats[n], :L), Symmetric(pt2.mats[n], :L))
    end

    return innerprod
end

function dot(pt1::PointE{T, Sparse{T}}, pt2::PointE{T, Dense{T}}) where T<:Number
    @assert pt1.dims == pt2.dims
    
    innerprod = dot(pt1.vec, pt2.vec)

    for n=1:length(pt1.dims)
        innerprod += dot(Symmetric(pt1.mats[n], :L), pt2.mats[n])
    end

    return innerprod
end

function dot(pt1::PointE{T, Dense{T}}, pt2::PointE{T, Sparse{T}}) where T<:Number
    return dot(pt2, pt1)
end

function dot(pt1::PointE{T, Dense{T}}, pt2::PointE{T, Dense{T}}) where T<:Number
    innerprod = dot(pt1.vec, pt2.vec)

    for n=1:length(pt1.mats)
        innerprod += dot(pt1.mats[n], pt2.mats[n])
    end

    return innerprod
end

function norm(pt1::PointE{T}) where T<:Number
    sqrt(dot(pt1, pt1))
end





function add!(x::PointE{T, U}, y::PointE{T, U}) where {T<:Number, U}
    for (i, mat) in enumerate(y.mats)
        x.mats[i] += mat
    end

    for (i, yi) in enumerate(y.vec)
        x.vec[i] += yi
    end
    nothing
end

function add!(x::PointE{T, Dense{T}}, y::PointE{T, Sparse{T}}) where {T<:Number, U}
    for (i, mat) in enumerate(y.mats)
        x.mats[i] += Matrix(mat)
    end

    for (i, yi) in enumerate(y.vec)
        x.vec[i] += yi
    end
    nothing
end

function add(x::PointE{T, U}, y::PointE{T, U}) where {T<:Number, U}
    mats = Vector{U}(undef, length(x.dims))

    for i=1:length(x.dims)
        mats[i] = x.mats[i] + y.mats[i]
    end

    vec = x.vec .+ y.vec
    return PointE(mats, vec)
end

function add(x::PointE{T, Sparse{T}}, y::PointE{T, Dense{T}}) where {T<:Number}
    res = PointE(x.dims, length(x.vec), T, Dense{T})

    for i=1:length(x.dims)
        res.mats[i] = Matrix(Symmetric(x.mats[i])) + y.mats[i]
    end

    res.vec .= x.vec .+ y.vec
    return res
end

function add(x::PointE{T, Dense{T}}, y::PointE{T, Sparse{T}}) where {T<:Number, U}
    return add(y, x)
end

+(x::PointE, y::PointE) = add(x, y)
-(x::PointE, y::PointE) = add(x, -y)

function product!(x::PointE{T, U}, lambda::T) where {T<:Number, U}
    for (i, mat) in enumerate(x.mats)
        x.mats[i].data *= lambda
    end

    x.vec *= lambda
    nothing
end

function product(x::PointE{T, U}, lambda::T) where {T<:Number, U}
    mats = Vector{U}(undef, length(x.dims))

    for (i, mati) in enumerate(x.mats)
        mats[i] = lambda * x.mats[i]
    end

    vec = lambda .* x.vec
    return PointE(mats, vec)
end

*(x::PointE{T, U}, lambda::T) where {T<:Number, U} = product(x, lambda)
*(lambda::T, x::PointE{T, U}) where {T<:Number, U} = product(x, lambda)

function -(x::PointE{T}) where T<:Number
    return product(x, convert(T, -1))
end

"""
    Compute the inverse of E, assuming `x` lies in strict feasible space
"""
function inv(x::PointE{T, Dense{T}}) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])

    for (i, mat) in enumerate(x.mats)
        mats[i] = inv(cholesky(mat))
    end

    vec = 1 ./ x.vec
    return PointE(mats, vec)
end

function hadamard(x::PointE{T, U}, y::PointE{T, U}) where {T<:Number, U}
    res = PointE(x.dims, length(x.vec), T, U)

    for (i, mati) in enumerate(x.mats)
        res.mats[i] = x.mats[i] .* y.mats[i]
    end

    res.vec = x.vec .* y.vec
    return res
end


function hadamard(x::PointE{T, Dense{T}}, y::PointE{T, Sparse{T}}, z::PointE{T, Dense{T}}; transposefirst=false) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])
    if transposefirst
        for matind=1:length(x.dims)
            n = size(x.mats[matind], 1)
            mats[matind] = zeros(n, n)
            
            for i=1:n, j=1:n
                mats[matind][i, j] = x.mats[matind][j, i] * y.mats[matind][max(i, j), min(i, j)] * z.mats[matind][i, j]
            end
        end
    else
        for n=1:length(x.dims)
            mats[n] = x.mats[n] .* y.mats[n] .* z.mats[n]
        end
    end

    vec = x.vec .* y.vec .* z.vec

    return PointE(mats, vec)
end

function hadamard(x::PointE{T, Dense{T}}, y::PointE{T, Dense{T}}, z::PointE{T, Dense{T}}; transposefirst=false) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])
    if transposefirst
        for matind=1:length(x.dims)
            n = size(x.mats[matind], 1)
            mats[matind] = zeros(n, n)
            
            for i=1:n, j=1:n
                mats[matind][i, j] = x.mats[matind][j, i] * y.mats[matind][max(i, j), min(i, j)] * z.mats[matind][i, j]
            end
        end
    else
        for n=1:length(x.dims)
            mats[n] = x.mats[n] .* y.mats[n] .* z.mats[n]
        end
    end

    vec = x.vec .* y.vec .* z.vec

    return PointE(mats, vec)
end
