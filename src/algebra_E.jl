import LinearAlgebra: dot, norm
import Base: +, -, inv, *, transpose, sqrt
export dot, norm, add, add!, product, -, +, *, transpose
export hadamard, inv, sqrt

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

function add!(x::PointE{T, Dense{T}}, y::PointE{T, Sparse{T}}) where {T<:Number}
    for (i, mat) in enumerate(y.mats)
        x.mats[i] += Symmetric(mat, :L)
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
        res.mats[i] = Matrix(Symmetric(x.mats[i], :L)) + y.mats[i]
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
        mats[i] = inv(factorize(mat))
    end

    vec = 1 ./ x.vec
    return PointE(mats, vec)
end

"""
    Compute the square root of E, assuming `x` lies in strict feasible space
"""
function sqrt(x::PointE{T, Dense{T}}) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])

    for (i, mat) in enumerate(x.mats)
        mats[i] = sqrt(factorize(mat))
    end

    vec = sqrt.(x.vec)
    return PointE(mats, vec)
end

function hadamard(x::PointE{T, U}, y::PointE{T, U}) where {T<:Number, U}
    res = PointE(x.dims, length(x.vec), T, U)

    for (i, mati) in enumerate(x.mats)
        res.mats[i] = x.mats[i] * y.mats[i]
    end

    res.vec = x.vec .* y.vec
    return res
end


function hadamard(x::PointE{T, Dense{T}}, y::PointE{T, Sparse{T}}, z::PointE{T, Dense{T}}; transposefirst=false) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])
    if transposefirst
        for n=1:length(x.dims)
            mats[n] = transpose(x.mats[n]) * Symmetric(y.mats[n], :L) * z.mats[n]
        end
    else
        for n=1:length(x.dims)
            mats[n] = x.mats[n] * y.mats[n] * z.mats[n]
        end
    end

    vec = x.vec .* y.vec .* z.vec

    return PointE(mats, vec)
end

function hadamard(x::PointE{T, Dense{T}}, y::PointE{T, Dense{T}}, z::PointE{T, Dense{T}}; transposefirst=false) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])
    if transposefirst
        for n=1:length(x.dims)
            mats[n] = transpose(x.mats[n]) * y.mats[n] * z.mats[n]
        end
    else
        for n=1:length(x.dims)
            mats[n] = x.mats[n] * y.mats[n] * z.mats[n]
        end
    end

    vec = x.vec .* y.vec .* z.vec

    return PointE(mats, vec)
end


function hadamard!(x::PointE{T, Dense{T}}, y::PointE{T, Sparse{T}}, z::PointE{T, Dense{T}}, out::PointE{T, Dense{T}}; transposefirst=false) where T<:Number
    if transposefirst
        for n=1:length(x.dims)
            out.mats[n] = transpose(x.mats[n])
            out.mats[n] *= Symmetric(y.mats[n], :L)
            out.mats[n] *= z.mats[n]
        end
    else
        for n=1:length(x.dims)
            out.mats[n] = x.mats[n] * y.mats[n] * z.mats[n]
        end
    end

    out.vec = x.vec .* y.vec .* z.vec

    nothing
end

function hadamard!(x::PointE{T, Dense{T}}, y::PointE{T, Dense{T}}, z::PointE{T, Dense{T}}, out::PointE{T, Dense{T}}; transposefirst=false) where T<:Number
    if transposefirst
        for n=1:length(x.dims)
            out.mats[n] = transpose(x.mats[n])
            out.mats[n] *= y.mats[n]
            out.mats[n] *= z.mats[n]
        end
    else
        for n=1:length(x.dims)
            out.mats[n] = x.mats[n] * y.mats[n] * z.mats[n]
        end
    end

    out.vec = x.vec .* y.vec .* z.vec

    nothing
end

#####################################################################"
## Point Primal Dual
#####################################################################"
function add(z1::PointPrimalDual, z2::PointPrimalDual)
    return PointPrimalDual(z1.x+z2.x, z1.y+z2.y, z1.s+z2.s)
end

function add!(z1::PointPrimalDual, z2::PointPrimalDual)
    add!(z1.x, z2.x)
    z1.y += z2.y
    add!(z1.s, z2.s)
end

+(z1::PointPrimalDual, z2::PointPrimalDual) = add(z1, z2)
-(z1::PointPrimalDual, z2::PointPrimalDual) = add(z1, -z2)

function product(z::PointPrimalDual{T}, lambda::U) where {T<:Number, U<:Number}
    return PointPrimalDual(z.x * convert(T, lambda), z.y * convert(T, lambda), z.s * convert(T, lambda))
end

*(z::PointPrimalDual, lambda::T) where T<:Number = product(z, lambda)
*(lambda::T, z::PointPrimalDual) where T<:Number = product(z, lambda)

function -(z::PointPrimalDual{T}) where T<:Number
    return product(z, convert(T, -1))
end

function norm(z::PointPrimalDual)
    return norm(z.x) + norm(z.y) + norm(z.s)
end

#####################################################################"
## transpose
#####################################################################"
function transpose(x::PointE{T, Dense{T}}) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])

    for (matind, mat) in enumerate(x.mats)
        n = x.dims[matind]
        for i=1:n, j=1:n
            mats[matind][j, i] = mat[i, j]
        end
    end

    return PointE(mats, x.vec)
end