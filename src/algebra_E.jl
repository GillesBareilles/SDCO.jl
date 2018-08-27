import LinearAlgebra: dot, norm
import Base: +, -, inv, *, transpose, sqrt
export dot, norm, add, add!, product, -, +, *, transpose
export hadamard, inv, sqrt

"""
Scalar product over E, primal space
"""
function dot(pt1::PointE{T}, pt2::PointE{T}) where {T<:Number}
    @assert pt1.dims == pt2.dims

    innerprod = dot(pt1.vec, pt2.vec)

    for n=1:length(pt1.dims)
        innerprod += dot(pt1.mats[n], pt2.mats[n])
    end

    return innerprod
end

function norm(pt1::PointE{T}) where {T<:Number}
    sqrt(dot(pt1, pt1))
end



################################################################################
### Additive algebra
################################################################################

function add!(x::PointE{T}, y::PointE{T}) where {T<:Number}
    for (i, mat) in enumerate(y.mats)
        x.mats[i] = add(x.mats[i], mat)
    end

    for (i, yi) in enumerate(y.vec)
        x.vec[i] += yi
    end
    nothing
end

## Matrix addition
function add(mat1::AbstractArray{T}, mat2::AbstractArray{T}) where {T<:Number}
    return mat1 + mat2
end

function add(mat1::Sparse{T}, mat2::Sparse{T}) where {T<:Number}
    return mat1 + mat2
end

function add(mat1::Symmetric{T}, mat2::Symmetric{T}) where {T<:Number}
    if mat1.uplo == mat2.uplo
        mat1.data += mat2.data
    else
        mat1.data += transpose(mat2.data)
    end
    nothing
end

# function add!(mat1::DenseHerm{T}, mat2::DenseHerm{T}) where {T<:Number}
#     if mat1.uplo == mat2.uplo
#         mat1.data += mat2.data
#     else
#         mat1.data += adjoint(mat2.data)
#     end
#     nothing
# end


function add(x::PointE{T, U}, y::PointE{T, U}) where {T<:Number, U<:AbstractArray}
    point = PointE(x.dims, length(x.vec), T, U)

    for i=1:length(x.dims)
        point.mats[i] = x.mats[i] + y.mats[i]
        # add!(point.mats[i], x.mats[i])
        # add!(point.mats[i], y.mats[i])
    end

    point.vec = x.vec .+ y.vec
    return point
end

function add(x::PointE{T, U}, y::PointE{T, V}) where {T<:Number, U<:AbstractArray, V<:AbstractSparseArray}
    point = PointE(x.dims, length(x.vec), T, U)

    for i=1:length(x.dims)
        point.mats[i] = x.mats[i] + y.mats[i]
    end

    point.vec = x.vec .+ y.vec
    return point
end

function add(x::PointE{T, U}, y::PointE{T, V}) where {T<:Number, U<:AbstractSparseArray, V<:AbstractArray}
    return add(y, x)
end

function add(x::PointE{T, U}, y::PointE{T, V}) where {T<:Number, U<:AbstractArray, V<:Symmetric}
    point = PointE(x.dims, length(x.vec), T, U)

    for i=1:length(x.dims)
        point.mats[i] = x.mats[i] + y.mats[i]
    end

    point.vec = x.vec .+ y.vec
    return point
end

function add(x::PointE{T, U}, y::PointE{T, V}) where {T<:Number, U<:Symmetric, V<:AbstractArray}
    return add(y, x)
end


+(x::PointE, y::PointE) = add(x, y)
-(x::PointE, y::PointE) = add(x, -y)

################################################################################
### Multiplicative algebra
################################################################################

function product!(x::PointE{T}, lambda::T) where {T<:Number}
    for (i, mat) in enumerate(x.mats)
        x.mats[i] *= lambda
    end

    x.vec *= lambda
    nothing
end

function product(x::PointE{T}, lambda::T) where {T<:Number}
    xprod = copy(x)
    product!(xprod, lambda)
    return xprod
end

*(x::PointE{T, U}, lambda::T) where {T<:Number, U} = product(x, lambda)
*(lambda::T, x::PointE{T, U}) where {T<:Number, U} = product(x, lambda)

-(x::PointE{T}) where T<:Number = product(x, convert(T, -1))

################################################################################
### Factorization ops: inverse, sqrt
################################################################################
"""
    Compute the inverse of E, assuming `x` lies in strict feasible space
"""
function inv(x::PointE{T}) where T<:Number
    xinv = PointE(x.dims, length(x.vec), T, Dense{T})

    for (i, mat) in enumerate(x.mats)
        xinv.mats[i] = inv(factorize(mat))
    end

    xinv.vec = 1 ./ x.vec
    return xinv
end

"""
    Compute the square root of E, assuming `x` lies in strict feasible space
"""
function sqrt(x::PointE{T, Union{Dense{T}, DenseSym{T}}}) where T<:Number
    xsqrt = PointE(c.dims, length(x.vec), T, Dense{T})

    for (i, mat) in enumerate(x.mats)
        xsqrt.mats[i] = sqrt(factorize(mat))
    end

    xsqrt.vec = sqrt.(x.vec)
    return xsqrt
end


################################################################################
### Hadamard operations
################################################################################
"""
    hadamard(x::PointE{T, U}, y::PointE{T, U}) where {T<:Number, U<:SDCOAbstractMatrix}

Compute the hadamard product of the points x and y, returning a dense matrix object.
"""
function hadamard(x::PointE{T}, y::PointE{T}) where {T<:Number}
    res = PointE(x.dims, length(x.vec), T, Dense{T})

    for (i, mati) in enumerate(x.mats)
        res.mats[i] = x.mats[i] * y.mats[i]
    end

    res.vec = x.vec .* y.vec
    return res
end

function hadamard(x::PointE{T}, y::PointE{T}, z::PointE{T}; transposefirst=false) where T<:Number
    res = PointE(x.dims, length(x.vec), T, Dense{T})

    hadamard!(x, y, z, res, transposefirst=transposefirst)
    return res
end

function hadamard!(x::PointE{T}, y::PointE{T}, z::PointE{T}, out::PointE{T, Dense{T}}; transposefirst=false) where T<:Number
    if transposefirst
        for n=1:length(x.dims)
            # println("---")
            # @show typeof(x.mats[n])
            # @show typeof(y.mats[n])
            # @show typeof(z.mats[n])
            out.mats[n] = transpose(x.mats[n])
            out.mats[n] *= y.mats[n]
            out.mats[n] *= z.mats[n]
            # println("---")
        end
    else
        for n=1:length(x.dims)
            out.mats[n] = x.mats[n] * y.mats[n] * z.mats[n]
        end
    end

    out.vec = x.vec .* y.vec .* z.vec

    nothing
end

# function hadamard!(x::PointE{T, Dense{T}}, y::PointE{T, Dense{T}}, z::PointE{T, Dense{T}}, out::PointE{T, Dense{T}}; transposefirst=false) where T<:Number
#     if transposefirst
#         for n=1:length(x.dims)
#             out.mats[n] = transpose(x.mats[n])
#             out.mats[n] *= y.mats[n]
#             out.mats[n] *= z.mats[n]
#         end
#     else
#         for n=1:length(x.dims)
#             out.mats[n] = x.mats[n] * y.mats[n] * z.mats[n]
#         end
#     end
#
#     out.vec = x.vec .* y.vec .* z.vec
#
#     nothing
# end

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

#####################################################################
## transpose
#####################################################################
function transpose(x::PointE{T, Dense{T}}) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])

    for (matind, mat) in enumerate(x.mats)
        mats[matind] = transpose(mat)
        # n = x.dims[matind]
        # for i=1:n, j=1:n
        #     mats[matind][j, i] = mat[i, j]
        # end
    end

    return PointE(mats, x.vec)
end
