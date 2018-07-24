import LinearAlgebra: dot, norm, product
import Base: -
export dot, norm, add!, product, -


function dot(mat1::SymMat{T}, mat2::SymMat{T}) where T<:Number
    res = 0

    for i=1:mat1.n
        res += mat1[i, i] * mat2[i, i]
    end

    for j=1:(mat1.m-1), i=(j+1):mat1.n
        res += 2*mat1[i, j] * mat2[i, j]
    end

    return res
end

"""
Scalar product over E, primal space
"""
function dot(pt1::PointE{T}, pt2::PointE{T}) where T<:Number
    innerprod = dot(pt1.vec, pt2.vec)

    @assert pt1.dims == pt2.dims

    for n=1:length(pt1.dims)
        innerprod += dot(pt1.mats[n], pt2.mats[n])
    end

    innerprod += dot(pt1.vec, pt2.vec)

    return innerprod
end

function norm(pt1::PointE{T}) where T<:Number
    sqrt(dot(pt1, pt1))
end


function add!(x::PointE{T}, y::PointE{T}) where T<:Number
    for (i, mat) in enumerate(y.mats)
        x.mats[i] += mat
    end

    for (i, yi) in enumerate(y.vec)
        x.vec[i] += yi
    end
end



function product!(x::PointE{T}, lambda::T) where T<:Number
    for (i, mat) in enumerate(x.mats)
        x.mats[i] *= lambda
    end

    x.vec *= lambda
    return nothing
end

function product(x::PointE{T}, lambda::T) where T<:Number
    res = PointE(x.dims, length(x.vec), T)

    for (i, mati) in enumerate(x.mats)
        res.mats[i] = lambda * x.mats[i]
    end

    res.vec .= lambda .* x.vec
    return res
end

function -(x::PointE{T}) where T<:Number
    return product(x, convert(T, -1))
end

# """
#     Compute the inverse of E, assuming `x` lies in strict feasible space
# """
# function inv(x::PointE{T}) where T<:Number
#     matsinvs = Vector{SymMat{T}}()

#     for (i, mat) in enumerate(x.mats)
#         matsinvs[i] = inv(mat)
#     end

#     vecinv = 1 ./ x.vec
#     return PointE(matsinvs, vecinv)
# end


# function ⊗(x::PointE, y::PointE)

# end