
""" 
Scalar product over E, primal space
"""
function dot(pt1::PointE, pt2::PointE)

end

norm(pt1::PointE) = sqrt(dot(pt1, pt1))


"""
    Compute the inverse of E, assuming `x` lies in strict feasible space
"""
function inv(x::PointE)

end


function +(x::PointE, y::POintE)

end

function *(x::PointE{T}, ::U) where T<:Real, U<:Real

end

