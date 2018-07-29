export NesterovToddstep, NTget_g, NTget_w, NTget_M

function NesterovToddstep(pb::SDCOContext, x::PointE{T}, y, s, mu::Float64) where T<:Number
    io = stdout
    println(io, "------ NesterovToddstep ------")
    println(io, "Input point :")
    println(io, "Primal / dual obj      :    ", get_primobj(pb, x), " / ", get_dualobj(pb, y))
    println(io, "Primal feasability err :    ", get_primfeaserr(pb, x))
    println(io, "Dual feasability err   :    ", get_dualfeaserr(pb, y, s))
    println(io, "Min_K(x)               :    ", min(pb, x))
    println(io, "Min_K(s)               :    ", min(pb, s))
    println(io, "x . s                  :    ", hadamard(x, s))

    # Step 1. Compute necessary quatities
    g = NTget_g(pb, x, s)
    w = NTget_w(pb, g)
    sinv = inv(s)
    muxs = x - product(sinv, mu)

    # Step 2. Solve M.dy = A(x - \mu inv(s))
    M = NTget_M(pb, g)
    rhs = evaluate(pb.A, muxs)

    # Choleski fact, solve
    dy = cholesky(M) \ rhs

    # Step 3. 
    ds = - evaluate(pb.A, dy)

    dx = - muxs - hadamard(w, ds, w)

    # Step 4.
    xstep = x+dx
    ystep = y+dy
    sstep = s+ds

    println(io, "NT step computed: z+dz")
    println(io, "Primal / dual obj      :    ", get_primobj(pb, xstep), " / ", get_dualobj(pb, ystep))
    println(io, "Primal feasability err :    ", get_primfeaserr(pb, xstep))
    println(io, "Dual feasability err   :    ", get_dualfeaserr(pb, ystep, sstep))
    println(io, "Min_K(x)               :    ", min(pb, xstep))
    println(io, "Min_K(s)               :    ", min(pb, sstep))
    
    # Check that z = (x, y, s) is strict feasable
    # Check that z+dz is strict feasable too.

    return dx, dy, ds
end



function NTget_g(pb::SDCOContext, x::PointE{T, Dense{T}}, s::PointE{T, Dense{T}}) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])

    ## Deal with SDP part
    for j in eachindex(x.mats)
        xj_fact = LinearAlgebra.cholesky(x.mats[j], check = true)
        sj_fact = LinearAlgebra.cholesky(s.mats[j], check = true)

        svd_xs = svd(sj_fact.U * xj_fact.L)

        mats[j] = xj_fact.L * transpose(svd_xs.Vt) * Diagonal( 1 ./ sqrt.(svd_xs.S))
    end

    ## Vectorial part
    vec = sqrt.( sqrt.(x.vec) ./ sqrt.(s.vec) )
    
    return PointE(mats, vec)
end


function NTget_w(pb::SDCOContext, g::PointE{T, Dense{T}}) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in g.dims])

    for (i, mat) in enumerate(g.mats)
        mats[i] = mat * transpose(mat)
    end

    vec = g.vec .* g.vec

    return PointE(mats, vec)
end



function NTget_M(pb::SDCOContext, g::PointE{T}) where T<:Number
    m = pb.m
    M = zeros(m, m)

    vectors = Vector{PointE{T, Dense{T}}}(undef, pb.m)
    for i=1:m
        vectors[i] = hadamard(g, pb.A[i], g, transposefirst = true)
    end

    for i=1:m, j=1:m
        M[i, j] = dot(vectors[i], vectors[j])
    end
    return M
end