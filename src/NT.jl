export NesterovToddstep, NTget_g, NTget_w, NTget_M

function NTpredcorr(pb::SDCOContext, z::PointPrimalDual{T, Dense{T}}) where T<:Number

    mu_z = mu(pb, z)

    if mu_z < 1e-5
        return 1
    end

    ## Corrector step
    dc = NesterovToddstep(pb, z, mu_z)
    zc = z + dc

    ## Predictor step
    da = NesterovToddstep(pb, z, 0.)
    
    gtr = transpose(g)
    dasdax = hadamard(da.s, da.x)
    Enorm = norm(hadamard(inv(g), dasdax, g) + hadamard(gtr, dasdax, inv(gtr)))
    alpha = 2 / (1 + sqrt(1+13*Enorm/(2*mu_zc)))
    
    zp = zc + alpha * da


    mu_zc = mu(zc)
    mu_zp = mu(zp)
    @show mu_zp
    @show (1-alpha) * mu_zc + alpha * mu_z

    @show mu_z - mu_zc
    @show mu_zp - (1-alpha) * mu_z
end


function NesterovToddstep(pb::SDCOContext{T}, z::PointPrimalDual{T, Dense{T}}, mu::Float64) where T<:Number
    io = stdout
    println(io, "------ NesterovToddstep ------")
    
    # Step 1. Compute necessary quatities
    g = NTget_g(pb, z.x, z.s)
    w = NTget_w(pb, g)
    sinv = inv(z.s)
    muxs = z.x - product(sinv, mu)

    # Step 2. Solve M.dy = A(x - \mu inv(s))
    M = NTget_M(pb, g)
    rhs = evaluate(pb.A, muxs)

    # Choleski fact, solve
    dy = cholesky(M) \ rhs

    # Step 3. 
    ds = - evaluate(pb.A, dy)

    dx = - muxs - hadamard(w, ds, w)

    # Step 4.
    xstep = z.x+dx
    ystep = z.y+dy
    sstep = z.s+ds

    @printf("||A*(dy) + ds||                    = %.3e\n", norm(evaluate(pb.A, dy) + ds))
    @printf("||A(dx)||                          = %.3e\n", norm(evaluate(pb.A, dx)))
    @printf("||dx + w.ds.w - mu inv(s) + x||    = %.3e\n", norm(dx + hadamard(w, ds, w) - mu * inv(z.s) + z.x))

    # Check that z = (x, y, s) is strict feasable
    # Check that z+dz is strict feasable too.

    return PointPrimalDual(dx, dy, ds)
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