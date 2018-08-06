export NesterovToddstep, NTget_g, NTget_w, NTget_M
export NTpredcorr

get_nc(z::PointPrimalDual) = get_nc(z.x)
get_nc(x::  PointE) = sum(x.dims) + length(x.vec)

mu(z::PointPrimalDual) = dot(z.x, z.s) / (get_nc(z))

function delta(pb::SDCOContext, z::PointPrimalDual)
    g = NTget_g(pb, z.x, z.s)
    mu_z = mu(z)

    return 0.5 * norm(hadamard(g, sqrt(mu_z)*inv(z.x) - 1/sqrt(mu_z) * z.s, g, transposefirst=true))
end


function NTpredcorr(pb::SDCOContext, z::PointPrimalDual{T, Dense{T}}) where T<:Number

    z_curr = z
    epsilon = 1e-15
    i = 0

    nc = get_nc(z.x)
    K = ceil((1 + sqrt(1+ 13*nc/2)) * log(mu(z) / epsilon))
    @show K

    while (i < 20) && (mu(z) > epsilon)
        printstyled("\n--------- Iteration i = $i\n", color = :light_cyan)
        printstyled("Starting point - central point:             ", mu(z), "\n", color=:light_cyan)
        printstyled("Starting point - distance to central path:  ", delta(pb, z), "\n", color=:light_cyan)
        # print_pointsummary(pb, z)

        # centering step
        mu_z = mu(z)
        dc = NesterovToddstep(pb, z, mu_z)

        zc = z + dc

        printstyled("\nCentered step - central point:              ", mu(zc), "\n", color=:light_cyan)
        printstyled("Centered step - distance to central path:   ", delta(pb, zc), "\n", color=:light_cyan)

        # affine step
        da = NesterovToddstep(pb, zc, 0.)

        alpha = NT_getalpha(pb, zc, da)

        println("alpha = ", alpha)
        printstyled("mu(z) - mu(zc) : ", abs(mu(z) - mu(zc)), "\n", color=:red)
        
        
        z = zc + alpha*da
        printstyled("mu(z+) / mu(zc) - (1 - alpha): ", abs(mu(z) / mu(zc) - (1-alpha)), "\n", color=:red)

        i+=1
    end

    @show mu(z_curr), K
    printstyled("\n\nDone, mu = $(mu(z))\n\n", color=:green)
    print_pointsummary(pb, z)

    return z
end

function NesterovToddstep(pb::SDCOContext{T}, z::PointPrimalDual{T, Dense{T}}, mu::Float64) where T<:Number

    io = stdout
    printAll = 0

    (printAll >= 1) && println(io, "------ NesterovToddstep ------")
    
    # Step 1. Compute necessary quatities
    g = NTget_g(pb, z.x, z.s)
    w = NTget_w(pb, g)
    sinv = inv(z.s)
    muxs = z.x - product(sinv, mu)

    # if 1 == 0
    #     sqrtx = sqrt(z.x)
    #     t = hadamard(sqrtx, z.s, sqrtx)
    #     t = sqrt(inv(t))
    #     printstyled("|| w - its def || = ", norm(w - hadamard(sqrtx, t, sqrtx)), "\n", color=:light_red)

    #     display(hadamard(sqrtx, t, sqrtx))
    #     display(w)
    #     w = hadamard(sqrtx, t, sqrtx)
    # end


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

    if printAll >= 1
        @printf("||A*(dy) + ds||                    = %.3e\n", norm(evaluate(pb.A, dy) + ds))
        @printf("||A(dx)||                          = %.3e\n", norm(evaluate(pb.A, dx)))
        @printf("||dx + w.ds.w - mu inv(s) + x||    = %.3e\n", norm(dx + hadamard(w, ds, w) - mu * inv(z.s) + z.x))
        @printf("System actually solved:\n")
        @printf("||A(w . A*(dy) . w) - A(x-muinv(s))|| = %.3e\n", norm(evaluate(pb.A, hadamard(w, evaluate(pb.A, dy), w))
                                                                    - evaluate(pb.A, z.x - mu * inv(z.s))))
        @printf("||A*(dy) + ds||                       = %.3e\n", norm(evaluate(pb.A, dy) + ds))
        @printf("||dx + w.ds.w - mu inv(s) + x||       = %.3e\n", norm(dx + hadamard(w, ds, w) - mu * inv(z.s) + z.x))
    end

    # Check that z = (x, y, s) is strict feasable
    # Check that z+dz is strict feasable too.

    return PointPrimalDual(dx, dy, ds)
end



function NTget_g(pb::SDCOContext, x::PointE{T, Dense{T}}, s::PointE{T, Dense{T}}) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])

    ## Deal with SDP part
    for j in eachindex(x.mats)
        xj_fact = LinearAlgebra.cholesky(Symmetric(x.mats[j]))
        sj_fact = LinearAlgebra.cholesky(Symmetric(s.mats[j]))

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

function NT_getalpha(pb, zc, da)
    g = NTget_g(pb, zc.x, zc.s)
    gtr = transpose(g)
    dasdax = hadamard(da.s, da.x)
    Enorm = norm(hadamard(inv(g), dasdax, g) + hadamard(gtr, dasdax, inv(gtr)))
    mu_zc = mu(zc)

    return 2 / (1 + sqrt(1+13*Enorm/(2*mu_zc)))
end