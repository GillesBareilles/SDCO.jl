export NTget_g, NTget_w, NTget_M
export NesterovToddstep, NTpredcorr, NTgetcentralpathpoint

get_nc(z::PointPrimalDual) = get_nc(z.x)
get_nc(x::PointE) = sum(x.dims) + length(x.vec)

mu(z::PointPrimalDual) = dot(z.x, z.s) / (get_nc(z))

function delta(pb::SDCOContext, z::PointPrimalDual)
    g = NTget_g(pb, z.x, z.s, check=true)
    mu_z = mu(z)

    return 0.5 * norm(hadamard(g, sqrt(mu_z)*inv(z.x) - 1/sqrt(mu_z) * z.s, g, transposefirst=true))
end

function phi_mu(z::PointPrimalDual, mu::Float64)
    for i in eachindex(z.x.mats)
        if (eigmin(z.x.mats[i]) <= 0) || (eigmin(z.s.mats[i]) <= 0)
            return Inf
        end
    end

    for i in eachindex(z.x.vec)
        if (z.x.vec[i] <= 0) || (z.s.vec[i] <= 0)
            return Inf
        end
    end

    return dot(z.x, z.s) + mu * psi(hadamard(z.x, z.s))
end

function psi(x::PointE)
    res = 0

    for mat in x.mats
        res += -logdet(mat)
    end


    for xi in x.vec
        res += -log(xi)
    end
    return res
end

function NTgetcentralpathpoint(pb::SDCOContext, z0::PointPrimalDual{T, Dense{T}}) where T<:Number
    theta = 1/3
    tau = 0.99
    omega = 1e-4
    
    z = z0
    
    printstyled("\n---delta(pb, z) = $(delta(pb, z))\n", color = :light_cyan)

    it = 0
    while (delta(pb, z) > theta) && (it < 20)
        printstyled("\n--------- Iteration i = $it\n", color = :light_cyan)
        printstyled("--- delta(pb, z) = $(delta(pb, z))\n", color = :light_cyan)

        mu_z = mu(z)
        dc = NesterovToddstep(pb, z, mu_z)

        println("-- ", delta(pb, z), " < ", sqrt( (2*tau^2) / (1+2*tau^2) ))
        if delta(pb, z) < sqrt( (2*tau^2) / (1+2*tau^2) )
            alpha = 1
        else
            i = 0
            alpha = 2^(-i)

            while phi_mu(z+alpha*dc, mu_z) > (phi_mu(z, mu_z) - 4*omega*mu_z*alpha*delta(pb, z)^2)

                i += 1

                (i > 100) && error("i = $i...")
            end
        end
        @show alpha

        z = z+alpha * dc

        it += 1

    end

    printstyled("\n--- delta(pb, z) = $(delta(pb, z))\n", color = :green)

    return z
end

function NTpredcorr(pb::SDCOContext, z::PointPrimalDual{T, Dense{T}}) where T<:Number
    z_curr = z
    epsilon = 1e-15
    maxit = 20
    i = 0

    nc = get_nc(z.x)
    K = ceil((1 + sqrt(1+ 13*nc/2)) * log(mu(z) / epsilon))
    @show K

    while (i < maxit) && (mu(z) > epsilon)
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




function NesterovToddstep!(pb::SDCOContext{T}, z::PointPrimalDual{T, Dense{T}}, mu::Float64, d::PointPrimalDual) where T<:Number
    # Step 1. Compute necessary quatities
    g = NTget_g(pb, z.x, z.s)
    w = NTget_w(pb, g)

    muxs = z.x - product(inv(z.s), mu)

    # Step 2. Solve M.dy = A(x - \mu inv(s))
    M = NTget_M(pb, g)
    rhs = evaluate(pb.A, muxs)

    # Choleski fact, solve
    d.y = cholesky(M) \ rhs

    # Step 3. 
    d.s = - evaluate(pb.A, d.y)

    d.x = - muxs - hadamard(w, d.s, w)

    nothing
end



function NTget_g(pb::SDCOContext, x::PointE{T, Dense{T}}, s::PointE{T, Dense{T}}; check=true) where T<:Number
    mats = Vector{Dense{T}}([zeros(T, n, n) for n in x.dims])

    ## Deal with SDP part
    for j in eachindex(x.mats)
        xj_fact = LinearAlgebra.cholesky(Symmetric(x.mats[j]), check=check)
        sj_fact = LinearAlgebra.cholesky(Symmetric(s.mats[j]), check=check)

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

    vectors = [PointE(g.dims, length(g.vec), T, Dense{T}) for i=1:m]    
    for i=1:m
        hadamard!(g, pb.A[i], g, vectors[i], transposefirst = true)
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