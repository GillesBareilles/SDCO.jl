export solve

function print_header(io::IO)
    println("it  primal obj    dual obj           mu        delta        alpha   it. time")
end

function print_it(io::IO, it, primobj, dualobj, mu, delta, alpha, time)
    @printf("%2i  % .3e  % .3e   % .3e    % .6f   % .3e   %f\n", it, primobj, dualobj, mu, delta, alpha, time)
end

function solve(pb::SDCOContext{T}, z0) where T<:Number
    z = deepcopy(z0)

    ## Centering section
    theta = 1/3
    tau = 0.99
    omega = 1e-4

    cur_mu = mu(z)
    cur_delta = delta(pb, z)
    it = 0
    alpha = -1
    d = NesterovToddstep(pb, z, 0.)

    print_header(stdout)
    print_it(stdout, it, -1, -1, cur_mu, cur_delta, -1, alpha)

    if (cur_delta > theta)
        println("Getting to central path...")
    end

    while (cur_delta > theta) && (it < 20)
        t1 = time()
        alpha = NTcentering_it!(pb, tau, omega, z, cur_mu, cur_delta)

        cur_mu = mu(z)
        cur_delta = delta(pb, z)

        it += 1
        print_it(stdout, it, -1, -1, cur_mu, cur_delta, alpha, time() - t1)
    end


    ## Path following section
    println("Following central path...")
    epsilon = 5e-10
    maxit = 30
    alphacorr = 0.999

    nc = get_nc(z.x)
    # K = ceil((1 + sqrt(1+ 13*nc/2)) * log10(mu(z) / epsilon))
    # @show K

    ε = 5e-10
    μ = mu(z)

    K = log(ε / μ) / log(1 + 4 / (13*nc) * (1 - sqrt(1+13*nc/2)))
    @show K

    while (it < maxit) && (mu(z) > epsilon)
        t1 = time()

        res = alpha, cur_mu = NTpredcorr_it!(pb, alphacorr, cur_mu, cur_delta, z, d)

        it+=1

        cur_mu = mu(z)
        cur_delta = delta(pb, z)
        print_it(stdout, it, get_primobj(pb, z), get_dualobj(pb, z), cur_mu, cur_delta, alpha, time() - t1)
    end

    @show get_primobj(pb, z) - get_dualobj(pb, z)
    @show mu(z)

    return z
end



function NTcentering_it!(pb, tau, omega, z, cur_mu, cur_delta)
    # mu_z = mu(z)
    mu_z = cur_mu
    dc = NesterovToddstep(pb, z, mu_z)

    if cur_delta < sqrt( (2*tau^2) / (1+2*tau^2) )
        alpha = 1
    else
        i = 0
        alpha = 2^(-i)

        while phi_mu(z+alpha*dc, mu_z) > (phi_mu(z, mu_z) - 4*omega*mu_z*alpha*delta(pb, z)^2)
            i += 1
            alpha = (1/2)^(i)

            (i > 100) && error("i = $i...")
        end
    end

    add!(z, alpha * dc)

    return alpha
end

function symmetrize!(z)
    for i in 1:length(z.x.mats)
        z.x.mats[i] = (z.x.mats[i] + transpose(z.x.mats[i]))/2
    end

    for i in 1:length(z.s.mats)
        z.s.mats[i] = (z.s.mats[i] + transpose(z.s.mats[i]))/2
    end
end

function NTpredcorr_it!(pb, alphacorr, cur_mu, cur_delta, z, d)
    # centering step
    mu_z = mu(z)
    NesterovToddstep!(pb, z, mu_z, d)

    # zc = z + dc
    add!(z, d)
    mu_zc = mu(z)

    # affine step
    NesterovToddstep!(pb, z, 0., d)

    alpha = NT_getalpha(pb, z, d)

    printstyled("mu(z) - mu(zc) : ", abs(mu_z - mu_zc), "\n", color=:red)

    alphacorr = min(alpha, 0.99999)

    add!(z, alphacorr*d)
    mu_za = mu(z)

    printstyled("mu(z+) / mu(zc) - (1 - alpha): ", abs(mu_za / mu_zc - (1-alpha)), "\n", color=:red)

    return alpha, mu_za
end
