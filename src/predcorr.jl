export solve

function print_header(io::IO)
    println("it  primal obj    dual obj           mu        delta        alpha   it. time")
end

function print_it(io::IO, it, primobj, dualobj, mu, delta, alpha, time)
    @printf("%2i  % .3e  % .3e   % .3e    % .6f   % .3e   %f\n", it, primobj, dualobj, mu, delta, alpha, time)
end

function solve(pb::SDCOContext{T}, z0) where T<:Number
    z = deepcopy(z0)

    outlev = pb.options[:opt_outlev]

    ## Centering section
    θ = 1/3
    τ = 0.99
    ω = 1e-4

    cur_mu = mu(z)
    cur_delta = delta(pb, z)
    it = 0
    α = -1
    d = NesterovToddstep(pb, z, 0.)

    (outlev > 0) && print_header(stdout)
    (outlev > 0) && print_it(stdout, it, -1, -1, cur_mu, cur_delta, -1, α)

    if (cur_delta > θ)
        (outlev > 0) && println("Getting to central path...")
    end

    while (cur_delta > θ) && (it < 20)
        t1 = time()
        α = NTcentering_it!(pb, τ, ω, z, cur_mu, cur_delta)

        cur_mu = mu(z)
        cur_delta = delta(pb, z)

        it += 1
        (outlev > 0) && print_it(stdout, it, -1, -1, cur_mu, cur_delta, α, time() - t1)
    end


    ## Path following section
    (outlev > 0) && println("Following central path...")
    ε = pb.options[:opt_ε]
    maxit = pb.options[:opt_maxit]
    alphacorr = 0.999

    nc = get_nc(z.x)
    # K = ceil((1 + sqrt(1+ 13*nc/2)) * log10(mu(z) / epsilon))
    # @show K

    cur_μ = mu(z)

    # K = log(ε / cur_μ) / log(1 + 4 / (13*nc) * (1 - sqrt(1+13*nc/2)))
    # @show K

    while (it < maxit) && (cur_μ > ε)
        t1 = time()

        res = α, cur_μ = NTpredcorr_it!(pb, alphacorr, cur_μ, cur_delta, z, d)

        it+=1

        cur_μ = mu(z)
        cur_delta = delta(pb, z)
        (outlev > 0) && print_it(stdout, it, get_primobj(pb, z), get_dualobj(pb, z), cur_μ, cur_delta, α, time() - t1)
    end

    (outlev > 0) && println("Final duality gap: ", get_primobj(pb, z) - get_dualobj(pb, z))
    (outlev > 0) && println("Final mu         : ", mu(z))

    return z
end



function NTcentering_it!(pb, τ, ω, z, cur_mu, cur_delta)
    mu_z = cur_mu
    dc = NesterovToddstep(pb, z, mu_z)

    if cur_delta < sqrt( (2*τ^2) / (1+2*τ^2) )
        α = 1
    else
        i = 0
        α = 2^(-i)

        while phi_mu(z + α*dc, mu_z) > (phi_mu(z, mu_z) - 4*ω*mu_z*α*delta(pb, z)^2)
            i += 1
            α = (1/2)^(i)

            (i > 100) && error("i = $i...")
        end
    end

    add!(z, α * dc)

    return α
end

# function symmetrize!(z)
#     for i in 1:length(z.x.mats)
#         z.x.mats[i] = (z.x.mats[i] + transpose(z.x.mats[i]))/2
#     end
#
#     for i in 1:length(z.s.mats)
#         z.s.mats[i] = (z.s.mats[i] + transpose(z.s.mats[i]))/2
#     end
# end

function NTpredcorr_it!(pb, alphacorr, cur_mu, cur_delta, z, d)
    # centering step
    mu_z = mu(z)
    NesterovToddstep!(pb, z, mu_z, d)

    # zc = z + dc
    add!(z, d)
    mu_zc = mu(z)

    # affine step
    NesterovToddstep!(pb, z, 0., d)

    α = NT_getalpha(pb, z, d)

    (pb.options[:opt_outlev] > 1) && printstyled("mu(z) - mu(zc) : ", abs(mu_z - mu_zc), "\n", color=:red)

    alphacorr = min(α, 0.99999)

    add!(z, alphacorr*d)
    mu_za = mu(z)

    (pb.options[:opt_outlev] > 1) && printstyled("mu(z+) / mu(zc) - (1 - alpha): ", abs(mu_za / mu_zc - (1-α)), "\n", color=:red)

    return α, mu_za
end
