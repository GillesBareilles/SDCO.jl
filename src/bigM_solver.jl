function solve(bigMpb::SolverBigM{T}) where T
    pb = bigMpb.model
    e = ones(pb.c)

    ## Compute initial primal dual starting point, and big M initial values
    x0 = get_symmetricprimalpt(pb.A, pb.b)
    z = PointPrimalDual(x0.dims, length(x0.vec), bigMpb.model.m, T, Dense{T})

    printstyled("Begining solve...\n", color=:yellow)


    ## start solve
    it = 0
    keepon = true
    while it < 5 && keepon
        printstyled("--- iteration $it\n", color=:yellow)

        ## Setting problem with new values of M1, M2
        @assert norm(evaluate(pb.A, x0) - pb.b) < 1e-10

        z̄, M1, M2 = derivestartingpoint(bigMpb, x0)
        printstyled("M1, M2 = $M1, $M2\n", color=:yellow)

        update_parameters!(bigMpb, z̄, M1, M2)

        @show min(bigMpb.model_bigM, z̄.x)
        @show min(bigMpb.model_bigM, z̄.s)

        @assert get_primfeaserr(bigMpb.model_bigM, z̄) < 1e-10
        @assert get_dualfeaserr(bigMpb.model_bigM, z̄) < 1e-10

        ## Solving
        z̄opt = solve(bigMpb.model_bigM, z̄)

        ξ1 = z̄opt.x.vec[end-1]
        ξ2 = z̄opt.x.vec[end]
        η = z̄opt.y[end]
        σ1 = z̄opt.s.vec[end-1]
        σ2 = z̄opt.s.vec[end]

        println("ξ1 = ", ξ1)
        println("ξ2 = ", ξ2)
        println("η = ", η)
        println("σ1 = ", σ1)
        println("σ2 = ", σ2)

        extractinitpbsol!(z, z̄opt)
        printstyled("prim feaserr: ", get_primfeaserr(bigMpb.model, z), "\n", color=:yellow)
        printstyled("dual feaserr: ", get_primfeaserr(bigMpb.model, z), "\n", color=:yellow)
        printstyled("prim obj: ", get_primobj(bigMpb.model, z), "\n", color=:yellow)
        printstyled("dual obj: ", get_dualobj(bigMpb.model, z), "\n", color=:yellow)
        printstyled("min_K(x): ", min(bigMpb.model_bigM, z̄.x), "\n", color=:yellow)
        printstyled("min_K(s): ", min(bigMpb.model_bigM, z̄.s), "\n", color=:yellow)

        if ξ2 < 1e-10 && η < 1e-10
            keepon = false
            printstyled("Optimal solution found.\n", color=:yellow)
            return z

        elseif bigMpb.M1 > 1e5
            keepon = false
            printstyled("Dual problem is likely to be infeasible.\n", color=:yellow)
            return z

        elseif bigMpb.M2 > 1e5
            keepon = false
            printstyled("Primal problem is likely to be infeasible.\n", color=:yellow)
            return z
        end

        x0 = z.x
        # if η > 0
        #     M1 *= 2
        # end
        #
        # if ξ2 > 0
        #     M2 *= 2
        # end

        it += 1
    end

    return false
end
