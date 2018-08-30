using Test, SDCO


@testset "SDCO instance, feasibility, objectives" begin

    @testset "testcase1a sparse - symmetric = $issym" for issym in [true, false]
        pb, z = testcase1a(symmetric = issym);

        @test evaluate(pb.A, z.x) == [1., 1., 1.]

        @test get_primobj(pb, z) == 3.
        @test get_dualobj(pb, z) == 0.

        @test get_primslacks(pb, z) == [0., 0., 0.]
        @test norm(get_dualslacks(pb, z.y, z.s)) == 0

        @test get_primfeaserr(pb, z) == 0.
        @test get_dualfeaserr(pb, z) == 0.
    end
end

@testset "Nesterov Todd step" begin

    @testset "testcase1a sparse - symmetric = $issym" for issym in [true, false]
        pb, z = testcase1a(symmetric = issym)

        @testset "mu = $mu" for mu in [0.2 0.4 0.6]
            dz = NesterovToddstep(pb, z, mu)

            @test norm(dz.x - PointE([ (mu-1)*[0. -1/3 -1/3 ; -1/3 2/3 0.; -1/3 0. 2/3] ], Float64[])) < 1e-15
            @test norm(dz.y - (1-mu) .* [1., 1/3, 1/3]) < 1e-15
            @test norm(dz.s - PointE([ (mu-1)*[1. 1/3 1/3 ; 1/3 1/3 0.; 1/3 0. 1/3] ], Float64[])) < 1e-15
        end
    end

    @testset "testcase1b" begin
        pb, z = testcase1b()

        @testset "mu = $mu" for mu in [1. 0.2 0.4 0.6]
            dz = NesterovToddstep(pb, z, mu)

            @test norm(dz.y - [0.6 * (1-3*mu)]) < 1e-15

            @test norm(dz.s - PointE(SDCO.Dense{Float64}[], Float64[ 0.6*(3*mu-1), 1.2*(3*mu-1) ])) < 1e-15

            @test norm(dz.x - PointE(SDCO.Dense{Float64}[], Float64[ 0.4*mu - 2/15, -0.2*mu + 1/15 ])) < 1e-15
        end
    end

end


@testset "Central path following" begin

    @testset "testcase1a sparse - symmetric = $issym" for issym in [true, false]
        pb, z = testcase1a(symmetric = issym)

        @assert delta(pb, z) < 1/3

        zfinal = NTpredcorr(pb, z; outlev=0)

        epsilon = 1e-10
        @test mu(zfinal) < 1e-10

        @test get_primfeaserr(pb, zfinal) < epsilon
        @test get_dualfeaserr(pb, zfinal) < epsilon
        @test dot(zfinal.x, zfinal.s) < epsilon
    end

    @testset "testcase1b" begin
        pb, z = testcase1b()

        @assert delta(pb, z) < 1/3

        zfinal = NTpredcorr(pb, z; outlev=0)

        epsilon = 1e-10
        @test mu(zfinal) < 1e-10

        @test get_primfeaserr(pb, zfinal) < epsilon
        @test get_dualfeaserr(pb, zfinal) < epsilon
        @test dot(zfinal.x, zfinal.s) < epsilon
    end
end

@testset "Getting central point" begin
    @testset "testcase1c sparse - symmetric = $issym" for issym in [true, false]
        pb, z = testcase1c(symmetric = issym)

        @assert delta(pb, z) > 1/3

        zcentralpath = NTgetcentralpathpoint(pb, z; outlev=0)

        @test delta(pb, zcentralpath) < 1/3
    end
end

@testset "Minimum Matrix Norm, feasible starting point" begin
    @testset "mmn-$p-$q-$r" for (p, q, r) in [ (2, 2, 2), (4, 2, 2) ]
        ε = 1e-10

        pb, z = testcase2(p, q, r)
        pb.options[:opt_ε] = ε
        pb.options[:opt_outlev] = 0

        zfinal = solve(pb, z)

        @test mu(zfinal) < ε

        # @test get_primfeaserr(pb, zfinal) < epsilon
        # @test get_dualfeaserr(pb, zfinal) < epsilon
        # @test dot(zfinal.x, zfinal.s) < epsilon
    end
end
