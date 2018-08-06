using Test, SDCO


@testset "SDCO instance, feasibility, objectives" begin

    pb, z = test_case_1a();

    @test evaluate(pb.A, z.x) == [1., 1., 1.]
    # @test evaluate(pb.A, y) == PointE(x.dims, length(x.vec), Float64)


    @test get_primobj(pb, z) == 3.
    @test get_dualobj(pb, z) == 0.

    @test get_primslacks(pb, z) == [0., 0., 0.]
    @test norm(get_dualslacks(pb, z.y, z.s)) == 0

    @test get_primfeaserr(pb, z) == 0.
    @test get_dualfeaserr(pb, z) == 0.
end

@testset "Nesterov Todd step" begin

    @testset "test_case_a1" begin
        pb, z = test_case_1a()
    
        @testset "mu = $mu" for mu in [0.2 0.4 0.6]
            dz = NesterovToddstep(pb, z, mu)
        
            @test norm(dz.x - PointE([ (mu-1)*[0. -1/3 -1/3 ; -1/3 2/3 0.; -1/3 0. 2/3] ], Float64[])) < 1e-15
            @test norm(dz.y - (1-mu) .* [1., 1/3, 1/3]) < 1e-15
            @test norm(dz.s - PointE([ (mu-1)*[1. 1/3 1/3 ; 1/3 1/3 0.; 1/3 0. 1/3] ], Float64[])) < 1e-15
        end
    end

    @testset "test_case_1b" begin
        pb, z = test_case_1b()
    
        @testset "mu = $mu" for mu in [1. 0.2 0.4 0.6]
            dz = NesterovToddstep(pb, z, mu)
        
            @test norm(dz.y - [0.6 * (1-3*mu)]) < 1e-15
            
            @test norm(dz.s - PointE(SDCO.Dense{Float64}[], Float64[ 0.6*(3*mu-1), 1.2*(3*mu-1) ])) < 1e-15
            
            @test norm(dz.x - PointE(SDCO.Dense{Float64}[], Float64[ 0.4*mu - 2/15, -0.2*mu + 1/15 ])) < 1e-15
        end
    end

end

