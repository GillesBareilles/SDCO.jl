using Test, SDCO


@testset "SDCO instance, feasibility, objectives" begin

    pb, (x, y, s) = test_case_1a();

    @test evaluate(pb.A, x) == [1., 1., 1.]
    # @test evaluate(pb.A, y) == PointE(x.dims, length(x.vec), Float64)


    @test get_primobj(pb, x) == 3.
    @test get_dualobj(pb, y) == 0.

    @test get_primslacks(pb, x) == [0., 0., 0.]
    @test norm(get_dualslacks(pb, y, s)) == 0

    @test get_primfeaserr(pb, x) == 0.
    @test get_dualfeaserr(pb, y, s) == 0.
end

@testset "Nesterov Todd step" begin

    @testset "test_case_a1" begin
        pb, (x, y, s) = test_case_1a()
    
        @testset "mu = $mu" for mu in [0.2 0.4 0.6]
            dx, dy, ds = NesterovToddstep(pb, x, y, s, mu)
        
            @test norm(dx) == 0
            @test dy == (1-mu) .* [1., 1., 1.]
            @test norm(ds - PointE([ (mu-1)*[1. 1. 1. ; 1. 1. 0.; 1. 0. 1.] ], Float64[])) == 0
        end
    end

    @testset "test_case_1b" begin
        pb, (x, y, s) = test_case_1b()
    
        @testset "mu = $mu" for mu in [1. 0.2 0.4 0.6]
            dx, dy, ds = NesterovToddstep(pb, x, y, s, mu)
        
            @test norm(dy - [0.6 * (1-3*mu)]) < 1e-15
            
            @test norm(ds - PointE(SDCO.Dense{Float64}[], Float64[ 0.6*(3*mu-1), 1.2*(3*mu-1) ])) < 1e-15
            
            @test norm(dx - PointE(SDCO.Dense{Float64}[], Float64[ 0.4*mu - 2/15, -0.2*mu + 1/15 ])) < 1e-15
        end
    end

end

