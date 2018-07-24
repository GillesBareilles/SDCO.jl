export test_case_1a

# Test case 1.a
function test_case_1a_point()
    dims = Vector([3])

    ptj = Vector([1, 1, 1])
    ptk = Vector([1, 2, 3])
    ptl = Vector([1, 3, 2])
    ptjkl = Vector([1.0, 1.0, 1.0])
    vec = Vector{Float64}([])

    # SDCOContext
    PointE(ptj, ptk, ptl, ptjkl, dims, vec)
end

function test_case_1a()
    nsdpvar = Vector([3])

    ai = Vector([1, 2, 2, 2, 3, 3, 3])
    aj = Vector([1, 1, 1, 1, 1, 1, 1])
    ak = Vector([1, 1, 2, 3, 1, 2, 3])
    al = Vector([1, 3, 2, 1, 2, 1, 3])
    aijkl = Vector{Float64}([1, 1, 1, 1, 1, 1, 1])

    veci = Vector{Int}([])
    vecj = Vector{Int}([])
    vecij = Vector{Float64}([])
    
    b = Vector{Float64}([1, 1, 1])

    ci = Vector([1, 1, 1])
    cj = Vector([1, 2, 3])
    ck = Vector([1, 2, 3])
    cijk = Vector{Float64}([1, 1, 1])
    
    cvecj = Vector{Int}()
    cvecval = Vector{Float64}()

    problem = SDCOContext(ai, aj, ak, al, aijkl,
                          veci, vecj, vecij,
                          b,
                          ci, cj, ck, cijk,
                          cvecj, cvecval)

    ptj = Vector([1, 1, 1])
    ptk = Vector([1, 2, 3])
    ptl = Vector([1, 2, 3])
    ptjkl = Vector([1.0, 1.0, 1.0])
    vec = Vector{Float64}([])

    x = PointE(ptj, ptk, ptl, ptjkl, nsdpvar, vec)
    s = x
    y = zeros(Float64, length(b))

    return problem, (x, y, s)
end

function main()
    problem, init_point = test_case_1a()

    println(problem)

    println(init_point)
end
