export testcase1a, testcase1a_point, testcase1b, testcase1c
export testcase1abis, testcase1ater
export testcase2

# Test case 1.a
function testcase1a_point()

    # ptxmat = [sparse([1, 2, 3], [1, 2, 3], [1., 1., 1.])]
    # ptxmat = [Matrix{Float64}(I, 3, 3)]
    x = PointE([Matrix{Float64}(I, 3, 3)], Float64[])
end

function testcase1a()
    cmat = [sparse([1, 2, 3], [1, 2, 3], [1., 1., 1.])]
    c = PointE(cmat, Vector{Float64}())

    a1mats = [sparse([1], [1], [1.], 3, 3)]
    a1 = PointE(a1mats, Vector{Float64}())

    a2mats = [sparse([2, 3], [2, 1], [1., 1.], 3, 3)]
    a2 = PointE(a2mats, Vector{Float64}())

    a3mats = [sparse([2, 3], [1, 3], [1., 1.], 3, 3)]
    a3 = PointE(a3mats, Vector{Float64}())

    A = [a1, a2, a3]

    b = Vector{Float64}([1, 1, 1])

    problem = SDCOContext(c, A, b)
    
    # ptxmat = [sparse([1, 2, 3], [1, 2, 3], [1., 1., 1.])]
    # x = PointE(ptxmat, Vector{Float64}())
    x = PointE([Matrix{Float64}(I, 3, 3)], Float64[])

    # ptsmat = [sparse([1, 2, 3], [1, 2, 3], [1., 1., 1.])]
    # s = PointE(ptsmat, Vector{Float64}())
    s = PointE([Matrix{Float64}(I, 3, 3)], Float64[])

    y = zeros(Float64, length(b))

    return problem, PointPrimalDual(x, y, s)
end

function testcase1abis()
    cmat = [sparse([1, 2, 3], [1, 2, 3], [1., 1., 1.])]
    c = PointE(cmat, Vector{Float64}())

    a1mats = [sparse([1], [1], [1.], 3, 3)]
    a1 = PointE(a1mats, Vector{Float64}())

    a2mats = [sparse([2, 3], [2, 1], [1., 1.], 3, 3)]
    a2 = PointE(a2mats, Vector{Float64}())

    a3mats = [sparse([2, 3], [1, 3], [1., 1.], 3, 3)]
    a3 = PointE(a3mats, Vector{Float64}())

    A = [a1, a2, a3]

    b = Vector{Float64}([1, 1, 1])

    problem = SDCOContext(c, A, b)
    
    x = PointE([Matrix{Float64}(I, 3, 3)], Float64[])

    y = ones(Float64, length(b)) * -1 * (1+sqrt(17)) / 4
    
    s = c - evaluate(A, y)

    return problem, PointPrimalDual(x, y, s)
end

function testcase1ater()
    cmat = [sparse([1, 2, 3], [1, 2, 3], [1., 1., 1.])]
    c = PointE(cmat, Vector{Float64}())

    a1mats = [sparse([1], [1], [1.], 3, 3)]
    a1 = PointE(a1mats, Vector{Float64}())

    a2mats = [sparse([2, 3], [2, 1], [1., 1.], 3, 3)]
    a2 = PointE(a2mats, Vector{Float64}())

    a3mats = [sparse([2, 3], [1, 3], [1., 1.], 3, 3)]
    a3 = PointE(a3mats, Vector{Float64}())

    A = [a1, a2, a3]

    b = Vector{Float64}([1, 1, 1])

    problem = SDCOContext(c, A, b)
    
    x = PointE([Matrix{Float64}(I, 3, 3)], Float64[])

    y = Float64[0, -1, -0.5]
    
    s = c - evaluate(A, y)

    return problem, PointPrimalDual(x, y, s)
end


function testcase1b()
    c = PointE(Dense{Float64}[], Float64[1., 1.])
    
    A = [PointE(Dense{Float64}[], Float64[1., 2.])]
    b = [1.]

    problem = SDCOContext(c, A, b)

    x = PointE(Dense{Float64}[], Float64[1/3, 1/3])
    y = [0.]
    s = PointE(Dense{Float64}[], Float64[1., 1.])
    return problem, PointPrimalDual(x, y, s)
end

function testcase1c()
    cmat = [sparse(1.0I, 3, 3), sparse(1.0I, 3, 3)]
    c = PointE(cmat, Float64[1., 1.])

    mat1 = sparse([1], [1], [1.], 3, 3)
    mat2 = sparse([2, 3], [2, 1], [1., 1.], 3, 3)
    mat3 = sparse([2, 3], [1, 3], [1., 1.], 3, 3)
    matnull = spzeros(3,3)

    a1 = PointE([mat1, matnull], Float64[0, 0])
    a2 = PointE([mat2, matnull], Float64[0, 0])
    a3 = PointE([mat3, matnull], Float64[0, 0])

    a4 = PointE([matnull, mat1], Float64[0, 0])
    a5 = PointE([matnull, mat2], Float64[0, 0])
    a6 = PointE([matnull, mat3], Float64[0, 0])

    a7 = PointE([matnull, matnull], Float64[1., 2.])
    A = [a1, a2, a3, a4, a5, a6, a7]

    b = Vector{Float64}([1, 1, 1, 1, 1, 1, 1])

    problem = SDCOContext(c, A, b)

    y0 = -(1+sqrt(17)) / 4
    y = y0 * ones(7)

    x = PointE([Matrix{Float64}(I, 3, 3), Matrix([1 0 0.25; 0 0.5 0; 0.25 0 1])], Float64[(5-sqrt(17))/2, (-3+sqrt(17))/4])

    s = c - evaluate(A, y)

    return problem, PointPrimalDual(x, y, s)
end

function testcase2(p, q, r)
    Bis = Vector{Matrix}(undef, p)

    for i=1:p
        Bis[i] = rand(r, q)
    end
    B0 = rand(r, q)

    cmat = spzeros(r+q, r+q)
    for i=1:r, j=1:q
        cmat[j+q, i] = B0[i, j]
    end
    c = PointE([cmat], Float64[])

    A = PointE{Float64, Sparse{Float64}}[]
    b = Float64[]
    
    push!(A, PointE([sparse(-1.0I, r+q, r+q)], Float64[]))
    push!(b, -1.)
    
    for (i, Bi) in enumerate(Bis)
        aimat = spzeros(r+q, r+q)
        for i=1:r, j=1:q
            aimat[j+q, i] = Bi[i, j]
        end
        push!(A, PointE([aimat], Float64[]))
        push!(b, 0.)
    end

    problem = SDCOContext(c, A, b)

    y = zeros(p+1)
    y[1] = norm(B0) + 1
    x = PointE([1/(q+r) * Matrix(1.0I, q+r, q+r)], Float64[])
    s = c - evaluate(A, y)
    
    z = PointPrimalDual(x, y, s)

    return problem, z
end