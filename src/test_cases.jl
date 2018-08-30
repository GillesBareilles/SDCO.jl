export testcase1a, testcase1a_point, testcase1b, testcase1c
export testcase1abis, testcase1ater
export testcase2

# Test case 1.a
function testcase1a_point()
    x = PointE([Matrix{Float64}(I, 3, 3)], Float64[])
end

function testcase1a(; symmetric=false)
    if symmetric
        cmat = [Symmetric(sparse([1, 2, 3], [1, 2, 3], [1., 1., 1.]))]
        a1mats = [Symmetric(sparse([1], [1], [1.], 3, 3), :L)]
        a2mats = [Symmetric(sparse([2, 3], [2, 1], [1., 1.], 3, 3), :L)]
        a3mats = [Symmetric(sparse([2, 3], [1, 3], [1., 1.], 3, 3), :L)]
    else
        cmat = [sparse([1, 2, 3], [1, 2, 3], [1., 1., 1.])]
        a1mats = [sparse([1], [1], [1.], 3, 3)]
        a2mats = [sparse([1, 2, 3], [3, 2, 1], [1., 1., 1.], 3, 3)]
        a3mats = [sparse([1, 2, 3], [2, 1, 3], [1., 1., 1.], 3, 3)]
    end


    c = PointE(cmat, Vector{Float64}())
    a1 = PointE(a1mats, Vector{Float64}())
    a2 = PointE(a2mats, Vector{Float64}())
    a3 = PointE(a3mats, Vector{Float64}())

    A = [a1, a2, a3]

    b = Vector{Float64}([1, 1, 1])

    problem = SDCOContext(c, A, b)

    x = PointE([Matrix{Float64}(I, 3, 3)], Float64[])
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

function testcase1c(; symmetric=false)
    if symmetric
        cmat = [Symmetric(sparse(1.0I, 3, 3), :L), Symmetric(sparse(1.0I, 3, 3), :L)]

        mat1 = Symmetric(sparse([1], [1], [1.], 3, 3), :L)
        mat2 = Symmetric(sparse([2, 3], [2, 1], [1., 1.], 3, 3), :L)
        mat3 = Symmetric(sparse([2, 3], [1, 3], [1., 1.], 3, 3), :L)
        matnull = Symmetric(spzeros(3,3), :L)
    else
        cmat = [sparse(1.0I, 3, 3), sparse(1.0I, 3, 3)]

        mat1 = sparse([1], [1], [1.], 3, 3)
        mat2 = sparse([1, 2, 3], [3, 2, 1], [1., 1., 1.], 3, 3)
        mat3 = sparse([1, 2, 3], [2, 1, 3], [1., 1., 1.], 3, 3)
        matnull = spzeros(3,3)
    end

    c = PointE(cmat, Float64[1., 1.])

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

function testcase2(p, q, r; storage=:sparsesym)

    ## Objective
    B0 = rand(q, r)

    cmat::Sparse{Float64} = spzeros(q+r, q+r)
    if storage == :sparsesym
        for i=1:r, j=1:q
            cmat[q+i, j] = B0[j, i]
        end
        c = PointE([Symmetric(cmat, :L)], Float64[])
    elseif storage == :sparsefull
        for i=1:r, j=1:q
            cmat[q+i, j] = B0[j, i]
            cmat[j, i+q] = B0[j, i]
        end
        c = PointE([cmat], Float64[])
    else
        error("Unknown parameter $storage, choose from :sparsesym, :sparsefull")
    end

    if storage == :sparsesym
        A = PointE{Float64, SparseSym{Float64}}[]
    elseif storage == :sparsefull
        A = PointE{Float64, Sparse{Float64}}[]
    end
    b = Float64[]

    if storage == :sparsesym
        push!(A, PointE([Symmetric(sparse(-1.0I, q+r, q+r), :L)], Float64[]))
    elseif storage == :sparsefull
        push!(A, PointE([sparse(-1.0I, q+r, q+r)], Float64[]))
    end
    push!(b, -1.)

    for ctrind=1:p
        aimat = spzeros(q+r, q+r)
        Bi = rand(q, r)
        if storage == :sparsesym
            for i=1:r, j=1:q
                aimat[q+i, j] = Bi[j, i]
            end
            push!(A, PointE([Symmetric(aimat, :L)], Float64[]))
        elseif storage == :sparsefull
            for i=1:r, j=1:q
                aimat[q+i, j] = Bi[j, i]
                aimat[j, q+i] = Bi[j, i]
            end
            push!(A, PointE([aimat], Float64[]))
        end
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
