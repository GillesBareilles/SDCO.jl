export AbstractSover, SolverBigM, SolverSelfDual
export pointE_2_linearidx, linearidx_2_pointE, linearize, get_symmetricprimalpt
export solve_bigM

abstract type AbstractSover end


mutable struct SolverBigM{T, matT} <: AbstractSover
    model::SDCOContext{T, matT}
    model_bigM::SDCOContext{T, matT}

    ## Big M method parameters
    M1::T
    M2::T
end

function SolverBigM(pb::SDCOContext{T}) where T
    e = ones(pb.c)

    ## Declare internal bigM model
    c̄ = PointE(pb.c.mats, [pb.c.vec..., T(0), T(0)])

    Ā = Vector{eltype(pb.A)}(undef, length(pb.A)+1)
    for (i, ai) in enumerate(pb.A)
        Ā[i] = PointE(ai.mats, [ai.vec..., 0, -dot(ai, e)])
    end

    Ā[end] = PointE(-e.mats, [-e.vec..., -T(1), T(pb.nc)])

    b̄ = [pb.b..., 0]

    return SolverBigM(pb, SDCOContext(c̄, Ā, b̄, options=pb.options), NaN, NaN)
end

"""
    update_parameters!(Mctx, M1, M2)

update the bigM model with the new M1, M2 parameters.
"""
function update_parameters!(Mctx::SolverBigM, M1, M2)
    Mctx.M1 = M1
    Mctx.M2 = M2

    e = ones(Mctx.model.c)

    Mctx.model_bigM.b[end] = -M1
    Mctx.model_bigM.c.vec[end] = M2 - dot(Mctx.model.c, e)
end

"""
    z̄, M1, M2 = derivestartingpoint(Mctx::SolverBigM, x0::PointE)

Derive the starting point, and adequate M parameters for building an extended
big M problem, starting from a point satisfying the primal linear constraint A(x)=b.
"""
function derivestartingpoint(Mctx::SolverBigM{T, matspT}, x0::PointE{T, matT}) where {T, matT, matspT}
    pb = Mctx.model

    e = ones(x0)

    ξ2 = max(0, -min(pb, x0)) + 1
    M1 = max(0, dot(e, x0)) + 1
    ξ1 = M1 - dot(e, x0)
    add!(x0, ξ2 * e)

    y0 = zeros(T, pb.m)
    η = max(0, -min(pb, pb.c - evaluate(pb.A, y0))) + 1

    s0 = pb.c - evaluate(pb.A, y0) + η * e
    M2 = max(0, dot(s0, e)) + 1
    σ1 = η
    σ2 = M2 - dot(s0, e)

    ## complete starting points for extended problem
    x̄0 = PointE(x0.mats, [x0.vec..., ξ1, ξ2])
    ȳ0 = [y0..., η]
    s̄0 = PointE(s0.mats, [s0.vec..., σ1, σ2])
    z̄ = PointPrimalDual(x̄0, ȳ0, s̄0)

    return z̄, M1, M2
end


"""
"""
function extractinitpbsol!(z::PointPrimalDual, z̄::PointPrimalDual)
    extractinitpbsol!(z.x, z̄.x)
    z.y = z̄.y[1:end-1]
    extractinitpbsol!(z.s, z̄.s)
    nothing
end

function extractinitpbsol!(x::PointE{T, matT}, x̄::PointE{T, matT}) where {T, matT}
    for (i, mat) in enumerate(x̄.mats)
        x.mats[i] = mat
    end
    x.vec = x̄.vec[1:end-2]
    nothing
end


"""
    get_symmetricprimalpt(A)

Compute a symmetric point satisfying the primal constraint A(x) - b = 0.
"""
function get_symmetricprimalpt(A::Vector{PointE{T, matT}}, b) where {T, matT}

    ## Write down linear system matrix
    M = linearize(A)

    ## Solve
    vecsol = M \ b

    ## Build feasible point
    x0 = PointE(first(A).dims, length(first(A).vec), T, Dense{T})
    for (linind, val) in  enumerate(vecsol)
        s, i, j = linearidx_2_pointE(x0, linind)

        if s == -1 && j == -1
            x0.vec[i] = val
        else
            x0.mats[s][i, j] = val
        end
    end

    ## Symmetrize point
    for (i, mat) in enumerate(x0.mats)
        x0.mats[i] = (mat + transpose(mat)) / 2
    end

    return x0
end

function pointE_2_linearidx(ai, s, i, j)
    dims, veclen = ai.dims, length(ai.vec)
    return sum(dims[1:(s-1)].^2) + (i-1)*dims[s] + j
end

function pointE_2_linearidx(ai, i)
    dims, veclen = ai.dims, length(ai.vec)
    return sum(dims.^2) + i
end

function linearidx_2_pointE(ai, i)
    dims, veclen = ai.dims, length(ai.vec)
    if i > sum(dims.^2)
        return -1, i - sum(dims.^2), -1
    else
        s = 1
        while i > sum(dims[1:s].^2)
            s+=1
        end
        matlinind = i - sum(dims[1:s-1].^2) -1 #-1 for 0 based indexing
        i = matlinind ÷ dims[s]
        j = matlinind % dims[s]
        return s, i+1, j+1
    end
end

function linearize(A)
    n = sum(A[1].dims.^2) + length(A[1].vec)

    Is = []
    Js = []
    Vs = Float64[]
    for (ctrind, ai) in enumerate(A)
        ## Matrices
        for (s, mat) in enumerate(ai.mats)
            for i=1:size(mat, 1), j=1:size(mat, 2)
                if mat[i, j] != 0
                    push!(Is, ctrind)
                    push!(Js, pointE_2_linearidx(ai, s, i, j))
                    push!(Vs, mat[i, j])
                end
            end
        end

        ## Linear part
        for (vecind, coeff) in enumerate(ai.vec)
            push!(Is, ctrind)
            push!(Js, pointE_2_linearidx(ai, vecind))
            push!(Vs, coeff)
        end
    end

    return sparse(Is, Js, Vs, length(A), n)
end
