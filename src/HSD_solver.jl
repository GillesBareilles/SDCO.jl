export solvesystem

"""
    (z, τ, κ) = solvesystem(pb::SDCOContext, z::PointPrimalDual{T matT}, τ::T, κ::T) where {T, matT}

    Solves the following system for (dx, dy, ds, dτ, dκ), where A, b, c are held in the pb::SDCOContext structure:

             A(dx)   -b dτ          =  η rP    =  η (bτ - A(x))
    - A*(dy)         +c dτ  -ds     = -η rD    = -η (-A*(y) - s + c τ)
    b⋅dy     -c⋅dx              -dκ =  η rG    =  η (-b⋅y + c⋅x + κ)

    dx + w•ds•w = γμinv(s) - x
    τ dκ + κ dτ = γμ  - τκ

    where μ = (x⋅s + τκ) / (nc + 1)

    **Note** : the first complementarity equation is the symmetrized and pseudo-linearized version of :
        x•ds + s•dx = γμe - x•s
"""
function solvesystem(pb::SDCOContext, x::PointE{T, matT}, y::AbstractArray{T}, s::PointE{T, matT}, τ::T, κ::T, η::T, γ::T) where {T<:Number, matT<:AbstractArray}
    A = pb.A
    c = pb.c
    b = pb.b

    tol = 1e-13

    μ = mu(x, s, τ, κ)
    sinv = inv(s)

    ## Compute right hand sides
    f1 =  η * get_rP(pb, x, τ)
    f2 = -η * get_rD(pb, y, s, τ)
    f3 =  η * get_rG(pb, x, y, κ)
    f4 = product(sinv, γ * μ) - x
    f5 = γ*μ  - τ*κ

    g = NTget_g(pb, x, s)
    w = NTget_w(pb, g)

    M = NTget_M(pb, g)
    factorize(M)

    ## Solving subsystem for dy, dτ
    wcw = hadamard(w, c, w)
    bAwcw = b + evaluate(A, wcw)

    f̂1 = f1 - evaluate(A, f4 + hadamard(w, f2, w))
    f̂2 = f3 + dot(c, f4) + f5 / τ + dot(c, hadamard(w, f2, w))

    sol1 = M \ f̂1
    sol2 = M \ bAwcw

    num   = f̂2 - dot(b, sol1) + dot(c, hadamard(w, evaluate(A, sol1), w))
    denom = dot(b, sol2) - dot(c, hadamard(w, evaluate(A, sol2), w)) + κ/τ + dot(c, wcw)

    dτ = num / denom
    dy = sol1 + sol2*dτ

    @show   norm( evaluate(A, hadamard(w, evaluate(A, dy), w)) - bAwcw * dτ - f̂1 )
    @assert norm( evaluate(A, hadamard(w, evaluate(A, dy), w)) - bAwcw * dτ - f̂1 ) < tol

    @show   norm( dot(b, dy) - dot(c, hadamard(w, evaluate(A, dy), w)) + (κ/τ + dot(c, wcw))*dτ - f̂2 )
    @assert norm( dot(b, dy) - dot(c, hadamard(w, evaluate(A, dy), w)) + (κ/τ + dot(c, wcw))*dτ - f̂2 ) < tol

    ## Deriving full solution

    ds = -evaluate(A, dy) + product(c, dτ) - f2
    dx = f4 - hadamard(w, ds, w)
    dκ = (f5 - κ*dτ) / τ

    # @show norm( evaluate(A, dx) - dτ*b - f1)
    @assert norm( evaluate(A, dx) - dτ*b - f1) < tol
    # @show norm( -evaluate(A, dy) - ds + product(c, dτ) - f2)
    @assert norm( -evaluate(A, dy) - ds + product(c, dτ) - f2) < tol
    # @show norm( dot(b, dy) - dot(c, dx) - dκ - f3)
    @assert norm( dot(b, dy) - dot(c, dx) - dκ - f3) < tol
    # @show norm( dx + hadamard(w, ds, w) - f4)
    @assert norm( dx + hadamard(w, ds, w) - f4) < tol
    # @show norm( τ*dκ + κ*dτ - f5)
    @assert norm( τ*dκ + κ*dτ - f5) < tol

    @assert norm(dot(dx, ds) + dκ*dτ - η*(1-γ-η)*(pb.nc+1)*μ) < tol

    return dx, dy, ds, dτ, dκ
end

function checksystem(pb::SDCOContext{T}, x, y, s, τ, κ, η, γ, dx, dy, ds, dτ, dκ) where {T<:Number}
    μ = mu(x, s, τ, κ)
    sinv = inv(s)
    A = pb.A
    c = pb.c
    b = pb.b

    ## Compute right hand sides
    f1 =  η * get_rP(pb, x, τ)
    f2 = -η * get_rD(pb, y, s, τ)
    f3 =  η * get_rG(pb, x, y, κ)
    f4 = product(ones(c, Dense{T}), γ * μ) - hadamard(x, s)
    f5 = γ*μ  - τ*κ

    tol = 1e-13

    # @show norm( evaluate(A, dx) - dτ*b - f1)
    @assert norm( evaluate(A, dx) - dτ*b - f1) < tol
    # @show norm( -evaluate(A, dy) - ds + product(c, dτ) - f2)
    @assert norm( -evaluate(A, dy) - ds + product(c, dτ) - f2) < tol
    # @show norm( dot(b, dy) - dot(c, dx) - dκ - f3)
    @assert norm( dot(b, dy) - dot(c, dx) - dκ - f3) < tol
    # @show norm( dx + hadamard(w, ds, w) - f4)
    @assert norm( hadamard(dx, s) + hadamard(ds, x) - f4) < tol
    # @show norm( τ*dκ + κ*dτ - f5)
    @assert norm( τ*dκ + κ*dτ - f5) < tol
end

function mu(z::PointPrimalDual{T, matT}, τ::T, κ::T) where {T<:Number, matT<:AbstractArray}
    return mu(x::PointE{T, matT}, s::PointE{T, matT}, τ, κ)
end

function mu(x::PointE{T, matT}, s::PointE{T, matT}, τ::T, κ::T) where {T<:Number, matT<:AbstractArray}
    return (dot(x, s) + τ * κ) / (get_nc(x) + 1)
end

get_rP(pb::SDCOContext, x, τ) = τ * pb.b - evaluate(pb.A, x)
get_rD(pb::SDCOContext, y, s, τ) = -evaluate(pb.A, y) - s + product(pb.c, τ)
get_rG(pb::SDCOContext, x, y, κ) = -dot(pb.b, y) + dot(pb.c, x) + κ

function solve(pb::SDCOContext{T}) where {T<:Number}

    x = ones(pb.c, Dense{T})
    y = zeros(T, pb.m)
    s = ones(pb.c, Dense{T})
    τ = T(1)
    κ = T(1)

    rP0 = get_rP(pb, x, τ)
    rD0 = get_rD(pb, y, s, τ)
    rG0 = get_rG(pb, x, y, κ)

    x0 = ones(pb.c)
    y0 = zeros(T)
    s0 = ones(pb.c)
    τ0 = 1
    κ0 = 1

    μ0 = mu(x, s, τ, κ)

    μ = μ0

    one = ones(pb.c, Dense{T})
    maxit = 5
    it = 0


    μ_hist = T[μ0]
    θ_hist = T[1]
    α_hist = T[]

    ittype = :corr

    print_header(pb)
    print_it(pb, it, x, y, s, τ, κ, rP0, rD0, last(μ_hist), last(θ_hist), 0.01)

    it += 1

    while it < maxit
        printstyled("------ iteration $it, $ittype\n", color=:red)

        μ = mu(x, s, τ, κ)
        println("μ = ", μ)
        println("| x•s - μe | = ", norm(hadamard(x, s) - μ * one))
        println(" | τκ - μ |  = ", norm(τ*κ - μ))

        if ittype == :pred
            η = T(0)
        else
            η = T(1)
        end
        γ = T(1-η)


        ## Solving system
        dx, dy, ds, dτ, dκ = solvesystem(pb, x, y, s, τ, κ, η, γ)
        checksystem(pb, x, y, s, τ, κ, η, γ, dx, dy, ds, dτ, dκ)


        # Property that should always hold
        @assert norm(dot(dx, ds) + dκ*dτ - η*(1-γ-η)*(pb.nc+1)*μ) < 1e-13

        @show norm(dx)
        @show norm(dy)
        @show norm(ds)

        ## Choosing stepsize α
        α = 0.3

        ## Updating variables
        add!(x, α*dx)
        add!(y, α*dy)
        add!(s, α*ds)
        τ += α*dτ
        κ += α*dκ

        ## Variables must remain in their cones
        @assert min(pb, x) > 0
        @assert min(pb, s) > 0
        @assert τ > 0
        @assert κ > 0

        @show min(pb, x), min(pb, s), τ, κ


        rPk = get_rP(pb, x, τ)
        rDk = get_rD(pb, y, s, τ)
        rGk = get_rG(pb, x, y, κ)

        θ = (1-α*η) * last(θ_hist)
        μ_th = (1-α*η) * (1 - α*(1-γ-μ)) * last(μ_hist)
        μ = mu(x, s, τ, κ)

        print_header(pb)
        print_it(pb, it, x, y, s, τ, κ, rPk, rDk, μ, θ, 0.01)

        ## Checking evolution rules hold, logging, it printing

        push!(θ_hist, θ)
        push!(μ_hist, μ)
        push!(α_hist, α)

        # @show norm(μ_th - mu(x, s, τ, κ))

        # @show norm(rPk - θ * rP0)
        # @show norm(rDk - θ * rD0)
        # @show norm(rGk - θ * rG0)

        # @show dot(x0, s) + dot(s0, x) + τ0 * κ + τ * κ0 - (μ/θ + θ*μ0)*(get_nc(x)+1)

        printstyled("Condition 1 (9) : |dx⋅ds + dτ⋅dκ| ≤ O(μ)   :  ", norm(dot(ds, dx) + dτ*dκ), "  ≤  ", μ, "\n", color=:yellow)
        printstyled("Condition 2 (12): Ω(1) ≤ μ/θ ≤ O(1), θ → 0 :  ", θ / μ, "    ", θ, "\n", color=:yellow)

        if ittype == :pred
            ittype = :corr
        else
            ittype = :pred
        end

        it+=1
    end

    @show μ_hist
    @show θ_hist
    @show α_hist

    nothing
end

function print_header(pb::SDCOContext)
    println("it  primerr    dualerr    gaperr       μ          θ          α     it. time")
end

function print_it(pb::SDCOContext, it, x, y, s, τ, κ, rPk, rDk, μ, θ, time)
    primfeaserr = norm(rPk, Inf) / ((1 + norm(pb.b, Inf)) * τ)
    dualfeaserr = norm(rDk, Inf) / ((1 + norm(pb.c, Inf)) * τ)
    gap = abs(dot(pb.c, x) - dot(pb.b, y)) / (τ + max(dot(pb.c, x), dot(pb.b, y)))

    @printf("%2i  %.3e  %.3e  %.3e    %.2e   %.2e   %f\n", it, primfeaserr, dualfeaserr, gap, μ, θ, 1)
end
