import Base.print, Base.show

export print, print_pointsummary

function print(io::IO, pb::SDCOContext)
    println(io, " * Nb of scalar variables:     $(pb.nscalvar)")
    println(io, " * Nb of matrix variables:     $(length(pb.nsdpvar))")
    println(io, "         individual sizes:     $(pb.nsdpvar)")
    println(io, " * Nb of constraints:          $(pb.m)")
    println(io)
    println(io, " * Objective:")
    print(io, " ", pb.c)
    println(io)
    println(io, " * Constraints:")
    for (i, ai) in enumerate(pb.A)
        println(io, " **** ctr $i: rhs = $(pb.b[i])")
        print(io, ai)
    end
    println(io)
    for (opt, val) in pb.options
        println(io, opt, "   => ", val)
    end
end


function print(io::IO, pt::PointE)
    for (i, mat) in enumerate(pt.mats)
        print(io, " - mat $i: ")
        display(mat)
    end
    print(io, " - vec: ")
    display(pt.vec)
end

function show(io::IO, pt::PointE)
    for (i, mat) in enumerate(pt.mats)
        print(io, " - mat $i - ")
        display(mat)
    end
    print(io, " - vec - ")
    display(pt.vec)
end

function show(io::IO, z::PointPrimalDual)
    println(io, "   * Primal space - x:")
    display(z.x)
    println(io, "   * Dual space - y:")
    display(z.y)
    println(io, "   * Dual space - s:")
    display(z.s)
end

function print_pointsummary(pb::SDCOContext, z::PointPrimalDual)
    println("Primal / dual obj      :    ", get_primobj(pb, z), " / ", get_dualobj(pb, z))
    println("Primal feasability err :    ", get_primfeaserr(pb, z))
    println("Dual feasability err   :    ", get_dualfeaserr(pb, z))
    println("Min_K(x)               :    ", min(pb, z.x))
    println("Min_K(s)               :    ", min(pb, z.s))
    println("x . s                  :    ", hadamard(z.x, z.s))
end
