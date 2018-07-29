import Base.print, Base.show

export print

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
end


function print(io::IO, pt::PointE)
    for (i, mat) in enumerate(pt.mats)
        print(io, " - mat $i - ")
        display(mat)
    end
    print(io, " - vec - ")
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