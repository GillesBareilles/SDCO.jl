function print(io::IO, pb::SDCOContext)
    println(io, " * Nb of scalar variables: $(pb.nscalvar)")
    println(io, " * Nb of matrix variables: $(length(pb.nsdpvar))")
    println(io, "         individual sizes: $(pb.nsdpvar)")
    println(io, " * Nb of constraints:      $(pb.m)")
    println(io, "")
    println(io, " * Objective: $(pb.c)")
    println(io, " * Constraints:")
    for (i, ai) in enumerate(pb.A)
        println(io, "    - $i, $(pb.b[i]) == $ai")
    end
end


function print(io::IO, pt::PointE)
    for (i, mat) in enumerate(pt.mats)
        println(io, "- Sym cone $i:\n$mat")
    end
    print(io, "- R^n space:\n$(pt.vec)")
end
