function NesterovTodd()
    # Step 1. Compute necessary quatities
    # g::PointE = NTget_g(x, s)
    # w::PointE = NTget_w(g)
    
    # Step 2. Solve M.dy = A(x-\mu inv(s))
    # M = NTget_M(g, (a_i))
    # Choleski fact, solve

    # Step 3. 
    # ds = -A*(dy)
    # dx = \mu inv(s) -x -w \otimes ds \otimes w

    # Step 4.
    # Check that z = (x, y, s) is strict feasable
    # Check that z+dz is strict feasable too.
end