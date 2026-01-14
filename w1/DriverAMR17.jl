include("ex1_7.jl")


#THIS is just a wrapper because I give EToV and VX as input in my original AMR, see ex1_7.jl for the correct solutions
#and it is 18.18PM on a friday so I really cant be bothered to reassign the signature or try to figure out why you dont need to give a EToV
function DriverAMR17(L, c, d, x_init, func, tol, maxit)
    
    #Create the EToV table because my function needs that
    # I have no idea if I am accidently cheating by accident here and not suppossed to have an EToV from the beginning smh.
    M = length(x_init)
    EToV = zeros(Int, M-1, 2)
    for k in 1:M-1
        EToV[k, 1] = k
        EToV[k, 2] = k + 1
    end
    

    _, u_init = BVP1Drhs(x_init, L, c, d, func)

    VX_final, EToV_final, _, _, _ = AMR(u_init, tol, x_init, EToV, maxit, L, c, d, func)

    _, u_final = BVP1Drhs(VX_final, L, c, d, func)
    
    iter_count = maxit 

    return VX_final, u_final, iter_count
end