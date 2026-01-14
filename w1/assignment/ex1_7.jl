#Author: Mikkel Koefoed Lindtner (s205421)
#Exercise 1.7.b
function L2_norm(a0,ah,h)
    return h/3 * (a0^2 + a0*ah + ah^2)
end

function errorestimate(xc, xf, uhc, uhf, EToVc, EToVf, Old2New)
    #Setup coarse
    c_elem_sz = size(EToVc, 1)
    
    #Setup fine
    f_elem_sz = size(EToVf, 1)

    #Setup miscellanous
    err = zeros(Float64, c_elem_sz) #we can only error estimates for as many coarse elements as we have

    for e_id in 1:1:f_elem_sz

        #Coarse Info
        c_e_id = Old2New[e_id] #Get old element id
        
        #get coarse points
        c_idx_L = EToVc[c_e_id, 1] #left point of course elmeent
        c_idx_R = EToVc[c_e_id, 2] #right point of course element

        c_xL = xc[c_idx_L] #get left element
        c_xR = xc[c_idx_R] #get right element
        h_coarse = c_xR - c_xL #I am assuming that xL < xR becaues we are in 1D


        #get coarse function values
        c_uL = uhc[c_idx_L] #left function coarse value
        c_uR = uhc[c_idx_R] #right function coarse value 

        #Fine Info
        f_idx_L = EToVf[e_id, 1] #Left point of fine element
        f_idx_R = EToVf[e_id, 2] #Right point of fine element

        #get fine point 
        f_xL = xf[f_idx_L] #get left fine point
        f_xR = xf[f_idx_R] #get right fine point
        h_fine = f_xR - f_xL

        #Get fine function values
        f_uL = uhf[f_idx_L]
        f_uR = uhf[f_idx_R]


        #Interpolation

        #find r points
        r_L = (f_xL - c_xL) / h_coarse #r left point
        r_R = (f_xR - c_xL) / h_coarse #r right point
        
        
        #interpolation for coarse poin ts
        u_coarse_L = c_uL*(1-r_L) + c_uR*r_L
        u_coarse_R = c_uL*(1-r_R) + c_uR*r_R

        #difference between fine interpolation and coarse interpolation
        a0 = f_uL - u_coarse_L
        ah = f_uR - u_coarse_R

        #integration step and store result
        err[c_e_id] += L2_norm(a0,ah,h_fine)
    end

    return sqrt.(err)
end 


#exercise1.7.c - Basically 1.6.b but we now keep track of the old elements we changed
function refine_marked(EToVcoarse, xcoarse, idxMarked)    
    #Find number of marks
    idxMarked_set = Set(idxMarked) #lookup is O(1) so faster later
    new_addtions = length(idxMarked_set) #every new refined point "xm" will yield a new linepiece "em". Hence "new elements = new points"
    
    #Setup VX Table
    n_vx_old = length(xcoarse) #No of points.

    #Number of new VX elements
    # vx_new_elems = vx_old_elems + no_of_new_elems
    vx_points_new = n_vx_old + new_addtions 
    
    #Instantiation
    vx_new = zeros(Float64, vx_points_new) 
    vx_new[1:n_vx_old] = xcoarse #set all old points
    
    
    #Setup EToVf table
    n_elem_old = size(EToVcoarse,1) #Number of elements
    EToV_elements_new = n_elem_old + new_addtions
    EToV_new = zeros(Int, EToV_elements_new,2) 
    
    #Setup Etc
    vx_cnt = n_vx_old + 1 #fill in points at the end of the old points
    elem_cnt = 1  #fill elements from beginning

    Old2New = zeros(Int, EToV_elements_new, 1) #tracker for old elements changed

    
    #We are going to iterate through all the elements and look to see which needs refinement! exciting stuff at 21.20 PM, WHO DOESN'T LOVE 13H WORKDAYS FUCK YES
    for elem_id in 1:1:n_elem_old 

        #find points connected to element
        i_left = EToVcoarse[elem_id, 1] #"ei" left connection to point
        i_right = EToVcoarse[elem_id, 2] #"ei" right connection to point

        if elem_id in idxMarked_set #O(1) runtime, get fucked bitches
            #get points connected to element
            xi = xcoarse[i_left]  #left point
            xip1 = xcoarse[i_right] #right point
            
            #Add new elemnt to VX
            xm = (xi+xip1) / 2.0 #new x point
            vx_new[vx_cnt] = xm #add to new point "xm" VX table

            i_mid = vx_cnt #new point in VX

            #Add new "em" element connections to EToV
            EToV_new[elem_cnt, 1] = i_left #set left side of "em" to the previous left connection "ei"
            EToV_new[elem_cnt, 2] = i_mid #set the right side of "em" to the new point "xm"
            
            #Track "em" change
            Old2New[elem_cnt] = elem_id #we keep track of the old index of the old "e_i" element that was changed
            elem_cnt += 1 #next row in EToV


            #Readjust old "ei" to be next element in EToV
            #Note that we need to reset all point connections to old "ei" because it has been pushed forward in the table
            EToV_new[elem_cnt, 1] = i_mid #set "ei" left connection to "xm"
            EToV_new[elem_cnt, 2] = i_right #set "ei" to the right connection
            #Track "old "ei" as requested
            Old2New[elem_cnt] = elem_id

            elem_cnt += 1 #next row in EToV
            vx_cnt += 1 #next row in VX table
        else
            #Keep The element unrefned
            EToV_new[elem_cnt,1] = i_left #element remains connected to same left point
            EToV_new[elem_cnt,2] = i_right #element remains connceted to same right point
            #Notice that we do not move the VX row because this will be the next refinement point

            #Track no changes
            Old2New[elem_cnt] = elem_id

            elem_cnt += 1 
        end

    end

    return vx_new, EToV_new, Old2New
end 



#Exercise1.7.e
using LinearAlgebra
using Plots

function assembly(M,h, fval)
    A = zeros(M,M)
    b = zeros(M)
    for i = 1:1:M-1
        #my lovely little k-matrix
        k1_1 = 1/h[i]   + h[i]/3
        k1_2 = -1/h[i]  + h[i]/6
        k2_2 = 1/h[i]   + h[i]/3
        
        #Setup A like algorithm 1
        A[i,i]     += k1_1
        A[i,i+1]   = k1_2 
        A[i+1,i+1] = k2_2

        #RHS dotproduct from final analytical RHS
        b1 = -(h[i]/6) * (2*fval[i] + 1*fval[i+1])
        b2 = -(h[i]/6) * (1*fval[i] + 2*fval[i+1])
        b[i]   += b1
        b[i+1] += b2
    end
    return A,b
end


function boundary_conditions(A,b,M,c,d)
    #A matrix 
    A[1,1] = 1
    A[M,M] = 1
    
    #Left boundary
    b[1] = c
    b[2] = b[2]-A[1,2]*c
    A[1,2] = 0
    
    #right boundary
    b[M] = d
    b[M-1] = b[M-1] - A[M-1,M]*d
    A[M-1,M] = 0
    return A,b 
end

function BVP1Drhs(x,L,c,d,func)
    #Sort refine_marked
    p = sortperm(x)  #this line the ai told me to do

    #Intitialize
    x = x[p]

    h = diff(x) 
    fx = func.(x)
    M = length(x)
    
    #Assembly process
    A = zeros(M,M)
    b = zeros(M,1)
    A,b = assembly(M,h, fx)
    A,b = boundary_conditions(A,b,M,c,d)

    # SOLVE SYSTEM
    chol = cholesky(Symmetric(A, :U), check=false)
    if issuccess(chol)
        u_sorted = chol \ b
    else
        println("A is not positive definite")
        return
    end
    
    #I have to re-sort because I fucked up on the mesh grid and didn't sort them, IDK man. life sucks. 
    u = zeros(M)
    u[p] = u_sorted
    return x, u
end



function AMR(u, tol, VX, EToVc, max_iter, L,c,d,f_RHS)
    history_dofs = []
    history_err = []
    history_cpu = [] # New array for timing in f

    u_curr = u #coarse solution

    for iter in 1:1:max_iter

        #Generate the refined mesh
        VXf, EToVf, Old2Newf = refine_marked(EToVc, VX, 1:size(EToVc,1)) #the previous EToVf is now EToVc
        _, u_ref = BVP1Drhs(VXf, L, c, d, f_RHS)
        
        #use 1.7.b to compute error 
        err_vec = errorestimate(VX, VXf, u_curr, u_ref, EToVc, EToVf, Old2Newf)
        max_err = maximum(err_vec) #NB! this is NOT the 1 norm. We can do this because we found analytical expression for the integral so we get a number for each computed error
        
        #store results
        push!(history_dofs, length(VX))
        push!(history_err, max_err)

        if max_err < tol #the maximum is below the threshold
            println("Converged at iteration $iter")
            break
        end

        idxMarks = findall(elem -> elem > tol, err_vec) #I go HARD and I take all the elmeents with a higher error and mark those fuckers.
        VX, EToVc, _ = refine_marked(EToVc, VX, idxMarks) #now 

        #added this for the CPU time
        t_elapsed = @elapsed begin
            _, u_curr = BVP1Drhs(VX, L, c, d, f_RHS)
        end
        push!(history_cpu, t_elapsed)
    end

    return VX, EToVc, history_dofs, history_err,history_cpu #return the refined points and elements table
end

#This is the exact solution for u''-u = f(x)
function f_RHS(x)
    term1 = (1600 * x^2 - 2560 * x + 1003.75) * exp(-1.6 * (5 * x - 4)^2)
    term2 = (2.56e6 * x^2 - 2.048e6 * x + 4.07999e5) * exp(-32 * (5 * x - 2)^2)
    return term1 + term2
end

#For testing if BVP1Drhs works w. exact solution i.e. if I have a fuck up on indicies
# x, u_approx = BVP1Drhs(L,c,d,M, f_RHS, u_exact) #BV1PD


#Setup
#Inititalize EToV
M = 69 #first coarse grid
EToV = zeros(Int, M-1, 2)
for i in 1:M-1
    EToV[i, 1] = i      # Left point in element
    EToV[i, 2] = i + 1  # Right point in elmeent
end

L = 1.0
u_exact = x -> exp(-800*(x-0.4)^2) + 0.25*exp(-40*(x-0.8)^2)
c = u_exact(0) #could also just look them up on maple but cha
d = u_exact(1) #could also just look them up on maple but cha
VX = collect(range(0, L, length=M))

#find initial solution to feed into 
_, u_init = BVP1Drhs(VX, L, c, d, f_RHS) 


#Run AMR
tol = 1e-5 
max_iter = 9
println("Starting AMR...")

VX_final, EToV_final, history_dofs, history_err, history_cpu = AMR(u_init, tol, VX, EToV, max_iter, L, c, d, f_RHS)
println("AMR Finished with $(length(VX_final)) points.")


function required_plots(h_dofs, h_err,h_cpu)
    # Plot 1: Convergence (Error vs DoF)
    p1 = plot(h_dofs, h_err, 
        xaxis=:log, yaxis=:log, 
        marker=:circle, 
        title="Convergence", 
        xlabel="DoF (M)", 
        ylabel="Max Error",
        label="Error",
        legend=:bottomleft)

    # Plot 2: CPU Time (Time vs DoF)
    p2 = plot(h_dofs[1:length(h_cpu)], h_cpu, 
        xaxis=:log, yaxis=:log, 
        marker=:square, 
        color=:green,
        title="Computational Cost", 
        xlabel="DoF (N)", 
        ylabel="Solver Time (s)",
        label="CPU Time")

    # Plot 3: Mesh Adaptation
    x_plot = range(0, 1, length=1000)
    p3 = plot(x_plot, u_exact.(x_plot), 
        line=:dash, color=:grey, label="Exact", title="Final Mesh")
    scatter!(p3, VX_final, u_exact.(VX_final), 
        markersize=2, color=:red, label="Grid Nodes")

    # Combine all 3 plots
    final_plot = plot(p1, p2, p3, layout=(1,3), size=(1200, 400))
    display(final_plot)
    savefig(final_plot, "AMR_Full_Analysis.png")
end

# required_plots(history_dofs, history_err, history_cpu)
