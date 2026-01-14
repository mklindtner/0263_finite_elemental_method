#Author: Mikkel Koefoed Lindtner (s205421)
#Exercise 1.6.a

function linear_slope(y_start, y_end, x_start, x_end)
    return (y_end - y_start) / (x_end - x_start)
end

function integration_term(a1, b1, a2, b2, a3, b3, x_m, x_i, x_ip1)
    A1_3 = a1-a3
    B1_3 = b1 - b3
    A2_3 = a2-a3
    B2_3 = b2 - b3

    first_int = (A1_3^2/3)*(x_m^3 - x_i^3 ) + A1_3*B1_3*(x_m^2 - x_i^2) + B1_3^2*(x_m-x_i)
    second_int = A2_3^2/3*(x_ip1^3 -x_m^3) + A2_3*B2_3*(x_ip1^2 - x_m^2) + B2_3^2*(x_ip1 - x_m)
    return sqrt(first_int + second_int)
end

function compute_error_decrease(fun, VX, EToV)
    elem_sz = size(EToV, 1)
    err = zeros(Float64, elem_sz)


    for i in 1:1:elem_sz
        idx_L = EToV[i, 1] #left end point
        idx_R = EToV[i, 2] #right end point
        
        # Get physical coordinates
        x_i = VX[idx_L] 
        x_ip1 = VX[idx_R]

        x_m = (x_i + x_ip1) / 2.0  # Midpoint

        # 2. Function Values 
        y_i = fun(x_i) 
        y_m = fun(x_m)
        y_ip1 = fun(x_ip1)

        a1 = linear_slope(y_i, y_m, x_i, x_m)
        b1 = y_i - a1*x_i

        a2 = linear_slope(y_m, y_ip1, x_m, x_ip1)
        b2 = y_m - a2*x_m

        a3 = linear_slope(y_i, y_ip1, x_i, x_ip1)
        b3 = y_i - a3*x_i

        err[i] = integration_term(a1, b1, a2, b2, a3, b3, x_m, x_i, x_ip1)
    end

    return err
end


#Exercise 1.6.b
function refine_marked(EToVcoarse, xcoarse, idxMarked)    
    #Find number of marks
    idxMarked_set = Set(idxMarked) #lookup is O(1) so faster later
    new_addtions = length(idxMarked_set) #every new refined point "xm" will yield a new linepiece "em". Hence "new elements = new points"
    
    #--- Setup VX Table ---
    n_vx_old = length(xcoarse) #No of points.

    
    #Number of new VX elements
    # vx_new_elems = vx_old_elems + no_of_new_elems
    vx_points_new = n_vx_old + new_addtions 
    
    #Instantiation
    vx_new = zeros(Float64, vx_points_new) 
    vx_new[1:n_vx_old] = xcoarse #set all old points
    
    
    #--- Setup EToV --- 
    n_elem_old = size(EToVcoarse,1) #Number of elements
    EToV_elements_new = n_elem_old + new_addtions
    EToV_new = zeros(Int, EToV_elements_new,2) #we need the connect from the left side and right side of the element
    
    #-- Setup Etc. ---
    vx_cnt = n_vx_old + 1 #fill in points at the end of the old points
    elem_cnt = 1  #fill elements from beginning
    
    #We are going to iterate through all the elements and look to see which needs refinement! exciting stuff at 21.20 PM, WHO DOESN'T LOVE 13H WORKDAYS FUCK YES
    for elem_id in 1:1:n_elem_old 

        #find points connected to element
        i_left = EToVcoarse[elem_id, 1] #"ei" left connection to point
        i_right = EToVcoarse[elem_id, 2] #"ei" right connection to point

        if elem_id in idxMarked_set

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
            elem_cnt += 1 #next row in EToV

            #Readjust old "ei" to be next element in EToV
            #Note that we need to reset all point connections to old "ei" because it has been pushed forward in the table
            EToV_new[elem_cnt, 1] = i_mid #set "ei" left connection to "xm"
            EToV_new[elem_cnt, 2] = i_right #set "ei" to the right connection
            elem_cnt += 1 #next row in EToV
            vx_cnt += 1 #next row in VX table
        
        else
            #--- Keep The element unrefned ---
            EToV_new[elem_cnt,1] = i_left #element remains connected to same left point
            EToV_new[elem_cnt,2] = i_right #element remains connceted to same right point
            #Notice that we do not move the VX row because this will be the next refinement point
            elem_cnt += 1 
        end

    end

    return vx_new, EToV_new
end 


#exercise 1.6.c
function AMR(u, tol, VX, EToV, max_iter)

    history_dofs = []
    history_err = []

    for iter in 1:1:max_iter
        #use 1.6.a to compute error 
        err_vec = compute_error_decrease(u, VX, EToV)
        max_err = maximum(err_vec) #NB! this is NOT the 1 norm. We can do this because we found analytical expression for the integral so we get a number for each computed error
        
        push!(history_dofs, length(VX))
        push!(history_err, max_err)

        if max_err < tol #the maximum is below the threshold
            break
        end

        idxMarks = findall(elem -> elem > tol, err_vec) #I go HARD and I take all the elmeents with a higher error and mark those fuckers.
        VX, EToV = refine_marked(EToV, VX, idxMarks)
    end

    return VX, EToV, history_dofs, history_err #return the refined points and elements table
end





using Plots

function plot_amr_results(VX, EToV, dofs, errs, u_exact)
    x_plot = range(minimum(VX), maximum(VX), length=1000)
    y_exact = u_exact.(x_plot)

    # Sort the unstructured nodes so the line plot connects them correctly
    perm = sortperm(VX)
    sorted_VX = VX[perm]
    sorted_u = u_exact.(sorted_VX) # Evaluation at nodes

    p1 = plot(x_plot, y_exact, 
        label="Exact u(x)", 
        lw=2, color=:black, 
        title="Computed vs Exact Solution",
        xlabel="x", ylabel="u(x)")

    # Plot the nodes as red dots
    plot!(p1, sorted_VX, sorted_u, 
        label="AMR Solution", 
        seriestype=:scatter, markersize=3, color=:red)
        
    # Connect nodes with dashed lines to show the piecewise linear approximation
    plot!(p1, sorted_VX, sorted_u, 
        label="", 
        lw=1, linestyle=:dash, color=:red)

    #Plot 2 Element Size Distribution
    # Calculate element sizes h = |x_R - x_L|
    n_elems = size(EToV, 1)
    element_sizes = zeros(n_elems)
    
    for i in 1:n_elems
        idx_L = EToV[i, 1]
        idx_R = EToV[i, 2]
        element_sizes[i] = abs(VX[idx_R] - VX[idx_L])
    end

    p2 = histogram(element_sizes, 
        bins=20, 
        title="Element Size Distribution", 
        xlabel="Element Size (h)", 
        ylabel="Frequency",
        legend=false, color=:blue)

    # --- Plot 3: Convergence (Error vs DOFs) ---
    p3 = plot(dofs, errs, 
        xaxis=:log, yaxis=:log, 
        marker=:circle, 
        title="Convergence: Max Error vs DOFs", 
        xlabel="Degrees of Freedom (Nodes)", 
        ylabel="Max L2 Error",
        label="Error Decay", 
        lw=2, color=:green)

    # --- Combine and Display ---
    final_plot = plot(p1, p2, p3, layout=(3, 1), size=(600, 900))
    display(final_plot)

    savefig(final_plot, "w1/1_6_d_final_plot.png")
end

u = x -> exp(-800*(x-0.4)^2) + 0.25*exp(-40*(x-0.8)^2)
VX = [0.0, 1/3, 2/3, 1.0]
EToV = [1 2; 2 3; 3 4]
tol = 1e-6
max_iter = 20

final_VX, final_EToV, history_dofs, history_err = AMR(u, tol, VX, EToV, max_iter)

plot_amr_results(final_VX, final_EToV, history_dofs, history_err, u)