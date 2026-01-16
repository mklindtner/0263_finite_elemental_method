#exercise 2.1.a
using Plots

function xy(x0, y0, L1, L2, noelms1, noelms2)

    #Setup
    L1_offset = x0 + L1
    L2_offset = y0 + L2

    dx = L1 / noelms1
    dy = L2 / noelms2

    total_nodes = (noelms1 + 1)*(noelms2+1)

    VX = zeros(total_nodes, 1)  #No. of elements in x-axis
    VY = zeros(total_nodes, 1)  #No. of elements in y-axis

    cnt = 1

    #Loop
    for x_id in x0:dx:L1_offset #x_0 + i*dx
        for y_id in L2_offset:-dy:y0
            VX[cnt] = x_id
            VY[cnt] = y_id 
            cnt += 1
        end
    end

    return VX, VY
end

#test case I made myself for easier check
# x0 = 0
# y0 = 0
# L1 = 2
# L2 = 2
# noelms1 = 4
# noelms2 = 3

#Works
#test case correct 
# x0 = -2.5
# y0 = -4.8
# L1 = 7.6
# L2 = 5.9 
# noelms1 = 4
# noelms2 = 3

# VX, VY = xy(x0, y0, L1, L2, noelms1, noelms2)

# p1 = scatter(VX, VY, 
#         marker=:circle, 
#         title="Vx, VY", 
#         xlabel="Vx elements", 
#         ylabel="Vy elements",
#         label="plot of VX, VY",
#         legend=:bottomleft)


# display(p1)
# savefig(p1, "w2/ex1_1_a_img.png")


#Exercise 2.1.b
function conelmtab(noelms1, noelms2)
    NoOfElements = noelms1*noelms2*2
    EToV = zeros(Int, NoOfElements,3)
    elem_id = 1
    height = noelms2 + 1
    width = noelms1 + 1

    #instead get the x points, y points
    for i in 1:1:(width-1) #xpoints, the columns, note that we cannot access the last element!
        for j in 1:1:(height-1) #ypoints, the rows, same idea as aobve!

            #Make a square
            #NB!!!! The length, the distance between columns, matters!!!
            
            #Top point 
            top_left         = (i-1)*height +   j                  
            top_right        = i*height     +   j            

            #Buttom points
            buttom_left     = (i-1)*height      +   (j+1)      
            buttom_right    = i*height          +   (j+1)       


            #First element
            EToV[elem_id, 1] = Int(top_right)
            EToV[elem_id, 2] = Int(top_left)
            EToV[elem_id, 3] = Int(buttom_right)

            elem_id += 1

            #Second element
            EToV[elem_id, 1] = Int(buttom_left)
            EToV[elem_id, 2] = Int(buttom_right)
            EToV[elem_id, 3] = Int(top_left)

            elem_id += 1
        end
    end

    return EToV
end

#Test case
# noelms1 = 4
# noelms2 = 3


# EToV = conelmtab(noelms1,noelms2)

# display(conelmtab(noelms1,noelms2))



    #Exercise 2.2.a
    function basfun(n, VX, VY, EToV)
        #Calculate delta
        
        #Global variables
        ea = EToV[n, 1]
        eb = EToV[n, 2]
        ec = EToV[n, 3]
        
        #Local variables 
        xi, yi = VX[ea], VY[ea]
        xj, yj = VX[eb], VY[eb]
        xk, yk = VX[ec], VY[ec]
        
        #Just so I dont fuck it up
        x1,x2, x3  = xi, xj, xk
        y1,y2,y3 = yi, yj,yk
        
        delta = 1/2*(x2*y3- y2*x3 - (x1*y3 - y1*x3) + x1*y2 - y1*x2)
        
        #Make abc
        abc = zeros(3, 3)

        #get all coefficients
        #the example below 2.3 is the counterclockwise rotation we are looking for indexing.

        #(i,j,k) = (1,2,3)
        abc[1, 1] = x2*y3 - x3*y2
        abc[1, 2] = y2 - y3
        abc[1, 3] = x3 - x2

        #(i.j,k) = (2,3,1)
        abc[2, 1] = x3*y1 - x1*y3
        abc[2, 2] = y3 - y1
        abc[2, 3] = x1 - x3

        #(i,j,k) = (3,1,2)
        abc[3, 1] = x1*y2 - x2*y1
        abc[3, 2] = y1 - y2
        abc[3, 3] = x2 - x1

        
        return delta, abc
    end

# n = 9
# delta, abc = basfun(n, VX, VY, EToV) #I have no idea why I get something different from the results, I suspect he fucked upa nd made them clockwise
# display(abc)
# println(delta)


#Test case
#VX, VY and EToV initialized as in exercise 2.1.a

#Exercise 2.2.b
function outernormal(n,k,VX,VY, EToV)
    #k is assumed to be 1,2 or 3 because we are IN 3D! (say it like the old commercials.. IN 3D)
    e1 = EToV[n, 1]
    e2 = EToV[n, 2]
    e3 = EToV[n, 3]
    
    if k == 1 #we are 
        p1, p2 = e1, e2        
    elseif k == 2
        p1, p2 = e2, e3
    elseif k == 3
        p1, p2 = e3, e1
    else
        error("FUCKidity doo you gave a wrong k awrroooo")
    end

    
    t1 = VX[p2] - VX[p1]
    t2 = VY[p2] - VY[p1]
    v_magnitude = sqrt(t1^2 + t2^2)

    # Nx = -t1 / v_magnitude 
    # Ny = t2 / v_magnitude

    #Setup for 2.7
    Nx = t2 / v_magnitude
    Ny = -t1 / v_magnitude 

    return Nx, Ny
end

# n = 9
# r1 = outernormal(n, 1, VX,VY, EToV)
# r2 = outernormal(n, 2, VX,VY, EToV)
# r3 = outernormal(n, 3, VX,VY, EToV)



#Exercise 2.3.a
lambda1 = 1
lambda2 = 1


#M Antal noder
#N er antal elementer
#qt Assume is (x,y) heat for a point
function assembly(VX,VY, EToV, lam1, lam2, qt)
    M = length(VX) #No. of nodes

    A = zeros(M, M)
    b = zeros(M, 1)
    N = size(EToV, 1)

    for n in 1:1:N
        #Looks up global, then local elements and calculates delta and local coefficients
        delta, abc = basfun(n, VX,VY, EToV) #very delicious, fix loop 

        global_ea,global_eb ,global_ec = EToV[n, 1], EToV[n, 2], EToV[n, 3] #get global coordnates 
        # x1, x2, x3 = VX[global_ea], VX[global_eb], VX[global_ec]
        # y1,y2,y3 = VY[global_ea], VY[global_eb], VY[global_ec]

        # qx, qy, qz = qt[global_ea], qt[global_eb], qt[global_ec]  #get local coordinates 
        # q1, q2, q3 = qt[x1,y1], qt[x2,y2], qt[x3,y3]
        q1,q2,q3 = qt[global_ea], qt[global_eb], qt[global_ec] 

        for r in 1:1:3
            qhat_tilde = delta/3.0 * (q1+q2+q3) / 3.0 #2.28 - > Assuming delta (area) is always positive!

            i = EToV[n,r] #Get global index
            b[i] = b[i] + qhat_tilde 

            for s in 1:1:3
                j = EToV[n, s] #get global index
                k_rs = 1/(4*delta) * (lam1*abc[r,2]*abc[s,2] + lam2*abc[r,3]*abc[s,3]) #2.27 -> assumes area is positive
                A[i,j] = A[i,j] + k_rs
            end
        end
    end 


    return A, b
end


#Test case 1
# x0, y0 = 0, 0 
# L1, L2 = 1, 1 
# Lam1, Lam2 = 1,1 
# noelms1, noelms2 = 4, 3

# VX,VY = xy(x0, y0, L1, L2, noelms1, noelms2) 
# EToV = conelmtab(noelms1, noelms2) 
# qhat = (x,y) -> return 0 

# A,b = assembly(VX,VY, EToV, Lam1, Lam2, qhat)
# display(A)
# # println("-"^30)
# display(b)
# # println("-"^30)



# #Test case 2
# x0, y0 = -2.5, -4.8
# L1, L2 = 7.6, 5.9
# noelms1, noelms2 = 4, 3
# qhat = (x,y) -> return -6*x + 2*y - 2
# A,b = assembly(VX,VY, EToV, Lam1, Lam2, qhat)
# display(A)
# display(b)



#Execise 2.4.a
function dirbc(bnodes, f,A,b)
    M = length(b) #No of nodes.
    all_nodes = 1:M
    omega = setdiff(all_nodes, bnodes) #remove boundary nodes

    #if is redundant because bnodes contain hte boundary so it is always true for i \in bnodes
    for k in 1:1:length(bnodes) #get all boundary points
        i = bnodes[k]
        A[i,i] = 1
        b[i] = f[k] #boundary value for f(x,y)

        for j in 1:1:M
            if i != j
                A[i,j] = 0
                if j in omega #this is techincally redundant because I could just loop over omega, but w/e
                    b[j] = b[j] - A[j,i]*f[k]
                    A[j,i] = 0
                end
            end
        end
    end
    return A, b
end

#Find boundary points

#assumes this is a square
function find_robin_boundary(noelms1, noelms2) 
    #Setup indicies for columns
    Nx = noelms1 + 1    #noelms1 is number columns 
    Ny = noelms2 + 1    #noelms is number of rows  
    M = Nx*Ny #all points 

    #left column
    g_col_idx_y = 1:1:Ny 
    #left column x is always first index, so no need to set it

    #right column
    r_offset = M - Ny + 1 
    g_col_idx_x = r_offset:1:M 
    
    top_idx = 1:Ny:M  #skip each column row
    bot_idx = Ny:Ny:M  #jump each col starting at Ny 

    boundary_points = unique([g_col_idx_y; g_col_idx_x; top_idx;bot_idx]) 
    return boundary_points 
end


# print(sort(find_robin_boundary(4,3)), length(find_robin_boundary(4,3)))
# #Test case 1 
# x0, y0 = 0, 0 
# L1, L2 = 1, 1
# noelms1 = noelms2 = 4
# qhat_tilde = (x,y) -> return 0 
# f = (x,y) -> return 1
# Lam1, Lam2 = 1, 1
# qhat = (x,y) -> return 0

# #Reuse from exericse 2.3
# VX,VY = xy(x0, y0, L1, L2, noelms1, noelms2) 
# EToV = conelmtab(noelms1, noelms2) 

# #Find bnodes and function values
# bnodes = find_robin_boundary(noelms1, noelms2)
# x_boundary, y_boundary = VX[bnodes], VY[bnodes]
# f_vals = f.(x_boundary, y_boundary)

# #Assembly and dirbc
# A,b = assembly(VX,VY, EToV, Lam1, Lam2, qhat)
# A,b = dirbc(bnodes, f_vals, A, b)

# display(A)
# display(b)


# #Test case 2
# x0, y0 = -2.5, -4.8
# L1, L2 = 7.6, 5.9
# noelms1, noelms2 = 4, 3
# qhat = (x,y) -> -6*x2*y-2
# f = (x,y) -> x^3 -x^2*y + y^2 - 1

# #Reuse from exericse 2.3
# VX,VY = xy(x0, y0, L1, L2, noelms1, noelms2) 
# EToV = conelmtab(noelms1, noelms2) 

# #Find bnodes and function values
# bnodes = find_robin_boundary(noelms1, noelms2)
# x_boundary, y_boundary = VX[bnodes], VY[bnodes]
# f_vals = f.(x_boundary, y_boundary)

# #Assembly and dirbc
# A,b = assembly(VX,VY, EToV, Lam1, Lam2, qhat)
# A,b = dirbc(bnodes, f_vals, A, b)



#Ex 2.5
import LinearSolve as LS

function solve_pde_boundary_2D(x0,y0, L1,L2,noelms1, noelms2, Lam1, Lam2, qhat, f)

    #Generate MESH
    VX, VY = xy(x0,y0, L1, L2, noelms1, noelms2)
    EToV = conelmtab(noelms1, noelms2)    
    
    #Assemble solution
    q_vals = qhat.(VX, VY)
    A,b = assembly(VX,VY, EToV, Lam1, Lam2, q_vals) 

    #Impose boundary conditions
    bnodes = find_robin_boundary(noelms1, noelms2)
    x_boundary, y_boundary = VX[bnodes], VY[bnodes]
    f_vals = f.(x_boundary, y_boundary)
    A,b = dirbc(bnodes, f_vals, A, b)

    #Solve Ax = b
    sol = A \ b 
    return sol, VX, VY
end

function est_error(u_fem, u_exact, VX, VY)
    return maximum(abs.(u_fem - u_exact.(VX, VY)))  #eq. 2.52
end

#Test case 1
# u_exact = (x,y) -> x^3 - x^2*y + y^2 - 1
# f = u_exact

# x0, y0 = -2.5, -4.8
# L1, L2 = 7.6, 5.9
# noelms1, noelms2 = 4, 3
# Lam1, Lam2 = 1, 1
# q = (x,y) -> -6*x + 2*y - 2 

# sol, VX, VY = solve_pde_boundary_2D(x0, y0, L1, L2, noelms1, noelms2, Lam1, Lam2, q, f)
# est = est_error(sol,u_exact, VX, VY)
# println("Error sol´: ", est)


#Test case 2

#TO DO


#Exercise 2.6.a

#So unfortunately I found an edge cases where construct beds will fail: if the hypothenous connects two edge points (e.g. a triangle at a corner) then it will be errornously counted as an edge
# function constructBeds(VX, VY, EToV, tol, fd,x0,y0)
#     M = size(VX,1)
#     is_b_node = zeros(bool, M)

#     #Find all boundary points 
#     # fd.(VX, VY, x0, y0) #I think this is a vectorized version for the loop below, no time to test though

#     for i in 1:1:M
#         xi, yi = VX[i], VY[i]
#         dst = fd(xi,yi, x0, y0)
#         if dst < tol
#             is_b_node[i] = true
#         end
#     end 

#     beds = []
    
#     #Find edges between boundary points
#     for i in 1:1:size(EToV, 1)
#         n1_idx, n2_idx, n3_idx = EToV[i, 1], EToV[i, 2], EToV[i, 3] #get indicies for the three normals making up an element

#         #check if n1 and n2 is connected
#         if is_b_node[n1_idx] && is_b_node[n2_idx]
#             push!(beds, [i,1])
#         end

#         #check if n2 and n3 is connected
#         if is_b_node[n2_idx] && is_b_node[n3_idx]
#             push!(beds, [i,2])
#         end

#         #check if n3 and n1 is connected
#         if is_b_node[n3_idx] && is_b_node[n1_idx]
#             push!(beds,[i,3])
#         end
#     end
    
#     return stack(beds,dim=1)
# end



#Distance function made to find boundaries
function fd(x0,y0, x, y)
    #x0,y0 is the anchor of the boundary
    #x,y is the local points 

    # loc = abs(bx - g_x) + abs(g_y - by)
    # first boundary 
    row_x = abs(x0 - x) #y coordinate are zero because we would have y0 - y0
    col_y = abs(y0 - y) #x coordinate are zero because we would have x0 - x0

    loc = min(row_x, col_y) #find closest line
    return loc
end




#I talked to two other groups, they calculated the midpoints, but didn't really understand why: I now know why this works, because if the point points are on a boundary and their mid point iso n the boundary => then then points are boundary notes. This solves the issue above (befcause the midpoint of an edge element is not on the boundary)
function constructBeds(VX, VY, EToV, tol, fd,x0,y0)
    beds = []

    for n in 1:1:size(EToV, 1)
        local_edges = [[1, 2], [2, 3], [3, 1]] #this is just the counter-clockwise rotation defined in the chapter

        for (line_idx, nodes) in enumerate(local_edges) #For every element in EToV, we look at all the lines of a triangle
            r = nodes[1] #local node 1
            s = nodes[2] #local node 2

            i = EToV[n, r] #global index for node1
            j = EToV[n, s] #global index for node2

            xm = (VX[i] + VX[j]) / 2.0
            ym = (VY[i] + VY[j]) / 2.0

            dist = fd(x0, y0, xm, ym)

            if abs(dist) < tol
                push!(beds, [n , line_idx])
            end
        end
    end

    return stack(beds, dims=1) #make into matrix for my dirty use later
end 

# return reduce(hcat, beds)
# return beds

#Exercise 2.6.b
function neubc(VX, VY, EToV, beds, q, b)
    for p in 1:1:size(beds,1)
        n = beds[p, 1]
        r = beds[p, 2]
        s = -1
        if r == 1
            s = 2
        elseif r == 2
            s = 3
        elseif r == 3
            s = 1
        else
            error("Wopsi: Invalid local node index")
        end

        i = EToV[n, r]
        j = EToV[n, s]
        xi, yi = VX[i], VY[i]
        xj, yj = VX[j], VY[j]

        q1 = q[p]/2 * sqrt((xj-xi)^2 + (yi-yj)^2)
        q2 = q1

        b[i] = b[i]- q1
        b[j] = b[j] - q2
    end
    return b
end


#Exercise 2.7.a
function compute_nue_boundary(beds, VX, VY, EToV, u_exact_gradient)
    # beds: [element_index, line_index]
    # exact_u_gradient: Function returning (du_dx, du_dy) at a point (x,y)
    
    num_edges = size(beds, 1)
    q_vals = zeros(num_edges)
    local_edges = [[1, 2], [2, 3], [3, 1]] #counterclcokwise ordering

    for p in 1:num_edges
        #Find line segment
        n = beds[p, 1]          # Element
        line_idx = beds[p, 2]   # Line index => 1,2 or 3

        r = local_edges[line_idx][1] 
        s = local_edges[line_idx][2]
        
        i = EToV[n, r] #Global idx for point 1
        j = EToV[n, s] #Global idx for point 2

        #Evaluate Gradient of exact u at midpoint
            #from constructBeds I know this is a guaranteed boundary point
        mx = (VX[i] + VX[j]) / 2.0
        my = (VY[i] + VY[j]) / 2.0

        # FINALLY Exercise 2.2b makes sense
        nx, ny = outernormal(n, line_idx, VX, VY, EToV)

        du_dx, du_dy = u_exact_gradient(mx, my)

        # Compute Flux q = -u_n = -(\grad u \dot n)
            # Problem defines: u_n = -q => q = -u_n
            # Ex 2.7 says u_n = -q. 
            # So q = -u_n = -(du_dx * nx + du_dy * ny)
        q_vals[p] = -(du_dx * nx + du_dy * ny)
    end

    return q_vals
end


function compute_dirchlet_boundary(x0,y0, VX, VY, fd_dirichlet,tol)
    bnodes = []

    for i in 1:1:size(VX,1)
        xi, yi = VX[i], VY[i]

        dst = fd_dirichlet(x0,y0, xi, yi)

        if abs(dst) < tol
            push!(bnodes, i)
        end
    end
    return bnodes
end

function solve_pde_neumann_2D(x0,y0, L1, L2, noelms1, noelms2, tol, u_exact, u_exact_gradient, fd_neumann, fd_dirichlet, q_tilde_exact)
    Lam1, Lam2 = 1, 1 #This is always 1 so no need to put it in the argument

    #Step 1. algorithm 4 -> Generale MESH and assemble solution
    VX, VY = xy(x0,y0, L1, L2, noelms1, noelms2)     #Generate MESH
    EToV = conelmtab(noelms1, noelms2)  #Generate elements
    
    #Assemble solution 
    q_vec = q_tilde_exact.(VX, VY)
    A,b = assembly(VX,VY, EToV, Lam1, Lam2, q_vec) 

    #Step 2. Algorithm 8 -> #Impose boundary conditions
    beds = constructBeds(VX,VY, EToV, tol, fd_neumann, x0, y0)  #consturct boundary points using neumann measure

    q = compute_nue_boundary(beds, VX, VY, EToV, u_exact_gradient)     #find the boundary values q(x,y)

    b = neubc(VX,VY, EToV, beds, q, b)     #find the neumann boundary conditions for bottom and left edge
    
    #find Dirichlet boundary condition for top and righte edge (we must mix neumann boundary conditions to ensure uniqueness)
    bnodes = compute_dirchlet_boundary(x0,y0, VX, VY, fd_dirichlet, tol) #NB! dirchlet must overwrite neumann boundary, so always later between the two!

    #Find f_vals for dirchlet boundary
    f_vals = u_exact.(VX[bnodes], VY[bnodes])

    #Step 3. algorithm 6 -> flex on bitches if this works
    A,b = dirbc(bnodes, f_vals, A,  b)

    #Solve Ax = b
    sol = A \ b 
    return sol, VX, VY
end


#Test case 1

#Parameters
x0, y0 = -2.5, -4.8
L1, L2 = 7.6, 5.9
Lam1, Lam2 = 1, 1 
noelms1, noelms2 = 4, 3


#Callbacks
u_exact = (x,y) -> 3*x+5*y-7                                                        #exact solution 
u_exact_gradient = (x,y) -> return 3,5                                              #w.r.t x and w.r.t y
fd_neumann = (x0, y0, xi, yi) -> min(abs(xi - x0), abs(yi - y0))                    #Neumann boundary distances for a square
fd_dirichlet = (x0,y0, xi, yi) -> min(abs(xi - (x0 + L1)), abs(yi - (y0 + L2)))     #Dirichlet boundary distnaces for a square
q_tilde_exact = (xi,yi) -> 0.0                                                      #the laplacian of any linear function is 0 LMAO im a fucking giAnuS (a giant anus I actually calculatedt this analytically)
tol=1e-4

sol, VX, VY = solve_pde_neumann_2D(x0, y0, L1, L2, noelms1, noelms2, tol, u_exact, u_exact_gradient, fd_neumann, fd_dirichlet, q_tilde_exact)


#Evaluate case 1
exact_values_at_nodes = u_exact.(VX, VY)
error_vec = abs.(sol - exact_values_at_nodes)
E = maximum(error_vec)

println("Test Case 1 Results:")
println("u-hat_j ", sol)
println("Max Error E = ", E)