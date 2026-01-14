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

#test case easy
# x0 = 0
# y0 = 0
# L1 = 2
# L2 = 2
# noelms1 = 4
# noelms2 = 3

#Works
#test case correct 
x0 = -2.5
y0 = -4.8
L1 = 7.6
L2 = 5.9 
noelms1 = 4
noelms2 = 3

VX, VY = xy(x0, y0, L1, L2, noelms1, noelms2)

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
noelms1 = 4
noelms2 = 3


EToV = conelmtab(noelms1,noelms2)

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

n = 9
delta, abc = basfun(n, VX, VY, EToV) #I have no idea why I get something different from the results, I suspect he fucked upa nd made them clockwise
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
    
    #do I care if n is even or un even? NO! But Idk why ?

    #find start node
    #find end node

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

    Nx = t1 / v_magnitude
    Ny = t2 / v_magnitude

    return Nx, Ny
end

n = 9
r1 = outernormal(n, 1, VX,VY, EToV)
r2 = outernormal(n, 2, VX,VY, EToV)
r3 = outernormal(n, 3, VX,VY, EToV)



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
        x1, x2, x3 = VX[global_ea], VX[global_eb], VX[global_ec]
        y1,y2,y3 = VY[global_ea], VY[global_eb], VY[global_ec]

        # qx, qy, qz = qt[global_ea], qt[global_eb], qt[global_ec]  #get local coordinates 
        q1, q2, q3 = qt(x1,y1), qt(x2,y2), qt(x3,y3)

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
x0, y0 = 0, 0 
L1, L2 = 1, 1 
Lam1, Lam2 = 1,1 
noelms1, noelms2 = 4, 3

VX,VY = xy(x0, y0, L1, L2, noelms1, noelms2) 
EToV = conelmtab(noelms1, noelms2) 
qhat = (x,y) -> return 0 

A,b = assembly(VX,VY, EToV, Lam1, Lam2, qhat)
display(A)
# println("-"^30)
display(b)
# println("-"^30)



#Test case 2
x0, y0 = -2.5, -4.8
L1, L2 = 7.6, 5.9
noelms1, noelms2 = 4, 3
qhat = (x,y) -> return -6*x + 2*y - 2
A,b = assembly(VX,VY, EToV, Lam1, Lam2, qhat)
display(A)
display(b)