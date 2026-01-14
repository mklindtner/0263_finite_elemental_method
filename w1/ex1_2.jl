#Author: Mikkel Koefoed Lindtner
using LinearAlgebra
using Plots


# INPUT PARAMETERS
# L : Domain length
# c : Left boundary condition
# d : Right boundary condition
# x : 1D mesh vector x(1:{M})
L = 2
c = 1
d = exp(2)
x = [0.0, 0.2, 0.4, 0.6, 0.7, 0.9, 1.4, 1.5, 1.8, 1.9, 2.0]


function BVP1D_a(L,c,d,x)
    #Hardcoded K-values for differential equation u'' -u = 0     
    M = length(x)
    h = diff(x)

    #Global Assembly
        #Assemble A (upper triangle)
        #Assemble b. (Algorithm 1)
    A = zeros(M,M)
    b = zeros(M,1)

    for i = 1:1:M-1
        k1_1 = 1/h[i] + h[i]/3
        k1_2 = -1/h[i] + h[i]/6
        k2_2 = 1/h[i] + h[i]/3
        
        A[i,i] = A[i,i] + k1_1
        A[i,i+1] = k1_2
        A[i+1,i+1] = k2_2
        
        #Use full matrix alternative
        #k2_1 = k1_2
        #A[i+1,i] = k2_1
    end
    #A[M, M-1] = 0 #I dont understand this?
    
    # IMPOSE BOUNDARY CONDITIONS
    # (Algorithm 2)
    #<INSERT YOUR CODE HERE>    
    b[1] = c
    b[2] = b[2]-A[1,2]*c
    A[1,1] = 1
    A[1,2] = 0
    b[M] = d
    b[M-1] = b[M-1] - A[M-1,M]*d
    A[M,M] = 1
    A[M-1,M] = 0


    #visually checking A matrix is correct
    open("w1/matrix_A.txt", "w") do io
        Base.print_matrix(io, A)
    end


    # SOLVE SYSTEM
    # Solve using the Cholesky factorization of A to solve A*u=b
    # u = A \ b
    chol = cholesky(Symmetric(A, :U), check=false)
    if issuccess(chol)
        u = chol \ b
    else
        println("A is not positive definite")
        return
    end

    # open("w1/matrix_A.txt", "w") do io
    #     Base.print_matrix(io, A)
    # end
    
    # %% OUTPUT
    p = plot(x, u, label="FEM Solution", title="BVP 1D Solution", xlabel="x", ylabel="u")
    plot!(p, x, exp.(x), label="exp(x)", linestyle=:dash)
    savefig(p, "w1/w1_2_a.png")
end

#Exercise 1.2.a
# BVP1D_a(L, c,d,x)

function BVP1D_b(L,c,d,M)
    x = range(0,L, length=M) #equidistant
    h = diff(x) 
    println(x)
    println(M)
    println(h, "legnth: $(length(h))")
    #Global Assembly
        #Assemble A (upper triangle)
        #Assemble b. (Algorithm 1)
    A = zeros(M,M)
    b = zeros(M,1)

    for i = 1:1:M-1
        k1_1 = 1/h[i] + h[i]/3
        k1_2 = -1/h[i] + h[i]/6
        k2_2 = 1/h[i] + h[i]/3
        
        A[i,i] = A[i,i] + k1_1
        A[i,i+1] = k1_2
        A[i+1,i+1] = k2_2
        
        #Use full matrix alternative
        #k2_1 = k1_2
        #A[i+1,i] = k2_1
    end
    #A[M, M-1] = 0 #I dont understand this?
    
    # IMPOSE BOUNDARY CONDITIONS
    # (Algorithm 2)
    #<INSERT YOUR CODE HERE>    
    b[1] = c
    b[2] = b[2]-A[1,2]*c
    A[1,1] = 1
    A[1,2] = 0
    b[M] = d
    b[M-1] = b[M-1] - A[M-1,M]*d
    A[M,M] = 1
    A[M-1,M] = 0


    #visually checking A matrix is correct
    open("w1/matrix_A.txt", "w") do io
        Base.print_matrix(io, A)
    end


    # SOLVE SYSTEM
    # Solve using the Cholesky factorization of A to solve A*u=b
    # u = A \ b
    chol = cholesky(Symmetric(A, :U), check=false)
    if issuccess(chol)
        u = chol \ b
    else
        println("A is not positive definite")
        return
    end

    # open("w1/matrix_A.txt", "w") do io
    #     Base.print_matrix(io, A)
    # end
    
    # %% OUTPUT
    p = plot(x, u, label="FEM Solution", title="BVP 1D Solution", xlabel="x", ylabel="u")
    plot!(p, x, exp.(x), label="exp(x)", linestyle=:dash)
    savefig(p, "w1/w1_2_b.png")    
    return x, u
end

#Exercise 1.2.b
M = 11
# x,uhat = BVP1D_b(L,c,d,M)


#Exercise 1.2.D
function assembly(M,h)
    A = zeros(M,M)
    b = zeros(M,1)
    for i = 1:1:M-1
        k1_1 = 1/h[i] + h[i]/3
        k1_2 = -1/h[i] + h[i]/6
        k2_2 = 1/h[i] + h[i]/3
        
        A[i,i] = A[i,i] + k1_1
        A[i,i+1] = k1_2
        A[i+1,i+1] = k2_2
    end
    return A,b
end


function boundary_conditions(A,b,M)
    b[1] = c
    b[2] = b[2]-A[1,2]*c
    A[1,1] = 1
    A[1,2] = 0
    b[M] = d
    b[M-1] = b[M-1] - A[M-1,M]*d
    A[M,M] = 1
    A[M-1,M] = 0
    return A,b 
end

function conv_rate(iters, L)
    M_values = Float64[]
    errors = Float64[]
    h_sq_values = Float64[]
    

    for M in range(3,iters)
        x = range(0,L, length=M)
        h = diff(x)
        
        A,b = assembly(M,h)
        A,b = boundary_conditions(A,b,M)

        chol = cholesky(Symmetric(A, :U), check=false)  
        if issuccess(chol)
            uhat = chol \ b
        else
            println("A is not positive definite")
            return
        end

        error_term = maximum(abs.(exp.(x) - uhat))
        push!(M_values, M)
        push!(errors, error_term)
        push!(h_sq_values, h[1] * h[1])
    end

    p = plot(M_values, errors, 
             xaxis=:log, yaxis=:log, 
             marker=:circle,
             title="convergence rate",
             xlabel="M", 
             ylabel="Max Error",
             label="FEM Error")
    
    plot!(p, M_values, 1 .* h_sq_values, 
        linestyle=:dash, 
        label="2 * h^2 Reference",
        color=:red)

    savefig(p, "w1/1_2_d_convergence_plot.png")
    display(p)

    #Get the slop between the first two points
    s_start = (log(errors[2]) - log(errors[1])) / (log(M_values[2]) - log(M_values[1]))    
    #Get the slop between last two points (we expect approx -2.0)
    s_end   = (log(errors[end]) - log(errors[end-1])) / (log(M_values[end]) - log(M_values[end-1]))

    println("Convergence Rate (First 2 points): ", s_start)
    println("Convergence Rate (Last 2 points):  ", s_end)
end


conv_rate(100, L)