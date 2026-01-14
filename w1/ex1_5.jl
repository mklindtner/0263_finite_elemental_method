
using Plots

#Exercise 1.5.D
#Testing analytical
function ex1_5_d()

    #Initialize   
    Psi = 1.0
    eps = [1.0, 0.01, 0.0001]
    x_vals = 5000
    x = range(0,1,length=x_vals)[2:x_vals-1]

    #analytical solution
    u1 = u_anal(x, eps[1], Psi)
    u2 = u_anal(x, eps[2], Psi)
    # u3 = u_anal(x, eps[3], Psi)

    #Convert x to float64
    x = Float64.(x)

    #Plots
    p = plot(x, u1, 
                marker=:circle,
                title="u anal",
                xlabel="x", 
                ylabel="u(x)",
                label="eps=1")

    plot!(p, x, u2, 
        linestyle=:dash,
        label="eps=0.01",
        color=:red)
    
    # plot!(p, x, u3,
    #     linestyle=:dot,
    #     label="eps=0.0001",
    #     color=:blue
    #     )

    display(p)
    savefig(p, "w1/1_5_d_anal_plot.png")

end


function u_anal(x, eps, Psi)
    x = BigFloat.(collect(x))
    eps = BigFloat(eps)
    Psi = BigFloat(Psi)

    numerator = (x,eps) -> 1 .+ (exp(Psi/eps) - 1 ) .* x .- exp.(x .* Psi / eps) 
    denominator = eps -> exp(Psi/eps) - 1
    u = (x,eps) -> 1/Psi .* (numerator(x,eps) ./ denominator(eps) )
    u = Float64.(u(x,eps))
    return u
end 

# ex1_5_d()


#Exercise 1.5.e
function assembly(M,h,eps, Psi,c)
    A = zeros(M,M)
    b = zeros(M,1)


    for i = 1:1:M-1
        k1_1 = eps/h + 1/2* Psi
        k1_2 = -eps/h + 1/2*Psi
        k2_1 = -eps/h - 1/2*Psi
        k2_2 = eps/h - 1/2*Psi

        A[i,i] = A[i,i] + k1_1
        A[i,i+1] = k1_2
        A[i+1,i+1] = k2_2
        A[i+1, i] = k2_1

        b[i] = h - k2_1*c
    end
    return A,b
end


function boundary_conditions(A,b,M,c,d)
    b[1] = c
    b[2] = b[2]-A[2,1]*c
    
    A[1,1] = 1.0
    A[1,2] = 0.0

    b[M] = d
    b[M-1] = b[M-1] - A[M-1,M]*d

    A[M,M] = 1.0
    A[M-1,M] = 0.0
    A[M, M-1] = 0.0
    return A,b 
end

function conv_rate1_5_e(iters, L,eps, Psi)
    M_values = Float64[]
    errors = Float64[]
    h_sq_values = Float64[]
    c = 0 #start boundary
    d = 0 #end boundary

    for M in range(3,iters)
        x = range(0,L, length=M)
        h = x[2]-x[1] #assumes equidistance between points
        
        A,b = assembly(M,h,eps, Psi,c)
        A,b = boundary_conditions(A,b,M,c,d)

        uhat = A \ b

        error_term = maximum(abs.(u_anal(x,eps, Psi) - uhat))
        push!(M_values, M)
        push!(errors, error_term)
        push!(h_sq_values, h[1] * h[1])
        
        #Plot best mesh
        if M == last(iters)
            q = plot(x, uhat,
                linestyle=:dash, 
                label="approx solution",
                color=:blue)
            savefig(q, "w1/1_5_e_approxsol_plot_eps_$(eps).png")
        end
    end

    p = plot(M_values, errors, 
             xaxis=:log, yaxis=:log, 
             marker=:circle,
             title="convergence rate",
             xlabel="M", 
             ylabel="Max Error",
             label="FEM Error")
    
    plot!(p, M_values, 20000 .* h_sq_values, 
        linestyle=:dash, 
        label="2000 * h^2 Reference",
        color=:red)

    # q = plot(M_values, uhat_vals,
    #     linestyle=:dash, 
    #     label="approx solution",
    #     color=:blue)
        

    savefig(p, "w1/1_5_e_convergence_plot_eps_$(eps).png")

    # display(p)

    
    #Get the slop between the first two points
    s_start = (log(errors[2]) - log(errors[1])) / (log(M_values[2]) - log(M_values[1]))    
    #Get the slop between last two points (we expect approx -2.0)
    s_end   = (log(errors[end]) - log(errors[end-1])) / (log(M_values[end]) - log(M_values[end-1]))

    println("Convergence Rate (First 2 points): ", s_start)
    println("Convergence Rate (Last 2 points):  ", s_end)
end


Psi = 1.0
conv_rate1_5_e(1000, 1, 1.0, Psi)
conv_rate1_5_e(1000, 1, 0.01, Psi)
conv_rate1_5_e(1000, 1, 0.0001, Psi)

