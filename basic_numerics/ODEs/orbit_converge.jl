using Plots, LaTeXStrings

include("/Users/xiaoquer/Desktop/julia code CAH/basic_numerics/ODEs/orbit.jl") 

function test_orbit_convergence()
    o = Orbit(1.0, 0.0)
    P = kepler_period(o)

    tstep = Float64[]
    err_Euler = Float64[]
    err_EC = Float64[]
    err_RK2 = Float64[]
    err_RK4 = Float64[]

    dt = 0.05

    for i in 1:5
        H_Euler = int_Euler(o, dt, P)
        H_EC = int_Euler_Cromer(o, dt, P)
        H_RK2 = int_RK2(o, dt, P)
        H_RK4 = int_RK4(o, dt, P)

        e1 = abs(final_R(H_Euler) - o.a)
        e2 = abs(final_R(H_EC) - o.a)
        e3 = abs(final_R(H_RK2) - o.a)
        e4 = abs(final_R(H_RK4) - o.a)

        println(dt, "\t", e1, "\t", e2, "\t", e3, "\t", e4)

        push!(tstep, dt)
        push!(err_Euler, e1)
        push!(err_EC, e2)
        push!(err_RK2, e3)
        push!(err_RK4, e4)

        dt /= 2
    end

    return tstep, err_Euler, err_EC, err_RK2, err_RK4
end

# Then in the script:
tstep, err_Euler, err_EC, err_RK2, err_RK4 = test_orbit_convergence()

# Convert to arrays for broadcasting
t = tstep
ref1(x, e0) = e0 * (x / t[1]).^-1
ref2(x, e0) = e0 * (x / t[1]).^-2
ref4(x, e0) = e0 * (x / t[1]).^-4

# Plot
p = plot(xscale=:log10, yscale=:log10,
         xlabel=L"\tau", ylabel="absolute error in radius after one period",
         legend=:bottomright, ylim=(1e-10, 10), lw=2)

scatter!(p, tstep, err_Euler, label="Euler", color=:black)
plot!(p, tstep, ref1.(tstep, err_Euler[1]), color=:black, lw=1, label=nothing)

scatter!(p, tstep, err_EC, label="Euler-Cromer", color=:red)
plot!(p, tstep, ref1.(tstep, err_EC[1]), color=:red, lw=1, label=nothing)

scatter!(p, tstep, err_RK2, label="R-K 2", color=:blue)
plot!(p, tstep, ref2.(tstep, err_RK2[1]), color=:blue, lw=1, label=nothing)

scatter!(p, tstep, err_RK4, label="R-K 4", color=:green)
plot!(p, tstep, ref4.(tstep, err_RK4[1]), color=:green, lw=1, label=nothing)

savefig(p, "orbit-converge.png")