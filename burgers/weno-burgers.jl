#= 
This Julia translation covers:
- WENO and WENO-M reconstruction routines
- Gaussian exact solution via method of characteristics
- Burgers' equation solver using RK4
- Time evolution and convergence tests
=#

using LinearAlgebra, Plots, LaTeXStrings
using Roots: find_zero, Brent

include("burgers.jl")
include("weno-coefficients.jl")

function weno(order, q)
    C = C_all[order]         # Linear weights C_k
    a = a_all[order]         # Coefficients for q_stencils[k] = sum a_kl * q_values
    sigma = sigma_all[order] # Smoothness indicator coefficients σ_klm
    ϵ = 1e-16                # Small number to prevent division by zero

    qL = zeros(length(q))              # Final reconstructed values
    beta = zeros(order, length(q))     # Smoothness indicators β_k
    w = zeros(order, length(q))        # Nonlinear weights ω_k

    for i in order+1 : length(q) - order
        q_stencils = zeros(order)      # Candidate polynomials q^{(k)}
        alpha = zeros(order)           # Unnormalized nonlinear weights α_k

        for k in 1:order
            # Compute smoothness indicator β_k
            for l in 1:order
                for m in 1:l
                    beta[k, i] += sigma[k, l, m] * q[i + k - l] * q[i + k - m]
                end
            end

            # Compute α_k = C_k / (ε + β_k)^2
            alpha[k] = C[k] / (ϵ + beta[k, i]^2)

            # Compute q_stencils[k] = ∑_l a[k, l] * q[i + k - l]
            for l in 1:order
                q_stencils[k] += a[k, l] * q[i + k - l]
            end
        end

        # Normalize α to get nonlinear weights ω_k
        alpha_sum = sum(alpha)
        for k in 1:order
            w[k, i] = alpha[k] / alpha_sum
        end

        # Final WENO reconstruction at i
        for k in 1:order
            qL[i] += w[k, i] * q_stencils[k]
        end
    end

    return qL
end

function weno_M(order, q)
    C = C_all[order]
    a = a_all[order]
    sigma = sigma_all[order]
    ϵ = 1e-16

    nq = length(q)
    qL = zeros(nq)
    beta = zeros(order, nq)
    w = zeros(order, nq)

    for i in order+1 : length(q) - order
        q_stencils = zeros(order)
        alpha_JS = zeros(order)

        for k in 1:order
            # Compute smoothness indicator beta[k, i]
            for l in 1:order
                for m in 1:l
                    beta[k, i] += sigma[k, l, m] * q[i + k - l] * q[i + k - m]
                end
            end

            # Compute alpha_JS[k]
            alpha_JS[k] = C[k] / (ϵ + beta[k, i]^2)

            # Compute q_stencils[k]
            for l in 1:order
                q_stencils[k] += a[k, l] * q[i + k - l]
            end
        end

        # Compute sum of alpha_JS
        alpha_JS_sum = 0.0
        for k in 1:order
            alpha_JS_sum += alpha_JS[k]
        end

        # Compute w_JS = alpha_JS / sum(alpha_JS)
        w_JS = zeros(order)
        for k in 1:order
            w_JS[k] = alpha_JS[k] / alpha_JS_sum
        end

        # Compute mapped alpha using WENO-M formula
        alpha = zeros(order)
        for k in 1:order
            numerator = w_JS[k] * (C[k] + C[k]^2 - 3 * C[k] * w_JS[k] + w_JS[k]^2)
            denominator = C[k]^2 + w_JS[k] * (1 - 2 * C[k])
            alpha[k] = numerator / denominator
        end

        # Normalize to get final weights
        alpha_sum = 0.0
        for k in 1:order
            alpha_sum += alpha[k]
        end
        for k in 1:order
            w[k, i] = alpha[k] / alpha_sum
        end

        # Reconstruct qL[i]
        for k in 1:order
            qL[i] += w[k, i] * q_stencils[k]
        end
    end

    return qL
end

mutable struct WENOSimulation
    grid::Grid1D
    C::Float64
    weno_order::Int
    t::Float64
end

function WENOSimulation(grid::Grid1D, C=0.5, weno_order=3)
    WENOSimulation(grid, C, weno_order, 0.0)
end

function init_cond!(sim::WENOSimulation, type::String)
    if type == "smooth_sine"
        sim.grid.u .= sin.(2π .* sim.grid.x)
    elseif type == "gaussian"
        sim.grid.u .= 1.0 .+ exp.(-60.0 .* (sim.grid.x .- 0.5).^2)
    else
        dummy = Simulation(sim.grid)
        init_cond!(dummy, type)
    end
end

function burgers_flux(q)
    return 0.5 .* q.^2
end

function rk_substep(sim::WENOSimulation)
    g = sim.grid
    fill_BCs!(g)
    f = burgers_flux(g.u)
    α = maximum(abs.(g.u))

    fp = (f .+ α .* g.u) ./ 2
    fm = (f .- α .* g.u) ./ 2

    fpr = scratch_array(g)
    fml = scratch_array(g)
    flux = scratch_array(g)

    # Match Python: fpr[2:] = weno(fp[1:end-1])
    fpr[2:end] .= weno(sim.weno_order, fp[1:end-1])

    # Match Python: fml[end:-1:1] = weno(fm[end:-1:1])
    fml .= reverse(weno(sim.weno_order, reverse(fm)))

    # flux[2:end-1] = fpr[2:end-1] + fml[2:end-1]
    flux[2:end-1] .= fpr[2:end-1] .+ fml[2:end-1]

    rhs = scratch_array(g)
    rhs[2:end-1] .= (flux[2:end-1] .- flux[3:end]) ./ g.dx

    return rhs
end

timestep(s::WENOSimulation, C::Float64) = begin
    g = s.grid
    umax = maximum(abs.(@view g.u[g.ilo:g.ihi]))
    C * g.dx / umax
end

function evolve!(sim::WENOSimulation, tmax::Float64)
    g = sim.grid
    t = sim.t

    while t < tmax
        fill_BCs!(g)
        dt = timestep(sim, sim.C)
        dt = min(dt, tmax - t)

        u0 = copy(g.u)
        k1 = dt .* rk_substep(sim)
        g.u .= u0 .+ 0.5 .* k1
        k2 = dt .* rk_substep(sim)
        g.u .= u0 .+ 0.5 .* k2
        k3 = dt .* rk_substep(sim)
        g.u .= u0 .+ k3
        k4 = dt .* rk_substep(sim)
        g.u .= u0 .+ (k1 .+ 2 .* (k2 .+ k3) .+ k4) ./ 6

        t += dt
    end
    sim.t = t
end

# Main script to run the WENO simulation on Burgers' equation with sine initial condition and periodic boundary conditions
xmin, xmax = 0.0, 1.0
nx = 256
order = 3
ng = order + 1
g = Grid1D(nx, ng; xmin=xmin, xmax=xmax, bc=:periodic)
tmax = xmax - xmin
C = 0.5

sim = WENOSimulation(g, C, order)
init_cond!(sim, "sine")

plt1 = plot()
plot(sim.grid.x[sim.grid.ilo:sim.grid.ihi], sim.grid.u[sim.grid.ilo:sim.grid.ihi], ls=:dot, label="t = 0", color=:gray)

for i in 1:10
    local tend = i * 0.02 * tmax
    evolve!(sim, tend)
    plot!(sim.grid.x[sim.grid.ilo:sim.grid.ihi], sim.grid.u[sim.grid.ilo:sim.grid.ihi], label="t = $(round(tend, digits=2))")
end

xlabel!("x")
ylabel!("u")
savefig("weno-burger-sine.pdf")


 # Compare the WENO and "standard" (from burgers.py) results at low res
nx = 64
tend = 0.2

g_hi = Grid1D(512, ng; xmin, xmax, bc=:periodic)
s_hi = WENOSimulation(g_hi, C, order)
init_cond!(s_hi, "sine")
evolve!(s_hi, tend)
plt2 = plot()
plot!(g_hi.x[g_hi.ilo:g_hi.ihi], g_hi.u[g_hi.ilo:g_hi.ihi], ls=:dot, label="High resolution")

gW3 = Grid1D(nx, 4; xmin, xmax, bc=:periodic)
sW3 = WENOSimulation(gW3, C, 3)
init_cond!(sW3, "sine")
evolve!(sW3, tend)

gW5 = Grid1D(nx, 6; xmin, xmax, bc=:periodic)
sW5 = WENOSimulation(gW5, C, 5)
init_cond!(sW5, "sine")
evolve!(sW5, tend)

g_plm = Grid1D(nx, ng; xmin, xmax, bc=:periodic)
s_plm = Simulation(g_plm)
init_cond!(s_plm, "sine")
evolve!(s_plm, C, tend)

scatter!(g_plm.x[g_plm.ilo:g_plm.ihi], g_plm.u[g_plm.ilo:g_plm.ihi], label="PLM, MC", marker=:diamond)
scatter!(sW3.grid.x[sW3.grid.ilo:sW3.grid.ihi], sW3.grid.u[sW3.grid.ilo:sW3.grid.ihi], label="WENO, r=3", marker=:circle)
scatter!(sW5.grid.x[sW5.grid.ilo:sW5.grid.ihi], sW5.grid.u[sW5.grid.ilo:sW5.grid.ihi], label="WENO, r=5", marker=:utriangle)
xlabel!("x")
ylabel!("u")
xlims!(0.5, 0.9)
savefig("weno-vs-plm-burger.pdf")

# Set up and run rarefaction simulation
xmin, xmax = 0.0, 1.0
nx = 256
order = 3
ng = order + 1
g = Grid1D(nx, ng; xmin=xmin, xmax=xmax, bc=:outflow)
C = 0.5
tmax = (xmax - xmin) / 1.0

sim = WENOSimulation(g, C, order)
init_cond!(sim, "rarefaction")

plt = plot()
plot(sim.grid.x[sim.grid.ilo:sim.grid.ihi], sim.grid.u[sim.grid.ilo:sim.grid.ihi], ls=:dot, label="t = 0", color=:gray)
for i in 1:10
    local tend = i * 0.02 * tmax
    evolve!(sim, tend)
    plot!(sim.grid.x[sim.grid.ilo:sim.grid.ihi], sim.grid.u[sim.grid.ilo:sim.grid.ihi], label="t = $(round(tend, digits=2))")
end

xlabel!("x")
ylabel!("u")
savefig("weno-burger-rarefaction.pdf")

# # Convergence parameters
# problem = "gaussian"
# xmin, xmax = 0.0, 1.0
# tmax = 0.05
# orders = [3, 4]
# N_list = [64, 81, 108, 128, 144, 192, 256]
# C = 0.5

# # Store errors
# errs = []

# # Gaussian exact solution via characteristic
# function burgers_gaussian_exact(x::Float64, t::Float64)
#     f(x) = 1.0 + exp(-60.0 * (x - 0.5)^2)
#     residual(x0) = x0 + f(x0) * t - x
#     x0 = find_zero(residual, (-2.0, 2.0), Brent())
#     return f(x0)
# end

# function burgers_gaussian_exact_array(x::Vector{Float64}, t::Float64)
#     return [burgers_gaussian_exact(xi, t) for xi in x]
# end

# for order in orders
#     local ng = order + 1
#     push!(errs, [])
#     for nx in N_list
#         local g = Grid1D(nx, ng; xmin=xmin, xmax=xmax, bc=:periodic)
#         local sim = WENOSimulation(g, C, order)
#         init_cond!(sim, "gaussian")
#         evolve!(sim, tmax)

#         u_exact = burgers_gaussian_exact_array(g.x, tmax)
         
#         error = sqrt(sum(abs2, sim.grid.u .- u_exact) / length(u_exact)) # L2 norm
#         push!(errs[end], error)
#     end
# end


# # # Plotting convergence
# plt = plot(xscale=:log10, yscale=:log10, legend=:topleft)
# # colors = [:blue, :red]
# # for (i, order) in enumerate(orders)
# #     scatter!(N_list, errs[i], label="WENO, r=$order", color=colors[i])
# # end

# # # Reference convergence rates
# N_float = Float64.(N_list)
# plot!(N_float, errs[1][end-1] * (N_float[end-1] ./ N_float).^5,
#       ls=:dash, color=:blue, label=L"\mathcal{O}(\Delta x^5)")
# plot!(N_float, errs[2][end-2] * (N_float[end-2] ./ N_float).^7,
#       ls=:dash, color=:red, label=L"\mathcal{O}(\Delta x^7)")

# ylims!(1e-7, 1e-3)
# xlabel!("N")
# # ylabel!(L"\| u^{\mathrm{final}} - u^{\mathrm{exact}} \|_2")
# # title!("Convergence of Burgers', Gaussian, RK4")
# savefig("weno-converge-burgers.pdf")