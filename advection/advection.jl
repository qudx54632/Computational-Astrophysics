using Plots, LinearAlgebra, LaTeXStrings

# minmod functions
function minmod(a, b)
    return abs(a) < abs(b) && a*b > 0 ? a : (abs(b) < abs(a) && a*b > 0 ? b : 0.0)
end

function maxmod(a, b)
    return abs(a) > abs(b) && a*b > 0 ? a : (abs(b) > abs(a) && a*b > 0 ? b : 0.0)
end

# Grid structure
mutable struct Grid1D
    nx::Int
    ng::Int
    xmin::Float64
    xmax::Float64
    dx::Float64
    x::Vector{Float64}
    xl::Vector{Float64}
    xr::Vector{Float64}
    a::Vector{Float64}
    ilo::Int
    ihi::Int
end

function Grid1D(nx, ng, xmin=0.0, xmax=1.0)
    ilo = ng + 1
    ihi = ng + nx
    dx = (xmax - xmin)/nx
    x = xmin .+ ((0:(nx+2*ng-1)) .- ng .+ 0.5) .* dx
    xl = xmin .+ ((0:(nx+2*ng-1)) .- ng) .* dx
    xr = xmin .+ ((0:(nx+2*ng-1)) .- ng .+ 1.0) .* dx
    a = zeros(nx + 2*ng)
    return Grid1D(nx, ng, xmin, xmax, dx, x, xl, xr, a, ilo, ihi)
end

function fill_BCs!(g::Grid1D)
    for n in 0:g.ng-1
        g.a[g.ilo-1-n] = g.a[g.ihi-n]
        g.a[g.ihi+1+n] = g.a[g.ilo+n]
    end
end

function scratch_array(g::Grid1D)
    return zeros(length(g.a))
end

function norm_error(g::Grid1D, e)
    return maximum(abs.(e[g.ilo:g.ihi]))
end

function abs_norm2(g::Grid1D, ginitial)
    all = g.a .- ginitial
    for i in 1:length(all)
        all[i]^2
    end
    return sum(all)
end

# Simulation structure
mutable struct Simulation
    grid::Grid1D
    t::Float64
    u::Float64
    C::Float64
    slope_type::String
end

function Simulation(grid::Grid1D, u::Float64; C::Float64=0.8, slope_type="centered")
    return Simulation(grid, 0.0, u, C, slope_type)
end

function init_cond!(s::Simulation, type="tophat")
    g = s.grid
    if type == "tophat"
        g.a .= 0.0
        for i in eachindex(g.x)
            if g.x[i] >= 1/3 && g.x[i] <= 2/3
                g.a[i] = 1.0
            end
        end
    elseif type == "sine"
        g.a .= sin.(2.0*pi .* g.x ./ (g.xmax - g.xmin))
    elseif type == "gaussian"
        g.a .= (1/6) .* (
            1.0 .+ exp.(-60.0 .* (g.xl .- 0.5).^2) .+
            4.0 .* (1.0 .+ exp.(-60.0 .* (g.x .- 0.5).^2)) .+
            1.0 .+ exp.(-60.0 .* (g.xr .- 0.5).^2))
    end
end

function timestep(s::Simulation)
    return s.C * s.grid.dx / s.u
end

function period(s::Simulation)
    return (s.grid.xmax - s.grid.xmin)/s.u
end

function states(s::Simulation, dt)
    g = s.grid
    slope = scratch_array(g)
    for i in g.ilo-1:g.ihi+1
        Δm = (g.a[i] - g.a[i-1]) / g.dx
        Δp = (g.a[i+1] - g.a[i]) / g.dx
        if s.slope_type == "centered"
            slope[i] = 0.5 * (g.a[i+1] - g.a[i-1]) / g.dx
        elseif s.slope_type == "minmod"
            slope[i] = minmod(Δm, Δp)
        elseif s.slope_type == "MC"
            slope[i] = minmod(minmod(2Δm, 2Δp), 0.5 * (g.a[i+1] - g.a[i-1]) / g.dx)
        elseif s.slope_type == "superbee"
            A = minmod(Δp, 2Δm)
            B = minmod(Δm, 2Δp)
            slope[i] = maxmod(A, B)
        else
            slope[i] = 0.0  # godunov (piecewise constant)
        end
    end
    al = scratch_array(g)
    ar = scratch_array(g)
    for i in g.ilo:g.ihi+1
        al[i] = g.a[i-1] + 0.5 * g.dx * (1.0 - s.u*dt/g.dx) * slope[i-1]
        ar[i] = g.a[i] - 0.5 * g.dx * (1.0 + s.u*dt/g.dx) * slope[i]
    end
    return al, ar
end

function riemann(s::Simulation, al, ar)
    return s.u > 0 ? s.u .* al : s.u .* ar
end

function update(s::Simulation, dt, flux)
    g = s.grid
    anew = scratch_array(g)
    for i in g.ilo:g.ihi
        anew[i] = g.a[i] + dt/g.dx * (flux[i] - flux[i+1])
    end
    return anew
end

function evolve!(s::Simulation; num_periods=1)
    tmax = num_periods * period(s)
    g = s.grid
    while s.t < tmax
        fill_BCs!(g)
        dt = timestep(s)
        dt = min(dt, tmax - s.t)
        al, ar = states(s, dt)
        flux = riemann(s, al, ar)
        anew = update(s, dt, flux)
        g.a .= anew
        s.t += dt
    end
end

# ------------------------------
# Compare limiting and no-limiting
# ------------------------------
xmin, xmax = 0.0, 1.0
nx, ng = 64, 2
u = 1.0

g = Grid1D(nx, ng, xmin, xmax)

s = Simulation(g, u, C=0.7, slope_type="centered")
init_cond!(s, "tophat")
ainit = copy(g.a)
evolve!(s, num_periods=5)

plot(g.x[g.ilo:g.ihi], ainit[g.ilo:g.ihi], linestyle=:dot, label="exact")
plot!(g.x[g.ilo:g.ihi], g.a[g.ilo:g.ihi], label="unlimited")

s = Simulation(g, u, C=0.7, slope_type="minmod")
init_cond!(s, "tophat")
evolve!(s, num_periods=5)

plot!(g.x[g.ilo:g.ihi], g.a[g.ilo:g.ihi], label="minmod limiter")
xlabel!("x")
ylabel!("a")
plot!(legend=:best)
savefig("/Users/xiaoquer/Desktop/julia code CAH/advection/fv-advect.pdf")

# ------------------------------
# Convergence test
# ------------------------------
problem = "gaussian"
Nvals = [32, 64, 128, 256, 512]
err_god, err_nolim, err_lim, err_lim2 = Float64[], Float64[], Float64[], Float64[]

for nx in Nvals
    gg = Grid1D(nx, ng, xmin, xmax)
    sg = Simulation(gg, u, C=0.8, slope_type="godunov")
    init_cond!(sg, problem)
    local ainit = copy(gg.a)
    evolve!(sg, num_periods=5)
    push!(err_god, norm(gg.a[gg.ilo:gg.ihi] .- ainit[gg.ilo:gg.ihi]) / sqrt(gg.ihi - gg.ilo + 1))

    gu = Grid1D(nx, ng, xmin, xmax)
    su = Simulation(gu, u, C=0.8, slope_type="centered")
    init_cond!(su, problem)
    local ainit = copy(gu.a)
    evolve!(su, num_periods=5)
    push!(err_nolim, norm(gu.a[gu.ilo:gu.ihi] .- ainit[gu.ilo:gu.ihi]) / sqrt(gu.ihi - gu.ilo + 1))

    gl = Grid1D(nx, ng, xmin, xmax)
    sl = Simulation(gl, u, C=0.8, slope_type="MC")
    init_cond!(sl, problem)
    local ainit = copy(gl.a)
    evolve!(sl, num_periods=5)
    push!(err_lim, norm(gl.a[gl.ilo:gl.ihi] .- ainit[gl.ilo:gl.ihi]) / sqrt(gl.ihi - gl.ilo + 1))

    gl2 = Grid1D(nx, ng, xmin, xmax)
    sl2 = Simulation(gl2, u, C=0.8, slope_type="minmod")
    init_cond!(sl2, problem)
    local ainit = copy(gl2.a)
    evolve!(sl2, num_periods=5)
    push!(err_lim2, norm(gl2.a[gl2.ilo:gl2.ihi] .- ainit[gl2.ilo:gl2.ihi]) / sqrt(gl2.ihi - gl2.ilo + 1))
end

Nf = Float64.(Nvals)

scatter(Nf, err_god, label="Godunov", color=:blue, legend=:bottomleft)
scatter!(Nf, err_nolim, label="unlimited center", color=:orange)
scatter!(Nf, err_lim, label="MC", color=:green)
scatter!(Nf, err_lim2, label="minmod", color=:red)

plot!(Nf, err_god[end] .* (Nf[end] ./ Nf), label="O(\$Δx\$)", color=:black)
plot!(Nf, err_nolim[end] .* (Nf[end] ./ Nf).^2, label="O(\$Δx^2\$)", color=:gray)

plot!(xaxis=:log, yaxis=:log)
xlabel!("\$N\$")
ylabel!(L"\| a^{\mathrm{final}} - a^{\mathrm{init}} \|_2")
savefig("/Users/xiaoquer/Desktop/julia code CAH/advection/plm-converge.pdf")

# ------------------------------
# Plot different limiters
# ------------------------------
nx = 512
g = Grid1D(nx, ng, xmin, xmax)

for p in ["gaussian", "tophat"]
    pl = plot(layout=(2,3), size=(1000,700))

    for (i, slope) in enumerate(["godunov", "centered", "minmod", "MC", "superbee"])
        local s = Simulation(g, u, C=0.8, slope_type=slope)
        init_cond!(s, p)
        local ainit = copy(g.a)
        evolve!(s, num_periods=5)

        plot!(pl[i], g.x[g.ilo:g.ihi], ainit[g.ilo:g.ihi], linestyle=:dot, label="initial")
        plot!(pl[i], g.x[g.ilo:g.ihi], g.a[g.ilo:g.ihi], label="final")
        title!(pl[i], slope)
    end

    plot!(pl[6], legend=:outertopright)
    savefig(pl, "fv-$(p)-limiters.pdf")
end