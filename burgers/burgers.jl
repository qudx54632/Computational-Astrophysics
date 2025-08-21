# 2nd-order finite-volume Burgers solver with MC slope limiter (Julia)
# u_t + u u_x = 0 with periodic or outflow BCs
#
# Port of M. Zingale's Python code (2013-03-26) to Julia.
using Plots, Statistics

# -------------------------
# Grid and boundary handling
# -------------------------
mutable struct Grid1D
    nx::Int
    ng::Int
    xmin::Float64
    xmax::Float64
    bc::Symbol              # :periodic or :outflow
    ilo::Int                # first real cell (1-based)
    ihi::Int                # last real cell  (1-based)
    dx::Float64
    x::Vector{Float64}      # cell centers (nx+2ng)
    u::Vector{Float64}      # conserved variable (nx+2ng)
end

function Grid1D(nx::Int, ng::Int; xmin=0.0, xmax=1.0, bc::Symbol=:outflow)
    @assert ng >= 2 "ng must be at least 2 for the stencils used here."
    ilo = ng + 1
    ihi = ng + nx
    dx  = (xmax - xmin) / nx
    x   = xmin .+ ((0:(nx+2*ng-1)) .- ng .+ 0.5) .* dx
    u   = zeros(nx + 2*ng)
    Grid1D(nx, ng, xmin, xmax, bc, ilo, ihi, dx, x, u)
end

scratch_array(g::Grid1D) = zeros(eltype(g.u), length(g.u))
# every time you call: arr = scratch_array(g), you get a new array of zeros the same size as g.u.

function fill_BCs!(g::Grid1D)
    if g.bc === :periodic
        # left ghost cells from right interior
        g.u[1:g.ilo-1] .= g.u[g.ihi-g.ng+1 : g.ihi]
        # right ghost cells from left interior
        g.u[g.ihi+1:end] .= g.u[g.ilo : g.ilo+g.ng-1]
    elseif g.bc === :outflow
        g.u[1:g.ilo-1]    .= g.u[g.ilo]
        g.u[g.ihi+1:end]  .= g.u[g.ihi]
    else
        error("invalid BC; use :periodic or :outflow")
    end
end

function grid_norm(g::Grid1D, e::AbstractVector)
    @assert length(e) == length(g.u)
    return sqrt(g.dx * sum(abs2, @view e[g.ilo:g.ihi]))
end

# -------------------------
# Simulation
# -------------------------
mutable struct Simulation
    grid::Grid1D
    t::Float64
end

Simulation(g::Grid1D) = Simulation(g, 0.0)

function init_cond!(s::Simulation, type::Symbol=:tophat)
    g = s.grid
    fill!(g.u, 0.0)

    if type === :tophat
        idx = (@. (g.x >= 0.333) & (g.x <= 0.666))
        g.u[idx] .= 1.0

    elseif type === :sine
        g.u .= 1.0
        idx = (@. (g.x >= 0.333) & (g.x <= 0.666))
        g.u[idx] .+= @. 0.5 * sin(2.0*pi*(g.x[idx]-0.333)/0.333)

    elseif type === :rarefaction
        g.u .= 1.0
        g.u[g.x .> 0.5] .= 2.0

    else
        error("unknown initial condition: $type")
    end

    s.t = 0.0
end

timestep(s::Simulation, C::Float64) = begin
    g = s.grid
    umax = maximum(abs.(@view g.u[g.ilo:g.ihi]))
    C * g.dx / umax
end


xmin, xmax = 0.0, 1.0
nx, ng     = 256, 2
tmax       = (xmax - xmin) / 1.0
C          = 0.8

g  = Grid1D(nx, ng; xmin=xmin, xmax=xmax, bc=:periodic)
s  = Simulation(g)
init_cond!(s, :rarefaction)

function states(g::Grid1D, dt::Float64)
    # MC-limited piecewise-linear reconstruction (van Leer 1977; LeVeque 2002)
    u   = g.u
    dx  = g.dx
    ib  = g.ilo - 1
    ie  = g.ihi + 1

    ul = scratch_array(g)
    ur = scratch_array(g)

    @inbounds for i in ib:ie
     # raw differences
        Δp   = u[i+1] - u[i]          # a_{i+1} - a_i
        Δm   = u[i]   - u[i-1]        # a_i - a_{i-1}
        Δc   = 0.5 * (u[i+1] - u[i-1])  # central diff ( (a_{i+1}-a_{i-1})/(2) )

        ζ = Δp * Δm                   # extrema test

        # limited slope (this equals (∂a/∂x)_i * Δx)
        slope = if ζ > 0
            sgn = sign(Δc)
            min(abs(Δc), 2.0 * abs(Δp), 2.0 * abs(Δm)) * sgn
        else
            0.0
        end

        # piecewise-linear interface states with characteristic shift
        ur[i]   = u[i] - 0.5 * (1 + u[i] * dt/dx) * slope
        ul[i+1] = u[i] + 0.5 * (1 - u[i] * dt/dx) * slope
    end

    return ul, ur
end

function riemann(ul::AbstractVector, ur::AbstractVector)
    # Exact Riemann solver for Burgers' equation (shock vs rarefaction)
    S      = 0.5 .* (ul .+ ur)
    ushock = ifelse.(S .> 0.0, ul, ur)
    ushock = ifelse.(S .== 0.0, 0.0, ushock)

    urare  = ifelse.(ur .<= 0.0, ur, 0.0)
    urare .= ifelse.(ul .>= 0.0, ul, urare)

    us = ifelse.(ul .> ur, ushock, urare)
    return 0.5 .* us .* us
end

function conservative_update(g::Grid1D, dt::Float64, flux::AbstractVector)
    unew = scratch_array(g)
    @views unew[g.ilo:g.ihi] .= g.u[g.ilo:g.ihi] .+
                                (dt/g.dx) .* (flux[g.ilo:g.ihi] .- flux[g.ilo+1:g.ihi+1])
    return unew
end

function evolve!(s::Simulation, C::Float64, tmax::Float64)
    s.t = 0.0
    g = s.grid

    while s.t < tmax
        fill_BCs!(g)
        dt = timestep(s, C)
        if s.t + dt > tmax
            dt = tmax - s.t
        end
        ul, ur = states(g, dt)
        flux   = riemann(ul, ur)
        unew   = conservative_update(g, dt, flux)
        g.u .= unew
        s.t += dt
    end
    return nothing
end

xmin, xmax = 0.0, 1.0
nx, ng     = 256, 2
tmax       = (xmax - xmin) / 1.0
C          = 0.8

# ------------- rarefaction -------------
g  = Grid1D(nx, ng; xmin=xmin, xmax=xmax, bc=:outflow)
s  = Simulation(g)
plt = plot()
init_cond!(s, :rarefaction)
uinit = copy(s.grid.u)

for i in 0:9
    tend = (i + 1) * 0.02 * tmax
    s.grid.u .= uinit
    s.t = 0.0

    evolve!(s, C, tend)

    c = 1.0 - (0.1 + i*0.1)
    plot!(g.x[g.ilo:g.ihi], g.u[g.ilo:g.ihi], label="", color=RGB(c,c,c))
end

plot!(g.x[g.ilo:g.ihi], uinit[g.ilo:g.ihi], label="", linestyle=:dot, color=RGB(0.9,0.9,0.9))
plot!(xlabel="\$x\$", ylabel="\$u\$")
savefig("/Users/xiaoquer/Desktop/julia code CAH/burgers/fv-burger-rarefaction-outflow.pdf")

# ------------- sine -------------
g2 = Grid1D(nx, ng; xmin=xmin, xmax=xmax, bc=:periodic)
s2 = Simulation(g2)

plt = plot()
init_cond!(s2, :sine)
uinit = copy(s2.grid.u)

plt = plot()
for i in 0:9
    tend = (i + 1) * 0.02 * tmax
    s.grid.u .= uinit
    s.t = 0.0

    evolve!(s, C, tend)

    c = 1.0 - (0.1 + i*0.1)
    plot!(g.x[g.ilo:g.ihi], g.u[g.ilo:g.ihi], label="", color=RGB(c,c,c))
end

plot!(g2.x[g2.ilo:g2.ihi], uinit[g2.ilo:g2.ihi], label="", linestyle=:dot, color=RGB(0.9,0.9,0.9))
plot!(xlabel="\$x\$", ylabel="\$u\$")
savefig("/Users/xiaoquer/Desktop/julia code CAH/burgers/fv-burger-sine-periodic.pdf")