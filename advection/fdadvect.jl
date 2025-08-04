using Plots

# Define FDGrid type
mutable struct FDGrid
    nx::Int
    ng::Int
    xmin::Float64
    xmax::Float64
    ilo::Int
    ihi::Int
    dx::Float64
    x::Vector{Float64}
    a::Vector{Float64}
    ainit::Vector{Float64}
end

function FDGrid(nx::Int, ng::Int, xmin::Float64=0.0, xmax::Float64=1.0)
    ilo = ng + 1
    ihi = ng + nx
    dx = (xmax - xmin) / (nx - 1)
    x = xmin .+ ((0:(nx+2*ng-1)) .- ng) .* dx
    a = zeros(nx + 2*ng)
    ainit = copy(a)
    return FDGrid(nx, ng, xmin, xmax, ilo, ihi, dx, x, a, ainit)
end

function scratch_array(g::FDGrid)
    return zeros(length(g.a))
end

function fill_BCs!(g::FDGrid)
    g.a[g.ilo-1] = g.a[g.ihi-1]
    g.a[g.ihi+1] = g.a[g.ilo+1]
end

function solve_advection!(g::FDGrid, u::Float64, C::Float64; method::String="upwind", tmax_factor::Float64=1.0)
    dt = C * g.dx / u
    t = 0.0
    tmax = tmax_factor * (g.xmax - g.xmin) / u

    g.a .= 0.0
    for i in eachindex(g.x)
        if g.x[i] >= 1/3 && g.x[i] <= 2/3
            g.a[i] = 1.0
        end
    end
    g.ainit .= g.a

    anew = scratch_array(g)

    while t < tmax
        fill_BCs!(g)
        for i in g.ilo:g.ihi
            if method == "upwind"
                anew[i] = g.a[i] - C * (g.a[i] - g.a[i-1])
            elseif method == "FTCS"
                anew[i] = g.a[i] - 0.5 * C * (g.a[i+1] - g.a[i-1])
            else
                error("Invalid method")
            end
        end
        g.a .= anew
        t += dt
    end
end

# Create grid
nx = 65
ng = 1
g = FDGrid(nx, ng)

Clist = [0.01, 0.1, 0.5, 0.9]
u = 1.0

# Upwind plot
p1 = plot(title="Upwind", xlabel="x", ylabel="a", legend=:best)
for (n, C) in enumerate(Clist)
    solve_advection!(g, u, C, method="upwind")
    if n == 1
        plot!(p1, g.x[g.ilo:g.ihi], g.ainit[g.ilo:g.ihi], label="exact", linestyle=:dot)
    end
    plot!(p1, g.x[g.ilo:g.ihi], g.a[g.ilo:g.ihi], label="C = $C")
end

savefig(p1, "/Users/xiaoquer/Desktop/julia code CAH/advection/fdadvect-upwind.pdf")

# FTCS plots
for C in Clist
    solve_advection!(g, u, C, method="FTCS", tmax_factor=0.1)
    p = plot(g.x[g.ilo:g.ihi], g.ainit[g.ilo:g.ihi], label="exact", linestyle=:dot,
             xlabel="x", ylabel="a", legend=:best, title="FTCS C = $C")
    plot!(p, g.x[g.ilo:g.ihi], g.a[g.ilo:g.ihi], label="C = $C")
    savefig(p, "fdadvect-FTCS-C$C.pdf")
end