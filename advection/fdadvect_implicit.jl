using LinearAlgebra, Plots

# Define FDGrid type
mutable struct FDGrid
    nx::Int
    xmin::Float64
    xmax::Float64
    ilo::Int
    ihi::Int
    dx::Float64
    x::Vector{Float64}
    a::Vector{Float64}
    ainit::Vector{Float64}
end

function FDGrid(nx::Int, xmin::Float64=0.0, xmax::Float64=1.0)
    ilo = 2
    ihi = nx
    dx = (xmax - xmin) / (nx - 1)
    x = xmin .+ (0:nx-1) * dx
    a = zeros(nx)
    ainit = copy(a)
    return FDGrid(nx, xmin, xmax, ilo, ihi, dx, x, a, ainit)
end

function fill_BCs!(g::FDGrid)
    g.a[1] = g.a[g.ihi]  # periodic: a[0] = a[N-1], but Julia is 1-based
end

function evolve(nx::Int, C::Float64, u::Float64, tmax::Float64)
    g = FDGrid(nx)
    dt = C * g.dx / u
    t = 0.0

    # Initialize top-hat
    for i in 1:g.nx
        if g.x[i] >= 1/3 && g.x[i] <= 2/3
            g.a[i] = 1.0
        end
    end
    g.ainit .= g.a
    fill_BCs!(g)

    # Preallocate coefficient matrix A (size (nx-1)x(nx-1))
    A = zeros(g.nx - 1, g.nx - 1)

    while t < tmax
        # Build the matrix A
        for i in 1:(g.nx - 1)
            A[i, i] = 1.0 + C
            if i > 1
                A[i, i-1] = -C
            else
                A[i, end] = -C  # wrap around for periodic BC
            end
        end

        # RHS vector excludes point 1 (which mirrors point nx)
        b = g.a[g.ilo:g.ihi]

        # Solve the system
        anew = A \ b
        g.a[g.ilo:g.ihi] .= anew
        fill_BCs!(g)
        t += dt
    end
    return g
end

# Parameters
u = 1.0
tmax = 1.0 / u
nx = 65
CFL = [0.01, 0.5, 1.0, 10.0]

# Plot results
p = plot(xlabel="x", ylabel="a", legend=:best, title="Implicit Linear Advection", size=(700,400))

for (n, C) in enumerate(CFL)
    g = evolve(nx, C, u, tmax)
    if n == 1
        plot!(p, g.x[g.ilo:g.ihi], g.ainit[g.ilo:g.ihi], label="exact", linestyle=:dot)
    end
    plot!(p, g.x[g.ilo:g.ihi], g.a[g.ilo:g.ihi], label="C = $C")
end

savefig(p, "/Users/xiaoquer/Desktop/julia code CAH/advection/fdadvect-implicit.pdf")