using Plots, LinearAlgebra, LaTeXStrings, OrdinaryDiffEq

include("/Users/xiaoquer/Desktop/julia code CAH/advection/weno_coefficients.jl")

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

# --------------------------------
# Lightweight WENO simulation type
# --------------------------------
mutable struct WENOSimulation
    grid
    u::Float64
    C::Float64
    weno_order::Int
    t::Float64
end

function WENOSimulation(grid, u; C=0.8, weno_order=3)
    WENOSimulation(grid, u, C, weno_order, 0.0)
end

# Initialize data (expects your grid to provide arrays/BCs)
function init_cond!(s::WENOSimulation; type::AbstractString = "tophat")
    g = s.grid
    L = g.xmax - g.xmin

    if type == "tophat"
        g.a .= 0.0
        @inbounds @simd for i in eachindex(g.x)
            if 1/3 <= g.x[i] <= 2/3
                g.a[i] = 1.0
            end
        end

    elseif type == "sine"
        @. g.a = sin(2π * g.x / L)

    elseif type == "gaussian"
        @. g.a = (1/6) * (
            1.0 + exp(-60.0 * (g.xl - 0.5)^2) +
            4.0 * (1.0 + exp(-60.0 * (g.x  - 0.5)^2)) +
            1.0 + exp(-60.0 * (g.xr - 0.5)^2)
        )

    elseif type == "sine_sine"
        @. g.a = sin(π * g.x - sin(π * g.x) / π)

    else
        error("Unknown init condition: $type (use \"tophat\", \"sine\", \"gaussian\", or \"sine_sine\")")
    end

    return nothing
end

# -------------------------
# Core WENO reconstruction
# -------------------------
function weno(order::Int, a::AbstractVector{<:Real})
    C = C_all[order]         # C_r: optimal (linear) weights, length r
    A = A_all[order]         # A_r: r×r reconstruction matrix
    σ = sigma_all[order]     # σ_r: r×r×r smoothness coefficients

    na = length(a)
    aL = zeros(eltype(a), na)          # reconstructed left states; boundaries stay 0
    β  = zeros(eltype(a), order, na)   # smoothness indicators per i
    np = na - 2 * order
    ϵ  = 1e-6


    @inbounds for i in order:(np + order)
        a_stencils = zeros(eltype(a), order)  # p_k(i)
        α = zeros(eltype(a), order)  # pre-weights

        for k in 1:order
            # β_k(i) = Σ_{ℓ=1..r} Σ_{m=1..ℓ} σ[k,ℓ,m] * a[i+k-ℓ] * a[i+k-m]
            accβ = zero(eltype(a))
            for l in 1:order
                for m in 1:l
                    accβ += σ[k,l,m] * a[i + k - l] * a[i + k - m]
                end
            end
            β[k,i] = accβ
            α[k]   = C[k] / (ϵ + β[k,i]^2)

            # p_k(i) = Σ_{ℓ=1..r} A[k,ℓ] * a[i + k - ℓ]
            accp = zero(eltype(a))
            for l in 1:order
                accp += A[k,l] * a[i + k - l]
            end
            a_stencils[k] = accp
        end

        # Nonlinear weights ω_k = α_k / Σ_j α_j
        denom = sum(α)
        @assert denom != 0
        ω = α ./ denom

        # a_{i+1/2}^- = Σ_k ω_k p_k(i)
        aL[i] = dot(ω, a_stencils)
    end


    return aL
end

function rk_substep(s::WENOSimulation)
    g = s.grid
    fill_BCs!(g)

    f  = s.u .* g.a
    α  = abs(s.u)
    fp = (f .+ α .* g.a) ./ 2      # positive flux part
    fm = (f .- α .* g.a) ./ 2      # negative flux part

    fpr  = similar(g.a)            # interface values at i+1/2 from the left
    fml  = similar(g.a)            # interface values at i-1/2 from the right
    flux = similar(g.a)

    # Right-going reconstruction: interfaces j = 2..end map to centers 1..end-1
    fpr[2:end] .= weno(s.weno_order, fp[1:end-1])

    # Left-going reconstruction: interfaces j = 1..end-1 map to centers 2..end
    fml[1:end-1] .= weno(s.weno_order, fm[2:end])

    # Combine upwind parts; flux[j] corresponds to interface j-1/2
    flux[2:end-1] .= fpr[2:end-1] .+ fml[2:end-1]

    rhs = zeros(eltype(g.a), length(g.a))
    rhs[2:end-1] .= (flux[2:end-1] .- flux[3:end]) ./ g.dx   # (F_{i-1/2} - F_{i+1/2})/dx
    return rhs
end

timestep(s::WENOSimulation) = s.C * s.grid.dx / max(eps(), abs(s.u))
period(s::WENOSimulation)   = (s.grid.xmax - s.grid.xmin) / max(eps(), abs(s.u))

function evolve!(s::WENOSimulation; num_periods::Int=1)
    s.t = 0.0
    g = s.grid
    tmax = num_periods * period(s)

    while s.t < tmax
        dt = timestep(s)
        if s.t + dt > tmax; dt = tmax - s.t; end

        a0 = copy(g.a)
        k1 = dt .* rk_substep(s)
        g.a .= a0 .+ k1 ./ 2
        k2 = dt .* rk_substep(s)
        g.a .= a0 .+ k2 ./ 2
        k3 = dt .* rk_substep(s)
        g.a .= a0 .+ k3
        k4 = dt .* rk_substep(s)
        g.a .= a0 .+ (k1 .+ 2 .* (k2 .+ k3) .+ k4) ./ 6
        s.t += dt
    end
    fill_BCs!(g)
    nothing
end

nx, ng = 256, 4
g   = Grid1D(nx, ng, 0.0, 1.0)
sim = WENOSimulation(g, 1.0; C=0.5, weno_order=3)

init_cond!(sim; type="sine")
a0 = copy(g.a)
evolve!(sim; num_periods=1)

err = norm(g.a .- a0) / sqrt(length(g.a))
@show err

plot(g.x[g.ilo:g.ihi], a0[g.ilo:g.ihi], ls=:dash, label="init")
plot!(g.x[g.ilo:g.ihi], g.a[g.ilo:g.ihi], lw=2, label="after 1 period")