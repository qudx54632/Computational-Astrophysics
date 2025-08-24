#  the main Riemann class for gamma-law Euler equations. The Riemann Problem class has methods to find the star state, plot the Hugoniot curves, and sample the solution.

using LinearAlgebra, LaTeXStrings, Plots
using Roots: find_zero, Brent

"""
A simple container for primitive variables.
"""
Base.@kwdef struct State
    p::Float64 = 1.0
    u::Float64 = 0.0
    rho::Float64 = 1.0
end

Base.show(io::IO, s::State) = print(io, "rho: $(s.rho); u: $(s.u); p: $(s.p)")


mutable struct RiemannProblem
    left::State
    right::State
    gamma::Float64
    ustar::Union{Nothing,Float64}
    pstar::Union{Nothing,Float64}
end

RiemannProblem(left::State, right::State; gamma::Float64=1.4) =
    RiemannProblem(left, right, gamma, nothing, nothing)

# --- helpers ---
sound_speed(gamma::Float64, s::State) = sqrt(gamma * s.p / s.rho)


"""
    u_hugoniot(rp, p, side; shock=false)

Hugoniot (u vs p) curve for the given `side` (:left or :right).
If `shock=true`, enforce the 2‑shock form.
"""
function u_hugoniot(rp::RiemannProblem, p::Float64, side::Symbol; shock::Bool=false)
    state, sgn = side === :left ? (rp.left, +1.0) : (rp.right, -1.0)
    g = rp.gamma
    c = sound_speed(g, state)

    if shock || p ≥ state.p
        # shock branch
        beta = (g + 1.0)/(g - 1.0)
        return state.u + sgn * (2.0*c / sqrt(2.0*g*(g - 1.0))) *
               (1.0 - p/state.p) / sqrt(1.0 + beta * p/state.p)
    else
        # rarefaction branch
        return state.u + sgn * (2.0*c/(g - 1.0)) *
               (1.0 - (p/state.p)^((g - 1.0)/(2.0*g)))
    end
end




# 1) Define states
left  = State(p=1.0,   u=0.0, rho=1.0)
right = State(p=0.1,   u=0.0, rho=0.125)

# 2) Make the problem (γ defaults to 1.4)
rp = RiemannProblem(left, right)