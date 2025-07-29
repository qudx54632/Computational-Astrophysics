using LinearAlgebra

const GM = 4π^2 

struct OrbitHistory
    t::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    u::Vector{Float64}
    v::Vector{Float64}
end

function final_R(H::OrbitHistory)
    return hypot(H.x[end], H.y[end])
end

function displacement(H::OrbitHistory)
    return hypot(H.x[1] - H.x[end], H.y[1] - H.y[end])
end

mutable struct Orbit
    a::Float64
    e::Float64
    x0::Float64
    y0::Float64
    u0::Float64
    v0::Float64

    function Orbit(a, e)
        x0 = 0.0
        y0 = a * (1.0 - e)
        u0 = -sqrt((GM / a) * (1 + e) / (1 - e))  # perihelion velocity
        v0 = 0.0
        new(a, e, x0, y0, u0, v0)
    end
end

# Orbital period and velocities
kepler_period(o::Orbit) = sqrt(o.a^3)
circular_velocity(o::Orbit) = sqrt(GM / o.a)
escape_velocity(o::Orbit) = sqrt(2GM / o.a)

# Right-hand side of ODEs
function rhs(X::Tuple{Float64, Float64}, V::Tuple{Float64, Float64})
    x, y = X
    u, v = V
    r = hypot(x, y)
    xdot, ydot = u, v
    udot = -GM * x / r^3
    vdot = -GM * y / r^3
    return xdot, ydot, udot, vdot
end

# Integrators
function int_Euler(o::Orbit, dt, tmax)
    x, y, u, v = o.x0, o.y0, o.u0, o.v0
    t = 0.0
    tpoints = Float64[]
    xpoints = Float64[]
    ypoints = Float64[]
    upoints = Float64[]
    vpoints = Float64[]

    while t < tmax
        push!(tpoints, t); push!(xpoints, x); push!(ypoints, y)
        push!(upoints, u); push!(vpoints, v)

        if t + dt > tmax
            dt = tmax - t
        end

        xdot, ydot, udot, vdot = rhs((x, y), (u, v))
        u += dt * udot
        v += dt * vdot
        x += dt * xdot
        y += dt * ydot
        t += dt

    end

    return OrbitHistory(tpoints, xpoints, ypoints, upoints, vpoints)
end

function int_Euler_Cromer(o::Orbit, dt, tmax)
    x, y, u, v = o.x0, o.y0, o.u0, o.v0
    t = 0.0
    tpoints = Float64[]
    xpoints = Float64[]
    ypoints = Float64[]
    upoints = Float64[]
    vpoints = Float64[]

    while t < tmax
        push!(tpoints, t); push!(xpoints, x); push!(ypoints, y)
        push!(upoints, u); push!(vpoints, v)

        if t + dt > tmax
            dt = tmax - t
            if dt < 1e-12
                break
            end
        end

        xdot, ydot, udot, vdot = rhs((x, y), (u, v))
        u += dt * udot
        v += dt * vdot
        x += dt * u
        y += dt * v
        t += dt
    end

    return OrbitHistory(tpoints, xpoints, ypoints, upoints, vpoints)
end

function int_RK2(o::Orbit, dt, tmax)
    x, y, u, v = o.x0, o.y0, o.u0, o.v0
    t = 0.0
    tpoints = Float64[]
    xpoints = Float64[]
    ypoints = Float64[]
    upoints = Float64[]
    vpoints = Float64[]

    while t < tmax
        push!(tpoints, t); push!(xpoints, x); push!(ypoints, y)
        push!(upoints, u); push!(vpoints, v)

        if t + dt > tmax
            dt = tmax - t
            if dt < 1e-12
                break
            end
        end

        xdot1, ydot1, udot1, vdot1 = rhs((x, y), (u, v))
        xtmp = x + 0.5 * dt * xdot1
        ytmp = y + 0.5 * dt * ydot1
        utmp = u + 0.5 * dt * udot1
        vtmp = v + 0.5 * dt * vdot1

        xdot2, ydot2, udot2, vdot2 = rhs((xtmp, ytmp), (utmp, vtmp))

        x += dt * xdot2
        y += dt * ydot2
        u += dt * udot2
        v += dt * vdot2
        t += dt
    end

    return OrbitHistory(tpoints, xpoints, ypoints, upoints, vpoints)
end

function int_RK4(o::Orbit, dt, tmax)
    x, y, u, v = o.x0, o.y0, o.u0, o.v0
    t = 0.0
    tpoints = Float64[]
    xpoints = Float64[]
    ypoints = Float64[]
    upoints = Float64[]
    vpoints = Float64[]

    while t < tmax
        push!(tpoints, t); push!(xpoints, x); push!(ypoints, y)
        push!(upoints, u); push!(vpoints, v)

        if t + dt > tmax
            dt = tmax - t
            if dt < 1e-12
                break
            end
        end

        x1, y1, u1, v1 = rhs((x, y), (u, v))

        x2, y2, u2, v2 = rhs(
            (x + 0.5*dt*x1, y + 0.5*dt*y1),
            (u + 0.5*dt*u1, v + 0.5*dt*v1)
        )

        x3, y3, u3, v3 = rhs(
            (x + 0.5*dt*x2, y + 0.5*dt*y2),
            (u + 0.5*dt*u2, v + 0.5*dt*v2)
        )

        x4, y4, u4, v4 = rhs(
            (x + dt*x3, y + dt*y3),
            (u + dt*u3, v + dt*v3)
        )

        x += dt * (x1 + 2x2 + 2x3 + x4) / 6
        y += dt * (y1 + 2y2 + 2y3 + y4) / 6
        u += dt * (u1 + 2u2 + 2u3 + u4) / 6
        v += dt * (v1 + 2v2 + 2v3 + v4) / 6
        t += dt
    end

    return OrbitHistory(tpoints, xpoints, ypoints, upoints, vpoints)
end