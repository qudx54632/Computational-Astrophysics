using Printf

# Sample equations
f(x) = (x - 1.0)^2 - 4
fprime(x) = 2.0 * (x - 1.0)

g(x) = exp(x) - 4.0
gprime(x) = exp(x)

h(x) = x^2  # Bisection can't find this root
hprime(x) = 2.0 * x

# Define a RootFinder type
struct RootFinder
    fun::Function
    tol::Float64
    fprime::Union{Function, Nothing}
end

# Bisection method
function bisection(r::RootFinder, xl, xr)
    f = r.fun
    xeval = Float64[xl, xr]

    fl = f(xl)
    fr = f(xr)

    if fl * fr >= 0
        return nothing
    end

    err = abs(xr - xl)
    while err > r.tol
        xm = 0.5 * (xl + xr)
        fm = f(xm)
        push!(xeval, xm)

        if fm * fl >= 0
            xl = xm
            fl = fm
        else
            xr = xm
            fr = fm
        end

        err = abs(xr - xl)
    end

    return 0.5 * (xl + xr), xeval
end

# Newton's method
function newton(r::RootFinder, x0)
    f = r.fun
    fprime = r.fprime
    xeval = Float64[x0]

    dx = -f(x0) / fprime(x0)
    x = x0 + dx

    while abs(dx) > r.tol
        dx = -f(x) / fprime(x)
        push!(xeval, x)
        x += dx
    end

    return x, xeval
end

# Secant method
function secant(r::RootFinder, xm1, x0)
    f = r.fun

    dx = -f(x0) * (x0 - xm1) / (f(x0) - f(xm1))
    xm1, x = x0, x0 + dx

    while abs(dx) > r.tol
        dx = -f(x) * (x - xm1) / (f(x) - f(xm1))
        xm1, x = x, x + dx
    end

    return x
end

# Main test
function main()
    r = RootFinder(h, 1e-6, hprime)

    result_b = bisection(r, 0.0, 10.0)
    if result_b === nothing
        println("Bisection failed: root not bracketed")
    else
        rootb, xeval_b = result_b
        println("Bisection: ", rootb)
    end

    rootn, xeval_n = newton(r, 10.0)
    @printf("Newton:    %.6f\n", rootn)
    # println("x evaluations: ", xeval_n)

    roots = secant(r, 10.0, 9.0)
    println("Secant:    ", roots)
end

main()
