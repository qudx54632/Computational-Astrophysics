using Plots, LaTeXStrings,Colors

# Define the root-finding function and its derivative
f(x) = (x - 1.0)^2 - 4
fprime(x) = 2.0 * (x - 1.0)

# Newton's method returning all iterates
function newton_trace(f, fprime, x0, tol=1e-5)
    xeval = [x0]
    dx = -f(x0)/fprime(x0)
    x = x0 + dx
    while abs(dx) > tol
        push!(xeval, x)
        dx = -f(x)/fprime(x)
        x += dx
    end
    return x, xeval
end

# Setup
xmin, xmax = 0.0, 5.0
xfine = range(xmin, xmax, length=200)
root, xeval = newton_trace(f, fprime, xmax)

for (n, x) in enumerate(xeval)
    plt = plot(xfine, f.(xfine),
        legend = false,
        xlabel = L"x", ylabel = L"f(x)",
        xlims = (xmin, 1.1*xmax),
        ylims = (-10, 15),
        framestyle = :zerolines,  # emulate axes through origin
        lw = 2,
    )

    # Newton point and tangent
    scatter!(plt, xeval[1:n], f.(xeval[1:n]), marker=:x, color=:red, ms=5)
    plot!(plt, xfine, fprime(x) * (xfine .- x) .+ f(x), color=RGB(0.5, 0.5, 0.5), lw=1)
    xint = x - f(x)/fprime(x)
    plot!(plt, [xint, xint], [0, f(xint)], linestyle=:dot, color=RGB(0.5, 0.5, 0.5))

    # Annotate each iteration
    dy = 0.3
    yval = f(x) + (-1)^n * dy
    annotate!(x, yval, text("$(n-1)", :red, 10, :center))

    # Label in lower right
    annotate!(4.0, -9.0, text("root approx = $(round(x, digits=6))", :black, 10))

    savefig(@sprintf("/Users/xiaoquer/Desktop/julia code CAH/basic_numerics/roots/newton_%02d.pdf", n-1))
end