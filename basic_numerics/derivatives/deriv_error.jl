using Plots, LaTeXStrings

# Define functions
f(x) = sin(x)
fprime(x) = cos(x)
dfdx(x, dx) = (f(x + dx) - f(x)) / dx

# Data
dx = 10 .^ range(-16.0, -1.0, length=1000)
x0 = 1.0
eps = abs.(dfdx.(x0, dx) .- fprime(x0))

# Plot
plot(dx, eps, xscale = :log10, yscale = :log10, xlabel = L"\delta x", ylabel = "error in difference approximation", legend = false)

current_dir = "/Users/xiaoquer/Desktop/julia code CAH/basic_numerics/derivatives/"
savefig(joinpath(current_dir, "deriv_error.pdf"))