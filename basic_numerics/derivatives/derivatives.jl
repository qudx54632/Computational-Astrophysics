using Plots, LaTeXStrings

# Define smooth and discrete data
x_smooth = range(0.0, π, length=500)
f_smooth = sin.(x_smooth)

x = range(0.0, π, length=10)
f = sin.(x)

i = 6
x0 = x[i]
f0 = f[i]

dx = x[2] - x[1]
d_exact = cos(x0)
d_left = (f[i] - f[i-1]) / dx
d_right = (f[i+1] - f[i]) / dx
d_central = (f[i+1] - f[i-1]) / (2 * dx)

# Line segment around x0
x_d = range(x[i] - 1.5*dx, x[i] + 1.5*dx, length=2)

# Plot
plot(x_smooth, f_smooth, label="", lw=2)
scatter!(x, f, label="", marker=:circle)

# Mark the chosen point
scatter!([x0], [f0], color=:red, label="")

# Show derivative approximations as lines
plot!(x_d, d_exact .* (x_d .- x0) .+ f0, color=:gray, lw=2, linestyle=:dot, label="exact")
plot!(x_d, d_left .* (x_d .- x0) .+ f0, label="left-sided")
plot!(x_d, d_right .* (x_d .- x0) .+ f0, label="right-sided")
plot!(x_d, d_central .* (x_d .- x0) .+ f0, label="centered")

# Add Δx annotation
y_ann = f[2] - 0.5dx
annotate!([(0.5*(x[2]+x[3]), y_ann - 0.1dx, L"\Delta x")])
plot!([x[2], x[2]], [f[2]-0.5dx, f[3]+0.5dx], lc=:gray, ls=:dot, label="")
plot!([x[3], x[3]], [f[2]-0.5dx, f[3]+0.5dx], lc=:gray, ls=:dot, label="")
# Arrow to show Δx (manual workaround, no direct <-> arrow in Plots.jl)
plot!([x[2], x[3]], [y_ann, y_ann], arrow=:arrow, lc=:black, label="")
plot!([x[3], x[2]], [y_ann, y_ann], arrow=:arrow, lc=:black, label="")

# Aesthetics
xlims!(-0.05, π+0.05)
ylims!(-0.05, 1.15)
xlabel!(L"x")
ylabel!(L"f(x)")

current_dir = "/Users/xiaoquer/Desktop/julia code CAH/basic_numerics/derivatives/"
savefig(joinpath(current_dir, "derivs.pdf"))