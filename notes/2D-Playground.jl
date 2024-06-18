include("../src/FPsim.jl")

using .FPsim
using LinearAlgebra, Statistics, Random, Distributions
using GLMakie, LaTeXStrings
using BenchmarkTools

macro mdisplay(fig)        # Display multiple figures with makie
    return :(display(GLMakie.Screen(), $fig))
end

sim_t = 1000
sim_ensemble = 30000

coff_simple(α, dif) = LangevinCoff(t->dif*I, t->(-t + α*[0, t[1]*t[2]]))

ens = init_LangevinEnsemble(2, sim_t, 0.005, coff_simple(0.7, 1.0), :Normal, sim_ensemble)

f = Figure(size = (1000, 1000))
    ax = Axis(f[1, 1], xlabel = L"x", ylabel = L"y", xlabelsize = 30, ylabelsize = 30,
                title = "Trajectories")

        # t = Observable(1)
t_slider = Slider(f[2, 1], range = 1:1:1000, startvalue = 1)
t = t_slider.value

xdata = @lift(ens.paths[1, $t, :])
ydata = @lift(ens.paths[2, $t, :])

scatter!(ax, xdata, ydata, color = (:Blue, 0.5))
xlims!(ax, -4.5, 4.5)
ylims!(ax, -4.5, 4.5)

