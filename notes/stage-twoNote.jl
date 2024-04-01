include("../src/FPsim.jl")

using .FPsim
using LinearAlgebra, Statistics, Random, Distributions
using UMAP
using ForwardDiff
using GLMakie, LaTeXStrings
using BenchmarkTools
# using GLFW      # For double monitor support!
# monitors = GLFW.GetMonitors()
GLMakie.activate!(inline = false)
# set_theme!(theme_dark())
# set_theme!(theme_light())

macro mdisplay(fig)        # Display multiple figures with makie
    return :(display(GLMakie.Screen(), $fig))
end

sim_t = 1000
sim_ensemble = 10000
U = t -> 0.25*(t't)
force_field = t -> -ForwardDiff.gradient(U, t)
# force_field = t -> -0.4*(t't)*t + ((t[1]*0.4 + 0.6)^2)*[0, 0, 4*t[3]] + ((t[3] + 0.6)^2)*[0, 0, 4*t[2]]
# force_field = t -> (-0.5*t + ((t[1]*0.4 + 0.6)^2)*[0, 0, 3*t[3]-0.36t[3]^3])
# force_field = t -> (-0.5*t + [0, 0, 3*t[3]-0.36t[3]^3])


coff = LangevinCoff(t-> Diagonal(1.0*ones(3)), force_field)
ens = init_LangevinEnsemble(3, sim_t, 0.1, coff, :Delta, sim_ensemble)

# Ploting
begin

    f = Figure(size = (1000, 1000))
    ax = Axis3(f[1, 1], xlabel = L"x", ylabel = L"y", xlabelsize = 30, ylabelsize = 30)

        # t = Observable(1)
    t_slider = Slider(f[2, 1], range = 1:1:1000, startvalue = 1)
    t = t_slider.value

    xdata = @lift(ens.paths[1, $t, :])
    ydata = @lift(ens.paths[2, $t, :])
    zdata = @lift(ens.paths[3, $t, :])

    # I don't use these txtboxes anymore
    # p1_index_box = Textbox(f[2,2], placeholder = "index 1", validator = Int64)
    # p2_index_box = Textbox(f[2,3], placeholder = "index 2", validator = Int64)
    # p1_index = Observable(1)
    # p2_index = Observable(2)

    # on(p1_index_box.stored_string) do s
    #     p1_index[] = parse(Int64, s)
    # end

    # on(p2_index_box.stored_string) do s
    #     p2_index[] = parse(Int64, s)
    # end

    # diffcolor = repeat([(:Cyan, 0.2)], ens.n_ensemble)
    # diffcolor[35] = diffcolor[4978] = (:Yellow, 1.0)

    scatter!(ax, xdata, ydata, zdata, color = (:Blue, 0.5))
    xlims!(ax, -5, 5)
    ylims!(ax, -5, 5)
    zlims!(ax, -5, 5)

    @mdisplay f
end

umodel = UMAP_(ens.paths[:, end, :], 2, n_neighbors = 20, min_dist = 0.8)
# uembd = umap(ens.paths[:, end, :], 2, n_neighbors = 80, min_dist = 0.1)

force_vec = stack(force_field.(eachcol(ens.paths[:, end, :])))
force_vec_embedd = vector_embedd(umodel, ens.paths[:, end, :], force_vec; Δh = 0.2)

# speed_vec = (ens.paths[:, end, :] - ens.paths[:, end-1, :] ) ./ ens.Δt
# speed_vec_embedd = vector_embedd(umodel, ens.paths[:, end, :], speed_vec; Δh = 0.5)

plot_vec = force_vec_embedd # Which vectors are going to be plotted
color_ori = atan.(plot_vec[2, :], plot_vec[1, :]) # We choose the color based on the vectors orientations

f2 = Figure(size = (1000, 1000))
ax2 = Axis(f2[1, 1], aspect = 1, xlabel = "umap1", ylabel = "umap2", xlabelsize = 30, ylabelsize = 30)

scatter!(ax2, umodel.embedding, color = ens.paths[3, end, :])
# arrows!(ax2, umodel.embedding[1, :], umodel.embedding[2, :],
        #  plot_vec[1, :], plot_vec[2,:], lengthscale = 0.05,
        #   color = color_ori, arrowsize = 12, linewidth = 2)
@mdisplay f2
