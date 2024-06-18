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

# U = t -> 0.1*(t't)^4 + ((t[1]*0.4 + 0.6)^2)*0.1*t[3]^2
# force_field = t -> -ForwardDiff.gradient(U, t)

# force_field = t -> (-0.5*t + [0, 0, 3*t[3]-0.36t[3]^3])
force_field = t -> -0.5*t + ((t[1]*0.7 + 0.7)^2)*[0, 0, 3*t[3]-0.36t[3]^3]
# force_field = t -> -0.5*t + ((t[1]*0.4 + 0.6)^2)*[0, 0, 3*t[3]-0.36t[3]^3] + 0.8*(0.6 + 0.4*t[1])*((3/2)t[3]^2 - (0.36/4)t[3]^4)*[1.0, 0, 0] # Last one but curlless

# force_field = t -> -0.4*(t't)*t + ((t[1]*0.4 + 0.6)^2)*[0, 0, 4*t[3]] + ((t[3] + 0.6)^2)*[0, 0, 4*t[2]]



coff = LangevinCoff(t-> Diagonal(1.0*ones(3)), force_field)
ens = init_LangevinEnsemble(3, sim_t, 0.005, coff, :Delta, sim_ensemble)

stationary_t = 1000


# Ploting
begin

    f = Figure(size = (1000, 1000))
    ax = Axis3(f[1, 1], xlabel = L"x", ylabel = L"y", xlabelsize = 30, ylabelsize = 30,
                title = "Orginal Trajectories")

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
    xlims!(ax, -4.5, 4.5)
    ylims!(ax, -4.5, 4.5)
    zlims!(ax, -4.5, 4.5)

    @mdisplay f
end

# UMAP embeddings of data and Force Field

umodel = UMAP_(ens.paths[:, stationary_t, :], 2, n_neighbors = 20, min_dist = 0.8)
# uembd = umap(ens.paths[:, stationary_t, :], 2, n_neighbors = 80, min_dist = 0.1)

f2 = Figure(size = (1000, 1000))
ax2 = Axis(f2[1, 1], aspect = 1, xlabel = "umap1", ylabel = "umap2", xlabelsize = 30,
            ylabelsize = 30, title = "UMAP embedding")
scatter!(ax2, umodel.embedding, color = ens.paths[3, stationary_t, :])
@mdisplay f2

force_vec = stack(force_field.(eachcol(ens.paths[:, stationary_t, :])))
force_vec_embedd = vector_embedd(umodel, ens.paths[:, stationary_t, :], force_vec; Δh = 0.3)

speed_vec = (ens.paths[:, stationary_t, :] - ens.paths[:, stationary_t-1, :] ) ./ (1*ens.Δt)
speed_vec_embedd = vector_embedd(umodel, ens.paths[:, stationary_t, :], speed_vec; Δh = 0.3)

plot_vec = force_vec_embedd # Which vectors are going to be plotted
color_ori = atan.(plot_vec[2, :], plot_vec[1, :]) # We choose the color based on the vectors orientations

f3 = Figure(size = (1000, 1000))
ax3 = Axis(f3[1, 1], aspect = 1, xlabel = "umap1", ylabel = "umap2", xlabelsize = 30,
            ylabelsize = 30, title = "Force Field UMAP embedding")
arrows!(ax3, umodel.embedding[1, :], umodel.embedding[2, :],
         plot_vec[1, :], plot_vec[2,:], lengthscale = 0.08,
          color = color_ori, arrowsize = 10, linewidth = 2)
@mdisplay f3

# Reconstruction of coefficients

g = Grid(3, [-4.0, -4, -4], 0.4, 20)


estimated_D1 = zeros(3, 20, 20, 20)
estimated_D2 = zeros(3, 3, 20, 20, 20)
sampling_range = 400:20:990
for t in sampling_range
    smp = sample_ensemble(ens, 1, 10000)
    D1_vec, D2_mat = KM2_estimate(smp, g, t, 5)
    estimated_D1 += D1_vec/ length(sampling_range)
    estimated_D2 += D2_mat/ length(sampling_range)
end

f4 = Figure(size = (1000, 1000))
ax4 = Axis3(f4[1, 1], xlabel = L"x", ylabel = L"y", xlabelsize = 30, ylabelsize = 30,
                title = "Reconstructed Force in the orginal space")

xlims!(ax4, -4.5, 4.5)
ylims!(ax4, -4.5, 4.5)
zlims!(ax4, -4.5, 4.5)
#TODO: Clean code for high-dim visuals

space_range = -3.98:0.4:4.0

pos3_array = [Point3f(x, y, z) for x in space_range for y in space_range for z in space_range]
vec3_array = [Vec3f(estimated_D1[1:3, k1, k2, k3]) for k1 in axes(estimated_D1, 2) for k2 in axes(estimated_D1, 3) for k3 in axes(estimated_D1, 4)]
arrows!(ax4, pos3_array, vec3_array, arrowsize = Vec3f(0.1, 0.1, 0.1), linewidth = 0.04 , lengthscale = 0.08)

@mdisplay f4


f5 = Figure(size = (1000, 1000))
ax5 = Axis3(f5[1, 1], xlabel = L"x", ylabel = L"y", xlabelsize = 30, ylabelsize = 30,
                title = "Force Field in the orginal space")

xlims!(ax5, -4.5, 4.5)
ylims!(ax5, -4.5, 4.5)
zlims!(ax5, -4.5, 4.5)

realf3_array = map(v -> force_field(v), pos3_array)
arrows!(ax5, pos3_array, realf3_array, arrowsize = Vec3f(0.1, 0.1, 0.1),
         linewidth = 0.04 , lengthscale = 0.1)

@mdisplay f5


f6 = Figure(size = (1000, 1000))
ax6 = Axis(f6[1, 1], aspect = 1, xlabel = "umap1", ylabel = "umap2", xlabelsize = 30,
            ylabelsize = 30, title = "Reconstructed Force UMAP embedding")

estimated_vec_embedd = vector_embedd(umodel, stack(pos3_array), stack(vec3_array); Δh = 0.3)
color_ori = atan.(estimated_vec_embedd[2, :], estimated_vec_embedd[1, :])
arrows!(ax6, umodel.embedding[1, :], umodel.embedding[2, :],
        estimated_vec_embedd[1, :], estimated_vec_embedd[2,:], lengthscale = 0.1,
        color = color_ori, arrowsize = 10, linewidth = 2, normalize = false)

@mdisplay f6



# More Plots!
pos_slice = [Point2f(x, y) for x in -5:0.5:5 for y in -5:0.5:5]
force_slice = [Vec2f(force_field([x, 0, z])[1:2:3]) for x in -5:0.5:5 for z in -5:0.5:5]

ftet = Figure(size = (1000, 1000))
axtet1 = Axis(ftet[1, 1], xlabel = L"x", ylabel = L"z", xlabelsize = 30, ylabelsize = 30,
                       title = "Force Field (y=0)(normalized)")

arrows!(axtet1, pos_slice, force_slice, arrowsize = 10.0, lengthscale = 0.2, linewidth = 3,
        normalize=true)

@mdisplay ftet

D1_slice = [Vec2f(estimated_D1[1:2:3, x, 10, z]) for x in 1:20 for z in 1:20]

ftet2 = Figure(size = (1000, 1000))
axtet2 = Axis(ftet2[1, 1], xlabel = L"x", ylabel = L"z", xlabelsize = 30, ylabelsize = 30,
                       title = "Reconstructed Force Field (y=0)")

arrows!(axtet2, pos_slice, D1_slice, arrowsize = 10.0, lengthscale = 0.2, linewidth = 3,
        normalize=false)

@mdisplay ftet2
