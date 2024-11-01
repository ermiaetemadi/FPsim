include("../src/FPsim.jl")

using .FPsim
using LinearAlgebra, Statistics, Random, Distributions
using GLMakie, LaTeXStrings
using BenchmarkTools
using NNlib         # For sigmoid function


macro mdisplay(fig)        # Display multiple figures with makie
    return :(display(GLMakie.Screen(), $fig))
end

sim_t = 1000
sim_ensemble = 1000

# coff_simple(α, dif) = LangevinCoff(v->dif, v->zero(v))
# coff_simple(α, dif) = LangevinCoff(v->dif*I, v->(-v + α*[0, sigmoid(v[1]-0.4)*(-v[2]^3+6*v[2])]))
# coff_simple(α, dif) = LangevinCoff(v->dif*(dif/(norm(v) + dif))*I, v->(-v + α*[0, sigmoid(v[1]-0.4)*(-v[2]^3+6*v[2])]))
coff_simple(α, dif) = LangevinCoff(v->dif*sigmoid(0.333*v'v - 1.0), v -> -α*v)
# coff_simple(α, dif) = LangevinCoff(v->dif, v -> -α*v)
# coff_simple(α, dif) = LangevinCoff(v -> dif*I, v -> (-α*v + [0, 0, 3*v[3]-0.36v[3]^3]))


ens = init_LangevinEnsemble(2, sim_t, 0.005, coff_simple(0.5, 1.0), :Delta, sim_ensemble)

f = Figure(size = (1000, 1000))
    ax = Axis(f[1, 1], xlabel = L"x", ylabel = L"y", xlabelsize = 30, ylabelsize = 30,
                title = "Trajectories")

        # t = Observable(1)
t_slider = Slider(f[2, 1], range = 1:1:sim_t, startvalue = 1)
t = t_slider.value
time_text = @lift("t = "*string(round($t*ens.Δt, digits = 2)))

xdata = @lift(ens.paths[1, $t, :])
ydata = @lift(ens.paths[2, $t, :])

scatter!(ax, xdata, ydata, color = (:Blue, 0.5))
xlims!(ax, -4.5, 4.5)
ylims!(ax, -4.5, 4.5)
text!(ax, -3.0, 4.0; text = time_text, fontsize = 30)

@mdisplay f


# Reconstruction

grid_N = 60

g = Grid(2, [-3.0, -3.0], 0.1, grid_N)

estimated_D1 = zeros(2, grid_N, grid_N)
estimated_D2 = zeros(2, 2, grid_N, grid_N)
sampling_range = 100:50:950
for t in sampling_range
    smp = sample_ensemble(ens, 1, 1000)
    D1_vec, D2_mat = KM2_estimate(smp, g, t, 10)
    estimated_D1 += D1_vec/ length(sampling_range)
    estimated_D2 += D2_mat/ length(sampling_range)
end

pos_list = stack(map(t->FPsim.get_pos(g, t), [CartesianIndex(j ,i) for i in 1:grid_N for j in 1:grid_N]))
alpha_list_D1 = [ifelse(norm(estimated_D1[1:2, j,i])>0.00000001, 1.0, 0.0) for i in 1:grid_N for j in 1:grid_N]
arrows(pos_list[1, :], pos_list[2, :], estimated_D1[1, :, :][:], estimated_D1[2, :, :][:]
        , lengthscale = 0.2, color = map(v->(:black, v), alpha_list_D1))

D2_intensity = mapslices(norm, estimated_D2, dims = (1,2))[1, 1, :, :]
fig3, ax3, hm = heatmap(D2_intensity)
Colorbar(fig3[1, 2], hm)
fig3
