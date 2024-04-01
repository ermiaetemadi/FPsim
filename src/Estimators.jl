function KM2_estimate(En::AbstractEnsemble, g::Grid, t_n::Int64, τ_n::Int64)
    # Estimates Kramers_Moyal coefficients 1 and 2
    dim = En.dim
    grid_dims = fill(g.N, g.dim)
    if typeof(En) <: LangevinEnsemble
        Δτ = τ_n * En.Δt
    elseif typeof(En) <: SampleEnsemble
        Δτ = τ_n * En.sample_Δt
    else
        error("what is this ensemble? "*typeof(En))
    end

    # Drif(vector) and Diffusion(matrix) for each bin in grid
    D1_vec = zeros((dim, grid_dims...))
    D2_mat = zeros((dim, dim, grid_dims...))

    particle_bins = find_bins(En, g, t_n)

    for ind in CartesianIndices(Tuple(grid_dims))
        particles_id = get(particle_bins, Tuple(ind), [])
        if length(particles_id) == 0
            continue
        end

        vec_initial = eachcol(En.paths[:, t_n, particles_id])
        vec_final = eachcol(En.paths[:, t_n + τ_n, particles_id])

        K1 = mean(vec_final - vec_initial)
        K2 = mean(cov_mat.(map(v-> v - K1, vec_final - vec_initial)))

        D1_vec[:, ind] = K1 / Δτ
        D2_mat[:, :, ind] = K2 / (2*Δτ)
    end

    return D1_vec, D2_mat
end
