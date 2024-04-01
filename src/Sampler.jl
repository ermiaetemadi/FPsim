struct SamplePath <: AbstractPath
    dim::Int
    N::Int
    Δt::Float64             # Real Δt
    sample_Δt::Float64      # Sampled Δt
    path::Matrix{Float64}   # dim ⨯ N Matrix
end

struct SampleEnsemble <: AbstractEnsemble
    dim::Int
    N::Int
    Δt::Float64
    sample_Δt::Float64
    n_ensemble::Int
    selected_parts::Array{Int}
    paths::Array{Float64}    # dim ⨯ N ⨯ n_ensemble Array
end

function sample_path(P::AbstractPath, sample_step::Int64)
    # itr_step = ceil(Int64, sample_Δt/P.Δt)  #NOTE: With "ceil" the sampled Δt is not accurate
    itr_range = 1:sample_step:P.N
    itr_n = length(itr_range)
    sample_Δt = P.Δt * sample_step

    return SamplePath(P.dim, itr_n, P.Δt, sample_Δt, P.path[:, itr_range])
end

function sample_ensemble(En::AbstractEnsemble, sample_step::Int64, sample_n::Int)
    selected_parts = shuffle(1:En.n_ensemble)[1:sample_n]
    # itr_step = ceil(Int64, sample_Δt/En.Δt)
    itr_range = 1:sample_step:En.N
    itr_n = length(itr_range)
    sample_Δt = En.Δt * sample_step


    return SampleEnsemble(En.dim, itr_n, En.Δt, sample_Δt, En.n_ensemble, selected_parts,
            En.paths[:, itr_range, selected_parts])
end

struct SampleDist
    dim::Int
    N::Int
    sample_Δt::Float64
    dist_n::Int
    data::Array{Float64}    # dim ⨯ N ⨯ dist_n Array
end

function sample_dist(En::AbstractEnsemble, sample_step::Int64, sample_n::Int)
    # itr_step = ceil(Int64, sample_Δt/En.Δt)
    itr_range = 1:sample_step:En.N
    itr_n = length(itr_range)
    sample_Δt = En.Δt * sample_step

    # TODO: The namings are a little confusing!
    data = zeros(En.dim, itr_n, sample_n)

    for i in 1:itr_n
        # Sampling random particles each iteration
        selected_parts = shuffle(1:En.n_ensemble)[1:sample_n]
        data[:, i, :] = En.paths[:, itr_range[i], selected_parts]
    end

    return SampleDist(En.dim, itr_n, sample_Δt, sample_n, data)
end

Tuple
