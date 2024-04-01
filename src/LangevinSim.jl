struct LangevinCoff
    g::Function
    # We can have either the potential or the force field
    # TODO: implement the "potential" methods
    # U::Union{Function, Nothing} # U(s)
    force::Union{Function, Nothing} # f(s)
end

struct LangevinPath <: AbstractPath
    dim::Int
    N::Int
    Δt::Float64
    path::Matrix{Float64}   # dim ⨯ N Matrix
end

function init_LangevinPath(dim::Int, N::Int, Δt::Float64, coff::LangevinCoff,
                            X₀::Array{Float64})::LangevinPath
    steps = sqrt(Δt)*randn(dim, N)  # Steps for d-dimensional random walk
    path = zeros(dim, N)
    Xₜ = X₀
    @inbounds for i in 1:N
        path[:, i] = Xₜ
        Xₜ += coff.force(Xₜ)*Δt + coff.g(Xₜ)*steps[:, i]
    end

    return LangevinPath(dim, N, Δt, path)
end
struct LangevinEnsemble <: AbstractEnsemble
    dim::Int
    N::Int
    Δt::Float64
    n_ensemble::Int
    paths::Array{Float64}    # dim ⨯ N ⨯ n_ensemble Array
end

function init_LangevinEnsemble(dim::Int, N::Int, Δt::Float64, coff::LangevinCoff,
                                X₀_dist::Union{Function, Symbol},
                                n_ensemble::Int)::LangevinEnsemble
    steps = sqrt(Δt)*randn(dim, N, n_ensemble)  # n_ensemble steps for d-dimensional random walk
    paths = zeros(dim, N, n_ensemble)
    Xₜ = zeros(dim)
    if typeof(X₀_dist) <: Symbol
        if X₀_dist == :Delta
            init_dist = () -> zeros(dim)
        elseif X₀_dist == :Normal
            init_dist = () -> randn(dim)
        else
            error("The symbol :"*string(X₀_dist)*" is not defined for X₀_dist")
        end
    else
        init_dist = X₀_dist
    end

    #TODO: More performance checkups needed
    @inbounds for k in 1:n_ensemble
        Xₜ = init_dist()
        for i in 1:N
            paths[:, i, k] = Xₜ
            Xₜ += coff.force(Xₜ)*Δt + coff.g(Xₜ)*steps[:, i, k]
        end
    end

    return LangevinEnsemble(dim, N, Δt, n_ensemble, paths)
end

# # This function gives a LangevinPath from a LangevinEnsemble
# function init_LangevinPath(ens::LangevinEnsemble, X₀::Vector{Float64})

#     return init_LangevinEnsemble

# end

# TODO add more integration methods
