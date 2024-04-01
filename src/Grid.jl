struct Grid
    dim::Int64
    x_c::Array{Float64}
    Δx::Float64
    N::Int64
    CartSteps::NTuple{N, Int64} where {N}     # Steps for column major representation
    Grid(dim, x_c, Δx, N) = new(dim, x_c, Δx, N, Tuple([N^i for i in 0:(dim-1)]))
end

get_pos(g::Grid, ijk::CartesianIndex) = g.Δx*collect(Tuple(ijk)).*ones(g.dim) + g.x_c
get_ijk(g::Grid, pos::AbstractArray{Float64}) = CartesianIndex((floor.(Int64, (pos - g.x_c)./g.Δx))...)
ijk_to_ind(g::Grid, ijk::CartesianIndex) = sum((Tuple(ijk) .- 1).*g.CartSteps) + 1


#TODO: find_bin() is going to be a performance bottleneck
function find_bins(En::AbstractEnsemble, g::Grid, t_n::Int64)   # Places particles in bins
    data = @view En.paths[:, t_n, :]
    bin_dict = Dict{NTuple{En.dim, Int64}, Vector{Int64}}()

    for (id, pos) in enumerate(eachcol(data))
        ijk = get_ijk(g, pos)
        if all(r -> r>=0, ijk.I)    # We ignore particles outside the grid
            append!(get!(bin_dict, ijk.I, Vector{Int64}[]), id)    # Appends particle's "id" to the "ijk" bin
        end
    end
    return bin_dict
end


#TODO: choose between .I and Tuple()
