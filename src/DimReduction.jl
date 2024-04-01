function vector_embedd(umodel::UMAP_, poss::Matrix{T}, vecs::Matrix{T}; Δh = 0.01)::Matrix{T} where T <: Real
    @assert size(umodel.data)[1] == size(vecs)[1] "Incosistant dimension for vectors and embedding"
    @assert size(umodel.data)[1] == size(poss)[1] "Incosistant dimension for positions and embedding"

    trans_vecs_0 = transform(umodel, poss)
    trans_vecs_1 = transform(umodel, poss + Δh*vecs)

    return (trans_vecs_1 - trans_vecs_0) ./ Δh
end
