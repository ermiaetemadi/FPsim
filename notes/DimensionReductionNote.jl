using LinearAlgebra, BenchmarkTools, Profile
using Statistics, Distributions, Random
using UMAP
using GLMakie, LaTeXStrings
using ForwardDiff
GLMakie.activate!(inline=false)


# f(v) = sum(v.^2)    # Parabola
f(v) = v[1]^2 - v[2]^2 # And this one
σ = 0.4
f(v) = v[1]^2 - σ*exp(-(v[2]/σ)^2)

Tdata = rand(Uniform(-1, 1), (2, 1000))
Zdata = [Tdata; stack(f.(eachcol(Tdata)))']
grad_data = -stack(ForwardDiff.gradient.(f, eachcol(Tdata)))
grad_data = stack(map(v->[v[1], v[2], -(v[1]^2 + v[2]^2)], eachcol(grad_data)))
grad_data = stack(map(v->v./norm(v), eachcol(grad_data)))

scatter(Zdata, color = Zdata[3, :])
arrows(Zdata[1,:], Zdata[2,:], Zdata[3,:], grad_data[1,:], grad_data[2,:], grad_data[3,:]
        , lengthscale = 0.05, arrowsize = 0.005, color = Zdata[3,:])

umap_model  = UMAP_(Zdata, 2, n_neighbors = 50, min_dist = 0.5);

scatter(umap_model.embedding, color = Zdata[3, :])


function vector_embedd(umodel::UMAP_, poss::Matrix{T}, vecs::Matrix{T}; Δh = 0.01)::Matrix{T} where T <: Real
    @assert size(umodel.data)[1] == size(vecs)[1] "Incosistant dimension for vectors and embedding"
    @assert size(umodel.data)[1] == size(poss)[1] "Incosistant dimension for positions and embedding"

    trans_vecs_0 = transform(umodel, poss)
    trans_vecs_1 = transform(umodel, poss + Δh*vecs)

    return (trans_vecs_1 - trans_vecs_0) ./ Δh
end

new_grad = vector_embedd(umap_model, Zdata, grad_data, Δh = 1)

arrows(umap_model.embedding[1,:], umap_model.embedding[2,:], new_grad[1,:], new_grad[2,:]
        , linewidth = 2, lengthscale = 0.04, arrowsize = 10, color = Zdata[3,:])

test_line = stack(map(t->[sin(3*t), cos(3*t), t^2], 0:0.001:1))
scatter(test_line, color = test_line[3, :])
new_line = transform(umap_model, test_line)
scatter(new_line, color = test_line[3, :])
