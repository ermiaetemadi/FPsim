module FPsim

    using LinearAlgebra, Statistics, Random, Distributions
    using UMAP
    using ForwardDiff
    using GLMakie, LaTeXStrings

    include("./types.jl")
    include("./MathUtils.jl")
    include("./LangevinSim.jl")
    include("./Sampler.jl")
    include("./Grid.jl")
    include("./Estimators.jl")
    include("./DimReduction.jl")

    export LangevinCoff, LangevinPath, LangevinEnsemble, init_LangevinPath, init_LangevinEnsemble
    export SamplePath, SampleEnsemble, sample_path, sample_ensemble, sample_dist
    export Grid, get_pos, get_ijk, find_bins
    export KM2_estimate
    export vector_embedd

end
