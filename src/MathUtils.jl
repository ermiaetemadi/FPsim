@inline function cov_mat(v::Vector{T}) where {T<:Number}
    return  v*v'
end
