# For why we use Base.unsafe_trunc instead of floor below see:
# https://github.com/CliMA/Oceananigans.jl/issues/828
# https://github.com/CliMA/Oceananigans.jl/pull/997

"""
    permute_index(i, N)

Permute `i` such that, for example, `i ∈ 1:N` becomes

    [1, 2, 3, 4, 5, 6, 7, 8] -> [1, 8, 2, 7, 3, 6, 4, 5]

    [1, 2, 3, 4, 5, 6, 7, 8, 9] -> [1, 9, 2, 8, 3, 7, 4, 6, 5]

for `N=8` and `N=9` respectively.

See equation (20) of [Makhoul80](@cite).
"""
@inline permute_index(i, N)::Integer = ifelse(isodd(i),
                                              Base.unsafe_trunc(Int, i/2) + 1,
                                              N - Base.unsafe_trunc(Int, (i-1)/2))

"""
    unpermute_index(i, N)

Permute `i` in the opposite manner as `_permute_index`, such that,
for example, `i ∈ 1:N` becomes

   [1, 2, 3, 4, 5, 6, 7, 8] -> [1, 3, 5, 7, 8, 6, 4, 2]

   [1, 2, 3, 4, 5, 6, 7, 8, 9] -> [1, 3, 5, 7, 9, 8, 6, 4, 2]

for `N=8` and `N=9` respectively.

See equation (20) of [Makhoul80](@cite).
"""
@inline unpermute_index(i, N) = ifelse(i <= ceil(N/2), 2i-1, 2(N-i+1))

@kernel function permute_x_indices!(dst, src, grid)
    i, j, k = @index(Global, NTuple)
    i′ = permute_index(i, grid.Nx)
    @inbounds dst[i′, j, k] = src[i, j, k]
end

@kernel function permute_y_indices!(dst, src, grid)
    i, j, k = @index(Global, NTuple)
    j′ = permute_index(j, grid.Ny)
    @inbounds dst[i, j′, k] = src[i, j, k]
end

@kernel function permute_z_indices!(dst, src, grid)
    i, j, k = @index(Global, NTuple)
    k′ = permute_index(k, grid.Nz)
    @inbounds dst[i, j, k′] = src[i, j, k]
end

@kernel function unpermute_x_indices!(dst, src, grid)
    i, j, k = @index(Global, NTuple)
    i′ = unpermute_index(i, grid.Nx)
    @inbounds dst[i′, j, k] = src[i, j, k]
end

@kernel function unpermute_y_indices!(dst, src, grid)
    i, j, k = @index(Global, NTuple)
    j′ = unpermute_index(j, grid.Ny)
    @inbounds dst[i, j′, k] = src[i, j, k]
end

@kernel function unpermute_z_indices!(dst, src, grid)
    i, j, k = @index(Global, NTuple)
    k′ = unpermute_index(k, grid.Nz)
    @inbounds dst[i, j, k′] = src[i, j, k]
end

permute_kernel! = Dict(
    1 => permute_x_indices!,
    2 => permute_y_indices!,
    3 => permute_z_indices!
)

unpermute_kernel! = Dict(
    1 => unpermute_x_indices!,
    2 => unpermute_y_indices!,
    3 => unpermute_z_indices!
)

function permute_indices!(dst, src, arch, grid, dim)
    event = launch!(arch, grid, :xyz, permute_kernel![dim], dst, src, grid, dependencies=Event(device(arch)))
    wait(device(arch), event)
end

function unpermute_indices!(dst, src, arch, grid, dim)
    event = launch!(arch, grid, :xyz, unpermute_kernel![dim], dst, src, grid, dependencies=Event(device(arch)))
    wait(device(arch), event)
end
