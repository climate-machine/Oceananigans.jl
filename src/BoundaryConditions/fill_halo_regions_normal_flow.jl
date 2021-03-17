#####
##### Outer functions for setting velocity on boundary and filling halo beyond boundary.
#####

@kernel function set_east_west_u_velocity!(u, i_boundary, bc, grid, args...)
    j, k = @index(Global, NTuple)
    @inbounds u[i_boundary, j, k] = getbc(bc, j, k, grid, args...)
end

@kernel function set_north_south_v_velocity!(v, j_boundary, bc, grid, args...)
    i, k = @index(Global, NTuple)
    @inbounds v[i, j_boundary, k] = getbc(bc, i, k, grid, args...)
end

@kernel function set_top_bottom_w_velocity!(w, k_boundary, bc, grid, args...)
    i, j = @index(Global, NTuple)
    @inbounds w[i, j, k_boundary] = getbc(bc, i, j, grid, args...)
end

  fill_west_halo!(u, bc::NFBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :yz, set_east_west_u_velocity!,   u,           1, bc, grid, args...; dependencies=dep, kwargs...)
  fill_east_halo!(u, bc::NFBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :yz, set_east_west_u_velocity!,   u, grid.Nx + 1, bc, grid, args...; dependencies=dep, kwargs...)
 fill_south_halo!(v, bc::NFBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xz, set_north_south_v_velocity!, v,           1, bc, grid, args...; dependencies=dep, kwargs...)
 fill_north_halo!(v, bc::NFBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xz, set_north_south_v_velocity!, v, grid.Ny + 1, bc, grid, args...; dependencies=dep, kwargs...)
fill_bottom_halo!(w, bc::NFBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xy, set_top_bottom_w_velocity!,  w,           1, bc, grid, args...; dependencies=dep, kwargs...)
   fill_top_halo!(w, bc::NFBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xy, set_top_bottom_w_velocity!,  w, grid.Nz + 1, bc, grid, args...; dependencies=dep, kwargs...)
