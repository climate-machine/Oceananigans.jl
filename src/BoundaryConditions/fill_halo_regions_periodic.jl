#####
##### Periodic boundary conditions
#####

  fill_west_halo!(c, ::PBC, H::Integer, N) = @views @. c.parent[1:H, :, :] = c.parent[N+1:N+H, :, :]
 fill_south_halo!(c, ::PBC, H::Integer, N) = @views @. c.parent[:, 1:H, :] = c.parent[:, N+1:N+H, :]
fill_bottom_halo!(c, ::PBC, H::Integer, N) = @views @. c.parent[:, :, 1:H] = c.parent[:, :, N+1:N+H]

 fill_east_halo!(c, ::PBC, H::Integer, N) = @views @. c.parent[N+H+1:N+2H, :, :] = c.parent[1+H:2H, :, :]
fill_north_halo!(c, ::PBC, H::Integer, N) = @views @. c.parent[:, N+H+1:N+2H, :] = c.parent[:, 1+H:2H, :]
  fill_top_halo!(c, ::PBC, H::Integer, N) = @views @. c.parent[:, :, N+H+1:N+2H] = c.parent[:, :, 1+H:2H]

# Generate interface functions for periodic boundary conditions
sides = (:west, :east, :south, :north, :top, :bottom)
coords = (:x, :x, :y, :y, :z, :z)

for (x, side) in zip(coords, sides)
    name = Symbol(:fill_, side, :_halo!)
    H = Symbol(:H, x)
    N = Symbol(:N, x)
    @eval begin
        $name(c, bc::PBC, arch, dependencies, grid, args...; kwargs...) =
            $name(c, bc, grid.$(H), grid.$(N))
    end
end
