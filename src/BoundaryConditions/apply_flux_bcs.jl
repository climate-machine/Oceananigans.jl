using Oceananigans: instantiated_location
using Oceananigans.Grids
using Oceananigans.Grids: AbstractGrid

#####
##### Algorithm for adding fluxes associated with non-trivial flux boundary conditions.
##### Value and Gradient boundary conditions are handled by filling halos.
#####

# Unpack
apply_x_bcs!(Gc, c, args...) = apply_x_bcs!(Gc, Gc.grid, c, c.boundary_conditions.west, c.boundary_conditions.east, args...)
apply_y_bcs!(Gc, c, args...) = apply_y_bcs!(Gc, Gc.grid, c, c.boundary_conditions.south, c.boundary_conditions.north, args...)
apply_z_bcs!(Gc, c, args...) = apply_z_bcs!(Gc, Gc.grid, c, c.boundary_conditions.bottom, c.boundary_conditions.top, args...)

# Shortcuts for...
#
# Nothing tendencies.
apply_x_bcs!(::Nothing, args...) = nothing
apply_y_bcs!(::Nothing, args...) = nothing
apply_z_bcs!(::Nothing, args...) = nothing

# Not-flux boundary conditions
const NotFluxBC = Union{PBC, VBC, GBC, OBC, ZFBC}

apply_x_bcs!(Gc, grid, c, ::NotFluxBC, NotFluxBC, args...) = nothing
apply_y_bcs!(Gc, grid, c, ::NotFluxBC, NotFluxBC, args...) = nothing
apply_z_bcs!(Gc, grid, c, ::NotFluxBC, NotFluxBC, args...) = nothing

# The real deal
"""
Apply flux boundary conditions to a field `c` by adding the associated flux divergence to
the source term `Gc` at the left and right.
"""
apply_x_bcs!(Gc, grid, c, west_bc, east_bc, arch, dep, args...) =
    launch!(arch, grid, :yz, _apply_x_bcs!, Gc, instantiated_location(Gc), grid, west_bc, east_bc, args..., dependencies=dep)

"""
Apply flux boundary conditions to a field `c` by adding the associated flux divergence to
the source term `Gc` at the left and right.
"""
apply_y_bcs!(Gc, grid, c, south_bc, north_bc, arch, dep, args...) =
    launch!(arch, grid, :xz, _apply_y_bcs!, Gc, instantiated_location(Gc), grid, south_bc, north_bc, args..., dependencies=dep)

"""
Apply flux boundary conditions to a field `c` by adding the associated flux divergence to
the source term `Gc` at the top and bottom.
"""
apply_z_bcs!(Gc, grid, c, bottom_bc, top_bc, arch, dep, args...) =
    launch!(arch, grid, :xy, _apply_z_bcs!, Gc, instantiated_location(Gc), grid, bottom_bc, top_bc, args..., dependencies=dep)

"""
    _apply_x_bcs!(Gc, grid, west_bc, east_bc, args...)

Apply a west and/or east boundary condition to variable `c`.
"""
@kernel function _apply_x_bcs!(Gc, loc, grid, west_bc, east_bc, args...)
    j, k = @index(Global, NTuple)
    apply_x_west_bc!(Gc, loc, west_bc, j, k, grid, args...)
    apply_x_east_bc!(Gc, loc, east_bc, j, k, grid, args...)
end

"""
    _apply_y_bcs!(Gc, grid, south_bc, north_bc, args...)

Apply a south and/or north boundary condition to variable `c`.
"""
@kernel function _apply_y_bcs!(Gc, loc, grid, south_bc, north_bc, args...)
    i, k = @index(Global, NTuple)
    apply_y_south_bc!(Gc, loc, south_bc, i, k, grid, args...)
    apply_y_north_bc!(Gc, loc, north_bc, i, k, grid, args...)
end

"""
    _apply_z_bcs!(Gc, grid, bottom_bc, top_bc, args...)

Apply a top and/or bottom boundary condition to variable `c`.
"""
@kernel function _apply_z_bcs!(Gc, loc, grid, bottom_bc, top_bc, args...)
    i, j = @index(Global, NTuple)
    apply_z_bottom_bc!(Gc, loc, bottom_bc, i, j, grid, args...)
       apply_z_top_bc!(Gc, loc, top_bc,    i, j, grid, args...)
end

# Fall back functions for boundary conditions that are not of type Flux.
@inline apply_x_east_bc!(  Gc, args...) = nothing
@inline apply_x_west_bc!(  Gc, args...) = nothing
@inline apply_y_north_bc!( Gc, args...) = nothing
@inline apply_y_south_bc!( Gc, args...) = nothing
@inline apply_z_top_bc!(   Gc, args...) = nothing
@inline apply_z_bottom_bc!(Gc, args...) = nothing

# Shortcuts for 'zero' flux boundary conditions.
@inline apply_x_east_bc!(  Gc, loc, ::ZFBC, args...) = nothing
@inline apply_x_west_bc!(  Gc, loc, ::ZFBC, args...) = nothing
@inline apply_y_north_bc!( Gc, loc, ::ZFBC, args...) = nothing
@inline apply_y_south_bc!( Gc, loc, ::ZFBC, args...) = nothing
@inline apply_z_top_bc!(   Gc, loc, ::ZFBC, args...) = nothing
@inline apply_z_bottom_bc!(Gc, loc, ::ZFBC, args...) = nothing

@inline flip(::Center) = Face()
@inline flip(::Face) = Center()

"""
    apply_x_west_bc!(Gc, loc, west_flux::BC{<:Flux}, j, k, grid, args...)

Add the flux divergence associated with a west flux boundary condition on `c`.
Note that because

    `tendency = ∂c/∂t = Gc = - ∇ ⋅ flux`

a positive west flux is associated with an *increase* in `Gc` near the west boundary.
If `west_bc.condition` is a function, the function must have the signature

    `west_bc.condition(j, k, grid, boundary_condition_args...)`

The same logic holds for south and bottom boundary conditions in `y`, and `z`, respectively.
"""
@inline function apply_x_west_bc!(Gc, loc, west_flux::BC{<:Flux}, j, k, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[1, j, k] += getbc(west_flux, j, k, grid, args...) * Ax(1, j, k, grid, flip(LX), LY, LZ) / volume(1, j, k, grid, LX, LY, LZ)
    return nothing
end

@inline function apply_y_south_bc!(Gc, loc, south_flux::BC{<:Flux}, i, k, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[i, 1, k] += getbc(south_flux, i, k, grid, args...) * Ay(i, 1, k, grid, LX, flip(LY), LZ) / volume(i, 1, k, grid, LX, LY, LZ)
    return nothing
end

@inline function apply_z_bottom_bc!(Gc, loc, bottom_flux::BC{<:Flux}, i, j, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[i, j, 1] += getbc(bottom_flux, i, j, grid, args...) * Az(i, j, 1, grid, LX, LY, flip(LZ)) / volume(i, j, 1, grid, LX, LY, LZ)
    return nothing
end

"""
    apply_x_east_bc!(Gc, loc, east_flux::BC{<:Flux}, j, k, grid, args...)

Add the part of flux divergence associated with a east boundary condition on `c`.
Note that because

    `tendency = ∂c/∂t = Gc = - ∇ ⋅ flux`

a positive east flux is associated with a *decrease* in `Gc` near the east boundary.
If `east_bc.condition` is a function, the function must have the signature

    `east_bc.condition(i, j, grid, boundary_condition_args...)`

The same logic holds for north and top boundary conditions in `y`, and `z`, respectively.
"""
@inline function apply_x_east_bc!(Gc, loc, east_flux::BC{<:Flux}, j, k, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[grid.Nx, j, k] -= getbc(east_flux, j, k, grid, args...) * Ax(grid.Nx+1, j, k, grid, flip(LX), LY, LZ) / volume(grid.Nx, j, k, grid, LX, LY, LZ)
    return nothing
end

@inline function apply_y_north_bc!(Gc, loc, north_flux::BC{<:Flux}, i, k, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[i, grid.Ny, k] -= getbc(north_flux, i, k, grid, args...) * Ay(i, grid.Ny+1, k, grid, LX, flip(LY), LZ) / volume(i, grid.Ny, k, grid, LX, LY, LZ)
    return nothing
end

@inline function apply_z_top_bc!(Gc, loc, top_flux::BC{<:Flux}, i, j, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[i, j, grid.Nz] -= getbc(top_flux, i, j, grid, args...) * Az(i, j, grid.Nz+1, grid, LX, LY, flip(LZ)) / volume(i, j, grid.Nz, grid, LX, LY, LZ)
    return nothing
end
