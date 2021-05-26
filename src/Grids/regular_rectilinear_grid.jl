import Oceananigans.Architectures: architecture

"""
    RegularRectilinearGrid{FT, TX, TY, TZ, R} <: AbstractRectilinearGrid{FT, TX, TY, TZ}

A rectilinear grid with with constant grid spacings `Δx`, `Δy`, and `Δz` between cell centers
and cell faces, elements of type `FT`, topology `{TX, TY, TZ}`, and coordinate ranges
of type `R`.
"""
struct RegularRectilinearGrid{FT, TX, TY, TZ, R} <: AbstractRectilinearGrid{FT, TX, TY, TZ}
    # Number of grid points in (x,y,z).
    Nx :: Int
    Ny :: Int
    Nz :: Int
    # Halo size in (x,y,z).
    Hx :: Int
    Hy :: Int
    Hz :: Int
    # Domain size [m].
    Lx :: FT
    Ly :: FT
    Lz :: FT
    # Grid spacing [m].
    Δx :: FT
    Δy :: FT
    Δz :: FT
    # Range of coordinates at the centers of the cells.
    xC :: R
    yC :: R
    zC :: R
    # Range of grid coordinates at the faces of the cells.
    xF :: R
    yF :: R
    zF :: R
end

"""
    RegularRectilinearGrid([FT=Float64]; size,
                           extent = nothing, x = nothing, y = nothing, z = nothing,
                           topology = (Periodic, Periodic, Bounded), halo = (1, 1, 1))

Creates a `RegularRectilinearGrid` with `size = (Nx, Ny, Nz)` grid points.

Keyword arguments
=================

- `size` (required): A tuple prescribing the number of grid points in non-`Flat` directions.
                     `size` is a 3-tuple for 3D models, a 2-tuple for 2D models, and either a
                     scalar or 1-tuple for 1D models.

- `topology`: A 3-tuple `(Tx, Ty, Tz)` specifying the topology of the domain.
              `Tx`, `Ty`, and `Tz` specify whether the `x`-, `y`-, and `z` directions are
              `Periodic`, `Bounded`, or `Flat`. The topology `Flat` indicates that a model does
              not vary in that directions so that derivatives and interpolation are zero.
              The default is `topology=(Periodic, Periodic, Bounded)`.

- `extent`: A tuple prescribing the physical extent of the grid in non-`Flat` directions.
            The origin for three-dimensional domains is the oceanic default `(0, 0, -Lz)`.

- `x`, `y`, and `z`: Each of `x, y, z` are 2-tuples that specify the end points of the domain
                     in their respect directions. Scalar values may be used in `Flat` directions.

*Note*: _Either_ `extent`, or all of `x`, `y`, and `z` must be specified.

- `halo`: A tuple of integers that specifies the size of the halo region of cells surrounding
          the physical interior for each non-`Flat` direction.

The physical extent of the domain can be specified via `x`, `y`, and `z` keyword arguments
indicating the left and right endpoints of each dimensions, e.g. `x=(-π, π)` or via
the `extent` argument, e.g. `extent=(Lx, Ly, Lz)` which specifies the extent of each dimension
in which case 0 ≤ x ≤ Lx, 0 ≤ y ≤ Ly, and -Lz ≤ z ≤ 0.

A grid topology may be specified via a tuple assigning one of `Periodic`, `Bounded, and `Flat`
to each dimension. By default, a horizontally periodic grid topology `(Periodic, Periodic, Bounded)`
is assumed.

Constants are stored using floating point values of type `FT`. By default this is `Float64`.
Make sure to specify the desired `FT` if not using `Float64`.

Grid properties
===============

- `(Nx, Ny, Nz)::Integer`: Number of physical points in the (x, y, z)-direction

- `(Hx, Hy, Hz)::Integer`: Number of halo points in the (x, y, z)-direction

- `(Lx, Ly, Lz)::FT`: Physical extent of the grid in the (x, y, z)-direction

- `(Δx, Δy, Δz)::FT`: Grid spacing (distance between grid nodes) in the (x, y, z)-direction

- `(xC, yC, zC)`: (x, y, z) coordinates of cell centers.

- `(xF, yF, zF)`: (x, y, z) coordinates of cell faces.

Examples
========

* A default grid with Float64 type:

```jldoctest
julia> using Oceananigans

julia> grid = RegularRectilinearGrid(size=(32, 32, 32), extent=(1, 2, 3))
RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}
                   domain: x ∈ [0.0, 1.0], y ∈ [0.0, 2.0], z ∈ [-3.0, 0.0]
                 topology: (Periodic, Periodic, Bounded)
  resolution (Nx, Ny, Nz): (32, 32, 32)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (0.03125, 0.0625, 0.09375)
```

* A default grid with Float32 type:

```jldoctest
julia> using Oceananigans

julia> grid = RegularRectilinearGrid(Float32; size=(32, 32, 16), x=(0, 8), y=(-10, 10), z=(-π, π))
RegularRectilinearGrid{Float32, Periodic, Periodic, Bounded}
                   domain: x ∈ [0.0, 8.0], y ∈ [-10.0, 10.0], z ∈ [-3.1415927, 3.1415927]
                 topology: (Periodic, Periodic, Bounded)
  resolution (Nx, Ny, Nz): (32, 32, 16)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (0.25f0, 0.625f0, 0.3926991f0)
```

* A two-dimenisional, horizontally-periodic grid:

```jldoctest
julia> using Oceananigans

julia> grid = RegularRectilinearGrid(size=(32, 32), extent=(2π, 4π), topology=(Periodic, Periodic, Flat))
RegularRectilinearGrid{Float64, Periodic, Periodic, Flat}
                   domain: x ∈ [0.0, 6.283185307179586], y ∈ [0.0, 12.566370614359172], z ∈ [0.0, 0.0]
                 topology: (Periodic, Periodic, Flat)
  resolution (Nx, Ny, Nz): (32, 32, 1)
   halo size (Hx, Hy, Hz): (1, 1, 0)
grid spacing (Δx, Δy, Δz): (0.19634954084936207, 0.39269908169872414, 0.0)
```

* A one-dimensional "column" grid:

```jldoctest
julia> using Oceananigans

julia> grid = RegularRectilinearGrid(size=256, z=(-128, 0), topology=(Flat, Flat, Bounded))
RegularRectilinearGrid{Float64, Flat, Flat, Bounded}
                   domain: x ∈ [0.0, 0.0], y ∈ [0.0, 0.0], z ∈ [-128.0, 0.0]
                 topology: (Flat, Flat, Bounded)
  resolution (Nx, Ny, Nz): (1, 1, 256)
   halo size (Hx, Hy, Hz): (0, 0, 1)
grid spacing (Δx, Δy, Δz): (0.0, 0.0, 0.5)
```
"""
function RegularRectilinearGrid(FT=Float64;
                                  size,
                                     x = nothing, y = nothing, z = nothing,
                                extent = nothing,
                              topology = (Periodic, Periodic, Bounded),
                                  halo = nothing
                              )

    TX, TY, TZ = validate_topology(topology)
    size = validate_size(TX, TY, TZ, size)
    halo = validate_halo(TX, TY, TZ, halo)
    Lx, Ly, Lz, x, y, z = validate_regular_grid_domain(TX, TY, TZ, FT, extent, x, y, z)

    # Unpacking
    Nx, Ny, Nz = N = size
    Hx, Hy, Hz = H = halo
                 L = (Lx, Ly, Lz)
    Δx, Δy, Δz = Δ = L ./ N
                X₁ = (x[1], y[1], z[1])

    # Face-node limits in x, y, z
    xF₋, yF₋, zF₋ = XF₋ = @. X₁ - H * Δ
    xF₊, yF₊, zF₊ = XF₊ = @. XF₋ + total_extent(topology, halo, Δ, L)

    # Center-node limits in x, y, z
    xC₋, yC₋, zC₋ = XC₋ = @. XF₋ + Δ / 2
    xC₊, yC₊, zC₊ = XC₊ = @. XC₋ + L + Δ * (2H - 1)

    TFx, TFy, TFz = total_length.(Face, topology, N, H)
    TCx, TCy, TCz = total_length.(Center, topology, N, H)

    # Include halo points in coordinate arrays
    xF = range(xF₋, xF₊; length = TFx)
    yF = range(yF₋, yF₊; length = TFy)
    zF = range(zF₋, zF₊; length = TFz)

    xC = range(xC₋, xC₊; length = TCx)
    yC = range(yC₋, yC₊; length = TCy)
    zC = range(zC₋, zC₊; length = TCz)

    # Offset.
    xC = OffsetArray(xC, -Hx)
    yC = OffsetArray(yC, -Hy)
    zC = OffsetArray(zC, -Hz)

    xF = OffsetArray(xF, -Hx)
    yF = OffsetArray(yF, -Hy)
    zF = OffsetArray(zF, -Hz)

    return RegularRectilinearGrid{FT, TX, TY, TZ, typeof(xC)}(
        Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, Δx, Δy, Δz, xC, yC, zC, xF, yF, zF)
end

"""
    with_halo(new_halo, old_grid::RegularRectilinearGrid)

Returns a new `RegularRectilinearGrid` with the same properties as
`old_grid` but with halos set to `new_halo`.

Note that in contrast to the constructor for `RegularRectilinearGrid`,
`new_halo` is expected to be a 3-`Tuple` by `with_halo`. The elements
of `new_halo` corresponding to `Flat` directions are removed (and are
therefore ignored) prior to constructing the new `RegularRectilinearGrid`.
"""
function with_halo(new_halo, old_grid::RegularRectilinearGrid)

    Nx, Ny, Nz = size = (old_grid.Nx, old_grid.Ny, old_grid.Nz)
    topo = topology(old_grid)

    x = x_domain(old_grid)
    y = y_domain(old_grid)
    z = z_domain(old_grid)

    # Remove elements of size and new_halo in Flat directions as expected by grid
    # constructor
    size = pop_flat_elements(size, topo)
    new_halo = pop_flat_elements(new_halo, topo)

    new_grid = RegularRectilinearGrid(eltype(old_grid); size=size, x=x, y=y, z=z,
                                      topology=topo, halo=new_halo)

    return new_grid
end

short_show(grid::RegularRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    "RegularRectilinearGrid{$FT, $TX, $TY, $TZ}(Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz))"

function domain_string(grid)
    xₗ, xᵣ = x_domain(grid)
    yₗ, yᵣ = y_domain(grid)
    zₗ, zᵣ = z_domain(grid)
    return "x ∈ [$xₗ, $xᵣ], y ∈ [$yₗ, $yᵣ], z ∈ [$zₗ, $zᵣ]"
end

function show(io::IO, g::RegularRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    print(io, "RegularRectilinearGrid{$FT, $TX, $TY, $TZ}\n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "  resolution (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δx, Δy, Δz): ", (g.Δx, g.Δy, g.Δz))
end


#
# Get minima of grid
#

min_Δx(grid::RegularRectilinearGrid) = grid.Δx
min_Δy(grid::RegularRectilinearGrid) = grid.Δy
min_Δz(grid::RegularRectilinearGrid) = grid.Δz

# All grid metrics are constants / functions / ranges, so there's no architecture.
architecture(::RegularRectilinearGrid) = nothing
