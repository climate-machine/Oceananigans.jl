struct VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ, R, A, Arch} <: AbstractRectilinearGrid{FT, TX, TY, TZ}

    architecture :: Arch

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
    Δzᵃᵃᶜ :: A
    Δzᵃᵃᶠ :: A

    # Range of coordinates at the centers of the cells.
    xᶜᵃᵃ :: R
    yᵃᶜᵃ :: R
    zᵃᵃᶜ :: A

    # Range of grid coordinates at the faces of the cells.
    # Note: there are Nx+1 faces in the x-dimension, Ny+1 in the y, and Nz+1 in the z.
    xᶠᵃᵃ :: R
    yᵃᶠᵃ :: R
    zᵃᵃᶠ :: A
end

"""
    VerticallyStretchedRectilinearGrid([FT=Float64]; architecture=CPU(), size, z_faces,
                                        x = nothing, y = nothing,
                                        topology = (Periodic, Periodic, Bounded), halo = nothing)

Creates a `VerticallyStretchedRectilinearGrid` with `size = (Nx, Ny, Nz)` grid points and
a vertical grid specified by `z_faces`.

Keyword arguments
=================

- `size` (required): A tuple prescribing the number of grid points in non-`Flat` directions.
                     `size` is a 3-tuple for 3D models, a 2-tuple for 2D models, and either a
                     scalar or 1-tuple for 1D models.

- `topology`: A 3-tuple `(Tx, Ty, Tz)` specifying the topology of the domain.
              `Tz` must be `Bounded` for `VerticallyStretchedRectilinearGrid`.
              `Tx` and `Ty` specify whether the `x`- and `y`-` directions are
              `Periodic`, `Bounded`, or `Flat`. The topology `Flat` indicates that a model does
              not vary in that directions so that derivatives and interpolation are zero.
              The default is `topology=(Periodic, Periodic, Bounded)`.

- `architecture`: Specifies whether the array of vertical coordinates, interfaces, and spacings
                  are stored on the CPU or GPU. Default: `architecture = CPU()`.

- `z_faces`: An array or function of vertical index `k` that specifies the location of cell faces
        in the `z-`direction for indices `k=1` through `k=Nz+1`, where `Nz` is the third element
        of `size`.

- `x`, `y`: Each of `x, y` are 2-tuples that specify the end points of the domain
            in their respect directions. Scalar values may be used in `Flat` directions.

- `halo`: A tuple of integers that specifies the size of the halo region of cells surrounding
          the physical interior for each non-`Flat` direction.

The physical extent of the domain can be specified via `x` and `y` keyword arguments
indicating the left and right endpoints of each dimensions, e.g. `x=(-π, π)`.

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

- `(Δx, Δy)::FT`: Grid spacing (distance between grid nodes) in the (x, y)-direction

- `Δzᵃᵃᶜ`: Grid spacing in the z-direction between cell faces.
           Defined at cell centers in `z` and independent of cell location in (x, y).

- `Δzᵃᵃᶠ`: Grid spacing in the z-direction between cell centers, and defined at cell faces in z.
           Defined at cell faces in `z` and independent of cell location in (x, y).

- `(xᶜᵃᵃ, yᵃᶜᵃ, zᵃᵃᶜ)`: (x, y, z) coordinates of cell centers.

- `(xᶠᵃᵃ, yᵃᶠᵃ, zᵃᵃᶠ)`: (x, y, z) coordinates of cell faces.
"""
function VerticallyStretchedRectilinearGrid(FT = Float64;
                                            architecture = CPU(),
                                            size,
                                            z_faces,
                                            x = nothing,
                                            y = nothing,
                                            halo = nothing,
                                            topology = (Periodic, Periodic, Bounded))

    TX, TY, TZ = validate_topology(topology)
    size = validate_size(TX, TY, TZ, size)
    halo = validate_halo(TX, TY, TZ, halo)
    x = validate_dimension_specification(TX, x, :x)
    y = validate_dimension_specification(TY, y, :y)
    Lx, Ly, x, y = validate_vertically_stretched_grid_xy(TX, TY, FT, x, y)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    # Initialize vertically-stretched arrays on CPU
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶜ, Δzᵃᵃᶠ = generate_stretched_vertical_grid(FT, topology[3], Nz, Hz, z_faces)

    # Construct uniform horizontal grid
    Lh, Nh, Hh, X₁ = (Lx, Ly), size[1:2], halo[1:2], (x[1], y[1])
    Δx, Δy = Δh = Lh ./ Nh

    # Face-node limits in x, y, z
    xF₋, yF₋ = XF₋ = @. X₁ - Hh * Δh
    xF₊, yF₊ = XF₊ = @. XF₋ + total_extent(topology[1:2], Hh, Δh, Lh)

    # Center-node limits in x, y, z
    xC₋, yC₋ = XC₋ = @. XF₋ + Δh / 2
    xC₊, yC₊ = XC₊ = @. XC₋ + Lh + Δh * (2Hh - 1)

    # Total length of Center and Face quantities
    TFx, TFy, TFz = total_length.(Face, topology, size, halo)
    TCx, TCy, TCz = total_length.(Center, topology, size, halo)

    # Include halo points in coordinate arrays
    xᶠᵃᵃ = range(xF₋, xF₊; length = TFx)
    yᵃᶠᵃ = range(yF₋, yF₊; length = TFy)

    xᶜᵃᵃ = range(xC₋, xC₊; length = TCx)
    yᵃᶜᵃ = range(yC₋, yC₊; length = TCy)

    xᶜᵃᵃ = OffsetArray(xᶜᵃᵃ,  -Hx)
    yᵃᶜᵃ = OffsetArray(yᵃᶜᵃ,  -Hy)
    zᵃᵃᶜ = OffsetArray(zᵃᵃᶜ,  -Hz)

    xᶠᵃᵃ = OffsetArray(xᶠᵃᵃ,  -Hx)
    yᵃᶠᵃ = OffsetArray(yᵃᶠᵃ,  -Hy)
    zᵃᵃᶠ = OffsetArray(zᵃᵃᶠ,  -Hz)

    Δzᵃᵃᶠ = OffsetArray(Δzᵃᵃᶠ, -Hz)
    Δzᵃᵃᶜ = OffsetArray(Δzᵃᵃᶜ, -Hz)

    # Needed for pressure solver solution to be divergence-free.
    # Will figure out why later...
    Δzᵃᵃᶠ[Nz] = Δzᵃᵃᶠ[Nz-1]

    # Seems needed to avoid out-of-bounds error in viscous dissipation
    # operators wanting to access Δzᵃᵃᶠ[Nz+2].
    Δzᵃᵃᶠ = OffsetArray(cat(Δzᵃᵃᶠ[0], Δzᵃᵃᶠ..., Δzᵃᵃᶠ[Nz], dims=1), -Hz-1)

    # Convert to appropriate array type for arch
    zᵃᵃᶠ  = OffsetArray(arch_array(architecture,  zᵃᵃᶠ.parent),  zᵃᵃᶠ.offsets...)
    zᵃᵃᶜ  = OffsetArray(arch_array(architecture,  zᵃᵃᶜ.parent),  zᵃᵃᶜ.offsets...)
    Δzᵃᵃᶜ = OffsetArray(arch_array(architecture, Δzᵃᵃᶜ.parent), Δzᵃᵃᶜ.offsets...)
    Δzᵃᵃᶠ = OffsetArray(arch_array(architecture, Δzᵃᵃᶠ.parent), Δzᵃᵃᶠ.offsets...)

    R = typeof(xᶠᵃᵃ)
    A = typeof(zᵃᵃᶠ)
    Arch = typeof(architecture)

    return VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ, R, A, Arch}(architecture,
        Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, Δx, Δy, Δzᵃᵃᶜ, Δzᵃᵃᶠ, xᶜᵃᵃ, yᵃᶜᵃ, zᵃᵃᶜ, xᶠᵃᵃ, yᵃᶠᵃ, zᵃᵃᶠ)
end

#####
##### Vertically stretched grid utilities
#####

get_z_face(z::Function, k) = z(k)
get_z_face(z::AbstractVector, k) = CUDA.@allowscalar z[k]

lower_exterior_Δzᵃᵃᶜ(z_topo,          zFi, Hz) = [zFi[end - Hz + k] - zFi[end - Hz + k - 1] for k = 1:Hz]
lower_exterior_Δzᵃᵃᶜ(::Type{Bounded}, zFi, Hz) = [zFi[2]  - zFi[1] for k = 1:Hz]

upper_exterior_Δzᵃᵃᶜ(z_topo,          zFi, Hz) = [zFi[k + 1] - zFi[k] for k = 1:Hz]
upper_exterior_Δzᵃᵃᶜ(::Type{Bounded}, zFi, Hz) = [zFi[end]   - zFi[end - 1] for k = 1:Hz]

function generate_stretched_vertical_grid(FT, z_topo, Nz, Hz, z_faces)

    # Ensure correct type for zF and derived quantities
    interior_zF = zeros(FT, Nz+1)

    for k = 1:Nz+1
        interior_zF[k] = get_z_face(z_faces, k)
    end

    Lz = interior_zF[Nz+1] - interior_zF[1]

    # Build halo regions
    ΔzF₋ = lower_exterior_Δzᵃᵃᶜ(z_topo, interior_zF, Hz)
    ΔzF₊ = lower_exterior_Δzᵃᵃᶜ(z_topo, interior_zF, Hz)

    z¹, zᴺ⁺¹ = interior_zF[1], interior_zF[Nz+1]

    zF₋ = [z¹   - sum(ΔzF₋[k:Hz]) for k = 1:Hz] # locations of faces in lower halo
    zF₊ = [zᴺ⁺¹ + ΔzF₊[k]         for k = 1:Hz] # locations of faces in width of top halo region

    zF = vcat(zF₋, interior_zF, zF₊)

    # Build cell centers, cell center spacings, and cell interface spacings
    TCz = total_length(Center, z_topo, Nz, Hz)
     zC = [ (zF[k + 1] + zF[k]) / 2 for k = 1:TCz ]
    ΔzC = [  zC[k] - zC[k - 1]      for k = 2:TCz ]

    # Trim face locations for periodic domains
    TFz = total_length(Face, z_topo, Nz, Hz)
    zF = zF[1:TFz]

    ΔzF = [zF[k + 1] - zF[k] for k = 1:TFz-1]

    return Lz, zF, zC, ΔzF, ΔzC
end

"""
    with_halo(new_halo, old_grid::VerticallyStretchedRectilinearGrid)

Returns a new `VerticallyStretchedRectilinearGrid` with the same properties as
`old_grid` but with halos set to `new_halo`.

Note that in contrast to the constructor for `VerticallyStretchedRectilinearGrid`,
`new_halo` is expected to be a 3-`Tuple` by `with_halo`. The elements
of `new_halo` corresponding to `Flat` directions are removed (and are
therefore ignored) prior to constructing the new `VerticallyStretchedRectilinearGrid`.
"""
function with_halo(new_halo, old_grid::VerticallyStretchedRectilinearGrid)

    Nx, Ny, Nz = size = (old_grid.Nx, old_grid.Ny, old_grid.Nz)
    topo = topology(old_grid)

    x = x_domain(old_grid)
    y = y_domain(old_grid)
    z = z_domain(old_grid)

    # Remove elements of size and new_halo in Flat directions as expected by grid
    # constructor
    size = pop_flat_elements(size, topo)
    new_halo = pop_flat_elements(new_halo, topo)

    new_grid = VerticallyStretchedRectilinearGrid(eltype(old_grid);
                                                  architecture = old_grid.architecture,
                                                  size = size,
                                                  x = x, y = y,
                                                  z_faces = old_grid.zᵃᵃᶠ,
                                                  topology = topo,
                                                  halo = new_halo)

    return new_grid
end

@inline x_domain(grid::VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TX, grid.Nx, grid.xᶠᵃᵃ)
@inline y_domain(grid::VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TY, grid.Ny, grid.yᵃᶠᵃ)
@inline z_domain(grid::VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)

short_show(grid::VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    "VerticallyStretchedRectilinearGrid{$FT, $TX, $TY, $TZ}(Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz))"

function show(io::IO, g::VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    Δz_min = minimum(view(parent(g.Δzᵃᵃᶜ), g.Hz+1:g.Nz+g.Hz))
    Δz_max = maximum(view(parent(g.Δzᵃᵃᶜ), g.Hz+1:g.Nz+g.Hz))
    print(io, "VerticallyStretchedRectilinearGrid{$FT, $TX, $TY, $TZ}\n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "  resolution (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δx, Δy, Δz): (", g.Δx, ", ", g.Δy, ", [min=", Δz_min, ", max=", Δz_max,"])",)
end

Adapt.adapt_structure(to, grid::VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ,
                                       typeof(grid.xᶠᵃᵃ),
                                       typeof(Adapt.adapt(to, grid.zᵃᵃᶠ)),
                                       Nothing}(
        nothing,
        grid.Nx, grid.Ny, grid.Nz,
        grid.Hx, grid.Hy, grid.Hz,
        grid.Lx, grid.Ly, grid.Lz,
        grid.Δx, grid.Δy,
        Adapt.adapt(to, grid.Δzᵃᵃᶜ),
        Adapt.adapt(to, grid.Δzᵃᵃᶠ),
        grid.xᶜᵃᵃ, grid.yᵃᶜᵃ,
        Adapt.adapt(to, grid.zᵃᵃᶜ),
        grid.xᶠᵃᵃ, grid.yᵃᶠᵃ,
        Adapt.adapt(to, grid.zᵃᵃᶠ))

#####
##### Should merge with grid_utils.jl at some point
#####

@inline xnode(::Center, i, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.xᶜᵃᵃ[i]
@inline xnode(::Face, i, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.xᶠᵃᵃ[i]

@inline ynode(::Center, j, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.yᵃᶜᵃ[j]
@inline ynode(::Face, j, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.yᵃᶠᵃ[j]

@inline znode(::Center, k, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.zᵃᵃᶜ[k]
@inline znode(::Face, k, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.zᵃᵃᶠ[k]

all_x_nodes(::Type{Center}, grid::VerticallyStretchedRectilinearGrid) = grid.xᶜᵃᵃ
all_x_nodes(::Type{Face}, grid::VerticallyStretchedRectilinearGrid) = grid.xᶠᵃᵃ
all_y_nodes(::Type{Center}, grid::VerticallyStretchedRectilinearGrid) = grid.yᵃᶜᵃ
all_y_nodes(::Type{Face}, grid::VerticallyStretchedRectilinearGrid) = grid.yᵃᶠᵃ
all_z_nodes(::Type{Center}, grid::VerticallyStretchedRectilinearGrid) = grid.zᵃᵃᶜ
all_z_nodes(::Type{Face}, grid::VerticallyStretchedRectilinearGrid) = grid.zᵃᵃᶠ

#
# Get minima of grid
#

min_Δx(grid::VerticallyStretchedRectilinearGrid) = grid.Δx
min_Δy(grid::VerticallyStretchedRectilinearGrid) = grid.Δy
min_Δz(grid::VerticallyStretchedRectilinearGrid) = minimum(view(grid.Δzᵃᵃᶜ, 1:grid.Nz))
