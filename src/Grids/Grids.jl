module Grids

export Center, Face
export AbstractTopology, Periodic, Bounded, Flat, Connected, topology
export AbstractGrid, AbstractPrimaryGrid, halo_size
export AbstractRectilinearGrid, RegularRectilinearGrid, VerticallyStretchedRectilinearGrid
export AbstractCurvilinearGrid, AbstractHorizontallyCurvilinearGrid
export RegularLatitudeLongitudeGrid, ConformalCubedSphereFaceGrid, ConformalCubedSphereGrid
export node, xnode, ynode, znode, xnodes, ynodes, znodes, nodes
export offset_data, new_data

using CUDA
using Adapt
using OffsetArrays

using Oceananigans
using Oceananigans.Architectures

import Base: size, length, eltype, show
import Oceananigans: short_show

#####
##### Abstract types
#####

"""
    Center

A type describing the location at the center of a grid cell.
"""
struct Center end

"""
	Face

A type describing the location at the face of a grid cell.
"""
struct Face end

"""
    AbstractTopology

Abstract supertype for grid topologies.
"""
abstract type AbstractTopology end

"""
    Periodic

Grid topology for periodic dimensions.
"""
struct Periodic <: AbstractTopology end

"""
    Bounded

Grid topology for bounded dimensions. These could be wall-bounded dimensions
or dimensions
"""
struct Bounded <: AbstractTopology end

"""
    Flat

Grid topology for flat dimensions, generally with one grid point, along which the solution
is uniform and does not vary.
"""
struct Flat <: AbstractTopology end

"""
    Connected

Grid topology for dimensions that are connected to other models or domains on both sides.
"""
const Connected = Periodic  # Right now we just need them to behave like Periodic dimensions except we change the boundary conditions.

"""
    AbstractGrid{FT, TX, TY, TZ}

Abstract supertype for grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractGrid{FT, TX, TY, TZ} end

"""
    AbstractPrimaryGrid{FT, TX, TY, TZ}

Abstract supertype for "primary" grids (as opposed to grids with immersed boundaries)
with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractPrimaryGrid{FT, TX, TY, TZ} <: AbstractGrid{FT, TX, TY, TZ} end

"""
    AbstractRectilinearGrid{FT, TX, TY, TZ}

Abstract supertype for rectilinear grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractRectilinearGrid{FT, TX, TY, TZ} <: AbstractPrimaryGrid{FT, TX, TY, TZ} end

"""
    AbstractCurvilinearGrid{FT, TX, TY, TZ}

Abstract supertype for curvilinear grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractCurvilinearGrid{FT, TX, TY, TZ} <: AbstractPrimaryGrid{FT, TX, TY, TZ} end

"""
    AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ}

Abstract supertype for horizontally-curvilinear grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ} <: AbstractCurvilinearGrid{FT, TX, TY, TZ} end

Base.eltype(::AbstractGrid{FT}) where FT = FT
Base.size(grid::AbstractGrid) = (grid.Nx, grid.Ny, grid.Nz)
Base.length(grid::AbstractGrid) = (grid.Lx, grid.Ly, grid.Lz)

halo_size(grid) = (grid.Hx, grid.Hy, grid.Hz)

topology(::AbstractGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = (TX, TY, TZ)
topology(grid, dim) = topology(grid)[dim]

include("grid_utils.jl")
include("zeros.jl")
include("new_data.jl")
include("automatic_halo_sizing.jl")
include("input_validation.jl")
include("regular_rectilinear_grid.jl")
include("vertically_stretched_rectilinear_grid.jl")
include("regular_latitude_longitude_grid.jl")
include("conformal_cubed_sphere_face_grid.jl")
include("grid_dimensions.jl")

end
