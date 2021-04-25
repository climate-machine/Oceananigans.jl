using Oceananigans.Grids: Center, Face

"""
Notes:
This file defines grid lengths, areas, and volumes for staggered structured grids.
Each "reference cell" is associated with an index i, j, k.
The "location" of each reference cell is roughly the geometric centroid of the reference cell.
On the staggered grid, there are 7 cells additional to the "reference cell"
that are staggered with respect to the reference cell in x, y, and/or z.

The staggering is denoted by the locations "Center" and "Face":
    - "Center" is shared with the reference cell;
    - "Face" lies between reference cell centers, roughly at the interface between
      reference cells.

The three-dimensional location of an object is defined by a 3-tuple of locations, and
denoted by a triplet of superscripts. For example, an object `φ` whose cell is located at
(Center, Center, Face) is denoted `φᶜᶜᶠ`. `ᶜᶜᶠ` is Centered in `x`, `Centered` in `y`, and on
reference cell interfaces in `z` (this is where the vertical velocity is located, for example).
The super script `ᵃ` denotes "any" location.

The operators in this file fall into three categories:

1. Operators needed for an algorithm valid on rectilinear grids with
   at most a stretched vertical dimension and regular horizontal dimensions.
2. Operators needed for an algorithm on a grid that is curvilinear in the horizontal
   at rectilinear (possibly stretched) in the vertical.
"""

#####
##### Grid lengths for horizontally-regular algorithms
#####

@inline Δx(i, j, k, grid::ARG) = grid.Δx
@inline Δy(i, j, k, grid::ARG) = grid.Δy

#=
#FJP: do we need to define ΔxC and ΔxF?  Need to think about this more
@inline ΔxC(i, j, k, grid::StretchedRectilinearGrid) = @inbounds grid.Δxᶠᵃᵃ[i]
@inline ΔxF(i, j, k, grid::StretchedRectilinearGrid) = @inbounds grid.Δxᶜᵃᵃ[i]
@inline Δxᶠᵃᵃ(i, j, k, grid::StretchedRectilinearGrid) = @inbounds grid.Δxᶠᵃᵃ[i]
@inline Δxᶜᵃᵃ(i, j, k, grid::StretchedRectilinearGrid) = @inbounds grid.Δxᶜᵃᵃ[i]
=#

@inline ΔzC(i, j, k, grid::RegularRectilinearGrid) = grid.Δz
@inline ΔzC(i, j, k, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.Δzᵃᵃᶠ[k]

@inline ΔzF(i, j, k, grid::RegularRectilinearGrid) = grid.Δz
@inline ΔzF(i, j, k, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.Δzᵃᵃᶜ[k]

@inline Δzᵃᵃᶠ(i, j, k, grid::RegularRectilinearGrid) = grid.Δz
@inline Δzᵃᵃᶠ(i, j, k, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.Δzᵃᵃᶠ[k]

@inline Δzᵃᵃᶜ(i, j, k, grid::RegularRectilinearGrid) = grid.Δz
@inline Δzᵃᵃᶜ(i, j, k, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.Δzᵃᵃᶜ[k]

#####
##### "Spacings" in Flat directions. Here we dispatch to `one`. This abuse of notation
##### makes volumes correct as we want to multiply by 1, and avoids issues with
##### derivatives such as those involved in the pressure correction step.
#####

using Oceananigans.Grids: Flat

@inline Δx(   i, j, k, grid::RegularRectilinearGrid{FT, Flat})         where FT           = one(FT)
@inline Δy(   i, j, k, grid::RegularRectilinearGrid{FT, TX, Flat})     where {FT, TX}     = one(FT)
@inline ΔzC(  i, j, k, grid::RegularRectilinearGrid{FT, TX, TY, Flat}) where {FT, TX, TY} = one(FT)
@inline ΔzF(  i, j, k, grid::RegularRectilinearGrid{FT, TX, TY, Flat}) where {FT, TX, TY} = one(FT)
@inline Δzᵃᵃᶠ(i, j, k, grid::RegularRectilinearGrid{FT, TX, TY, Flat}) where {FT, TX, TY} = one(FT)
@inline Δzᵃᵃᶜ(i, j, k, grid::RegularRectilinearGrid{FT, TX, TY, Flat}) where {FT, TX, TY} = one(FT)

#####
##### Areas for horizontally-regular algorithms
#####

@inline Axᵃᵃᶜ(i, j, k, grid) = Δy(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Axᵃᵃᶠ(i, j, k, grid) = Δy(i, j, k, grid) * ΔzC(i, j, k, grid)

@inline Ayᵃᵃᶜ(i, j, k, grid) = Δx(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Ayᵃᵃᶠ(i, j, k, grid) = Δx(i, j, k, grid) * ΔzC(i, j, k, grid)

@inline Azᵃᵃᵃ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)

#####
##### Volumes for horizontally-regular algorithms
#####

@inline Vᵃᵃᶜ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Vᵃᵃᶠ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzC(i, j, k, grid)

#####
##### Grid lengths for horizontally-curvilinear, vertically-rectilinear algorithms
#####

@inline Δxᶜᶜᵃ(i, j, k, grid::ARG) = Δx(i, j, k, grid)
@inline Δxᶜᶠᵃ(i, j, k, grid::ARG) = Δx(i, j, k, grid)
@inline Δxᶠᶠᵃ(i, j, k, grid::ARG) = Δx(i, j, k, grid)
@inline Δxᶠᶜᵃ(i, j, k, grid::ARG) = Δx(i, j, k, grid)

@inline Δyᶜᶜᵃ(i, j, k, grid::ARG) = Δy(i, j, k, grid)
@inline Δyᶠᶜᵃ(i, j, k, grid::ARG) = Δy(i, j, k, grid)
@inline Δyᶜᶠᵃ(i, j, k, grid::ARG) = Δy(i, j, k, grid)
@inline Δyᶠᶠᵃ(i, j, k, grid::ARG) = Δy(i, j, k, grid)

#####
##### Areas for algorithms that generalize to horizontally-curvilinear, vertically-rectilinear grids
#####

@inline Azᶜᶜᵃ(i, j, k, grid::ARG) = Δx(i, j, k, grid) * Δy(i, j, k, grid)
@inline Azᶠᶠᵃ(i, j, k, grid::ARG) = Δx(i, j, k, grid) * Δy(i, j, k, grid)
@inline Azᶜᶠᵃ(i, j, k, grid::ARG) = Δx(i, j, k, grid) * Δy(i, j, k, grid)
@inline Azᶠᶜᵃ(i, j, k, grid::ARG) = Δx(i, j, k, grid) * Δy(i, j, k, grid)

#####
##### Areas for three-dimensionally curvilinear algorithms
#####

@inline Axᶜᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # c
@inline Axᶠᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶠᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # u
@inline Axᶠᶠᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶠᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # ζ
@inline Axᶜᶠᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶜᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # η
@inline Axᶠᶜᶠ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶠᶜᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid) # η
@inline Axᶜᶜᶠ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid) # η

@inline Ayᶜᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # c
@inline Ayᶜᶠᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶜᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # v
@inline Ayᶠᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶠᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # v
@inline Ayᶠᶠᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶠᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # ζ
@inline Ayᶜᶠᶠ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶜᶠᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid) # ξ
@inline Ayᶜᶜᶠ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid) # ξ

#####
##### Volumes for three-dimensionally curvilinear algorithms
#####

@inline Vᶜᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Azᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
@inline Vᶠᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Azᶠᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
@inline Vᶜᶠᶜ(i, j, k, grid::Union{ARG, AHCG}) = Azᶜᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
@inline Vᶜᶜᶠ(i, j, k, grid::Union{ARG, AHCG}) = Azᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid)

#####
##### Temporary place for grid spacings and areas for RegularLatitudeLongitudeGrid
#####

@inline hack_cosd(φ) = cos(π * φ / 180)
@inline hack_sind(φ) = sin(π * φ / 180)

@inline Δxᶜᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλ)
@inline Δxᶠᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλ)
@inline Δxᶜᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Δxᶠᶜᵃ(i, j, k, grid)
@inline Δxᶠᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Δxᶜᶠᵃ(i, j, k, grid)

@inline Δyᶜᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.radius * deg2rad(grid.Δφ)
@inline Δyᶠᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Δyᶜᶠᵃ(i, j, k, grid)
@inline Δyᶜᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Δyᶜᶠᵃ(i, j, k, grid)
@inline Δyᶠᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Δyᶜᶠᵃ(i, j, k, grid)

@inline Δzᵃᵃᶜ(i, j, k, grid::RegularLatitudeLongitudeGrid) = grid.Δz
@inline Δzᵃᵃᶠ(i, j, k, grid::RegularLatitudeLongitudeGrid) = grid.Δz

@inline Azᶜᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.radius^2 * deg2rad(grid.Δλ) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶠᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.radius^2 * deg2rad(grid.Δλ) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Azᶜᶜᵃ(i, j, k, grid)
@inline Azᶜᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Azᶠᶠᵃ(i, j, k, grid)

#####
##### Temporary place for grid spacings and areas for ConformalCubedSphereFaceGrid
#####

@inline Δxᶜᶜᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δxᶜᶜᵃ[i, j]
@inline Δxᶠᶜᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δxᶠᶜᵃ[i, j]
@inline Δxᶜᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δxᶜᶠᵃ[i, j]
@inline Δxᶠᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δxᶠᶠᵃ[i, j]

@inline Δyᶜᶜᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δyᶜᶜᵃ[i, j]
@inline Δyᶠᶜᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δyᶠᶜᵃ[i, j]
@inline Δyᶜᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δyᶜᶠᵃ[i, j]
@inline Δyᶠᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δyᶠᶠᵃ[i, j]

@inline Δzᵃᵃᶜ(i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.Δz
@inline Δzᵃᵃᶠ(i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.Δz

@inline Azᶜᶜᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Azᶜᶜᵃ[i, j]
@inline Azᶠᶜᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Azᶠᶜᵃ[i, j]
@inline Azᶜᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Azᶜᶠᵃ[i, j]
@inline Azᶠᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Azᶠᶠᵃ[i, j]

#####
##### Generic functions for specified locations
#####
##### For example, Δx(i, j, k, Face, Center, LZ) is equivalent to = Δxᶠᶜᵃ(i, j, k, grid).
#####
##### We also use the function "volume" rather than `V`.
#####

location_code_xy(LX, LY) = Symbol(interpolation_code(LX), interpolation_code(LY), :ᵃ)
location_code(LX, LY, LZ) = Symbol(interpolation_code(LX), interpolation_code(LY), interpolation_code(LZ))

for LX in (:Center, :Face)
    for LY in (:Center, :Face)
        LXe = @eval $LX
        LYe = @eval $LY

        Ax_function = Symbol(:Ax, location_code(LXe, LYe, Center()))
        Ay_function = Symbol(:Ay, location_code(LXe, LYe, Center()))
        Az_function = Symbol(:Az, location_code_xy(LXe, LYe))

        @eval begin
            Az(i, j, k, grid, ::$LX, ::$LY, LZ) = $Az_function(i, j, k, grid)
            Ax(i, j, k, grid, ::$LX, ::$LY, ::Center) = $Ax_function(i, j, k, grid)
            Ay(i, j, k, grid, ::$LX, ::$LY, ::Center) = $Ay_function(i, j, k, grid)
        end
    end
end

Ax(i, j, k, grid, ::Face, ::Center, ::Face) = Axᶠᶜᶠ(i, j, k, grid)
Ay(i, j, k, grid, ::Center, ::Face, ::Face) = Ayᶜᶠᶠ(i, j, k, grid)

volume(i, j, k, grid, ::Center, ::Center, ::Center) = Vᶜᶜᶜ(i, j, k, grid)
volume(i, j, k, grid, ::Face,   ::Center, ::Center) = Vᶠᶜᶜ(i, j, k, grid)
volume(i, j, k, grid, ::Center, ::Face,   ::Center) = Vᶜᶠᶜ(i, j, k, grid)
volume(i, j, k, grid, ::Center, ::Center, ::Face)   = Vᶜᶜᶠ(i, j, k, grid)
