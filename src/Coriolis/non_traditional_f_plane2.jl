"""
    NonTraditionalFPlane{FT} <: AbstractRotation

A Coriolis implementation that accounts for both the locally vertical and
the locally horizontal components of the rotation vector. Traditionally
(see [`FPlane`](@ref)) only the locally vertical component is accounted for.
"""
struct NonTraditionalFPlane{FT} <: AbstractRotation
    fz :: FT
    fy :: FT
end

"""
    NonTraditionalFPlane([FT=Float64;] fz=nothing, fy=nothing,
                                       rotation_rate=Ω_Earth, latitude=nothing)

Returns a parameter object for constant rotation about an axis in the `y-z` plane
with `y`- and `z`-components `fy/2` and `fz/2`, and the background vorticity is
`(0, fy, fz)`.

In oceanography `fz` and `fy` represent the components of planetary voriticity which
are perpendicular and parallel to the ocean surface in a domain in which `x, y, z`
correspond to the directions east, north, and up.

If `fz` and `fy` are not specified, they are calculated from `rotation_rate` and `latitude`
according to the relations `fz = 2 * rotation_rate * sind(latitude)` and
`fy = 2 * rotation_rate * cosd(latitude)`, respectively. By default, `rotation_rate`
is assumed to be Earth's.
"""
function NonTraditionalFPlane(FT=Float64; fz=nothing, fy=nothing, rotation_rate=Ω_Earth, latitude=nothing)

    use_f = !isnothing(fz) && !isnothing(fy) && isnothing(latitude)
    use_planet_parameters = !isnothing(latitude) && isnothing(fz) && isnothing(fy)

    if !xor(use_f, use_planet_parameters)
        throw(ArgumentError("Either both rotation_rate and latitude must be " *
                            "specified, *or* both f and f′ must be specified."))
    elseif use_planet_parameters
        fz = 2rotation_rate*sind(latitude)
        fy = 2rotation_rate*cosd(latitude)
    end
    return NonTraditionalFPlane{FT}(fz, fy)
end

# This function is eventually interpolated to fcc to contribute to x_f_cross_U.
@inline fʸw_minus_fᶻv(i, j, k, grid, coriolis, U) =
    coriolis.fy * ℑzᵃᵃᶜ(i, j, k, grid, U.w) - coriolis.fz * ℑyᵃᶜᵃ(i, j, k, grid, U.v)

@inline fᶻu_minus_fˣw(i, j, k, grid, coriolis, U) =
    coriolis.fz * ℑxᶜᵃᵃ(i, j, k, grid, U.u) - coriolis.fx * ℑzᵃᵃᶜ(i, j, k, grid, U.w)

@inline fˣv_minus_fʸu(i, j, k, grid, coriolis, U) =
    coriolis.fx * ℑyᵃᶜᵃ(i, j, k, grid, U.v) - coriolis.fy * ℑxᶜᵃᵃ(i, j, k, grid, U.u)

@inline x_f_cross_U(i, j, k, grid, coriolis::NonTraditionalFPlane, U) = ℑxᶠᵃᵃ(i, j, k, grid, fʸw_minus_fᶻv, coriolis, U)
@inline y_f_cross_U(i, j, k, grid, coriolis::NonTraditionalFPlane, U) = ℑyᵃᶠᵃ(i, j, k, grid, fᶻu_minus_fˣw, coriolis, U)
@inline z_f_cross_U(i, j, k, grid, coriolis::NonTraditionalFPlane, U) = ℑzᵃᵃᶠ(i, j, k, grid, fˣv_minus_fʸu, coriolis, U)

Base.show(io::IO, f_plane::NonTraditionalFPlane{FT}) where FT =
    print(io, "NonTraditionalFPlane{$FT}: ", @sprintf("fz = %.2e, fy = %.2e", f_plane.fz, f_plane.fy))
