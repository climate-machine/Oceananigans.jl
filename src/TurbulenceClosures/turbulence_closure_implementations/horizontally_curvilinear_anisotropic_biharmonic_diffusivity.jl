"""
    HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity{N, K}

Holds viscosity and diffusivities for models with prescribed isotropic diffusivities.
"""
struct HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity{NH, NZ, KH, KZ} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization}
    νh :: NH
    νz :: NZ
    κh :: KH
    κz :: KZ
end

const HCABD = HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity

"""
    HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity([FT=Float64;] νh=0, κh=0, νz=0, κz=0)

Returns parameters for an anisotropic biharmonic diffusivity model on curvilinear grids.

Keyword args
============

    * `νh`: Horizontal viscosity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`.

    * `νz`: Vertical viscosity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`.

    * `κh`: Horizontal diffusivity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`, or
            `NamedTuple` of diffusivities with entries for each tracer.

    * `κz`: Vertical diffusivity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`, or
            `NamedTuple` of diffusivities with entries for each tracer.
"""
function HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity(FT=Float64; νh=0, κh=0, νz=nothing, κz=nothing)
    νh = convert_diffusivity(FT, νh)
    νz = convert_diffusivity(FT, νz)
    κh = convert_diffusivity(FT, κh)
    κz = convert_diffusivity(FT, κz)
    return HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity(νh, νz, κh, κz)
end

function with_tracers(tracers, closure::HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity)
    κh = tracer_diffusivities(tracers, closure.κh)
    κz = tracer_diffusivities(tracers, closure.κz)
    return HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity(closure.νh, closure.νz, κh, κz)
end

calculate_diffusivities!(K, arch, grid, closure::HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity, args...) = nothing

const NoVerticalViscHCABD = HCABD{<:Any, <:Nothing}
const NoVerticalDiffHCABD = HCABD{<:Any, <:Any, <:Any, <:Nothing}

@inline viscous_flux_ux(i, j, k, grid, closure::HCABD, clock, U, args...) = - ν_δ★ᶜᶜᶜ(i, j, k, grid, clock, closure.νh, U.u, U.v)   
@inline viscous_flux_uy(i, j, k, grid, closure::HCABD, clock, U, args...) = + ν_ζ★ᶠᶠᶜ(i, j, k, grid, clock, closure.νh, U.u, U.v)   
@inline viscous_flux_uz(i, j, k, grid, closure::HCABD, clock, U, args...) = - ν_uzzzᶠᶜᶠ(i, j, k, grid, clock, closure.νz, U.u)

@inline viscous_flux_vx(i, j, k, grid, closure::HCABD, clock, U, args...) = - ν_ζ★ᶠᶠᶜ(i, j, k, grid, clock, closure.νh, U.u, U.v)
@inline viscous_flux_vy(i, j, k, grid, closure::HCABD, clock, U, args...) = - ν_δ★ᶜᶜᶜ(i, j, k, grid, clock, closure.νh, U.u, U.v)
@inline viscous_flux_vz(i, j, k, grid, closure::HCABD, clock, U, args...) = - ν_uzzzᶠᶜᶠ(i, j, k, grid, clock, closure.νz, U.v)

@inline viscous_flux_uz(i, j, k, grid, closure::NoVerticalViscHCABD, clock, U, args...) = zero(eltype(grid))
@inline viscous_flux_vz(i, j, k, grid, closure::NoVerticalViscHCABD, clock, U, args...) = zero(eltype(grid))

@inline function diffusive_flux_x(i, j, k, grid, closure::HCABD, c, ::Val{tracer_index}, clock, args...) where tracer_index
    @inbounds κh = closure.κh[tracer_index]
    return - κᶠᶜᶜ(i, j, k, grid, clock, κh) * ∂x_∇²h_cᶠᶜᶜ(i, j, k, grid, c)
end

@inline function diffusive_flux_y(i, j, k, grid, closure::HCABD, c, ::Val{tracer_index}, clock, args...) where tracer_index
    @inbounds κh = closure.κh[tracer_index]
    return - κᶜᶠᶜ(i, j, k, grid, clock, κh) * ∂y_∇²h_cᶜᶠᶜ(i, j, k, grid, c)
end

@inline function diffusive_flux_z(i, j, k, grid, closure::HCABD, c, ::Val{tracer_index}, clock, args...) where tracer_index
    @inbounds κz = closure.κz[tracer_index]
    return - κᶜᶜᶠ(i, j, k, grid, clock, κz) * ∂³zᵃᵃᶠ(i, j, k, grid, c)
end

@inline diffusive_flux_z(i, j, k, grid, closure::NoVerticalDiffHCABD, args...) = zero(eltype(grid))

Base.show(io::IO, closure::HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity) =
    print(io, "HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity: " *
              "(νh=$(closure.νh), νz=$(closure.νz)), " *
              "(κh=$(closure.κh), κz=$(closure.κz))")
