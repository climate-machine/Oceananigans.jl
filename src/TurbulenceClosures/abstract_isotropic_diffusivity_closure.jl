"""
    abstract type AbstractIsotropicDiffusivity <: AbstractTurbulenceClosure end

Abstract type for closures with *isotropic* diffusivities.
"""
abstract type AbstractIsotropicDiffusivity{TD} <: AbstractTurbulenceClosure{TD} end

"""
    viscosity(closure, diffusivities)

Returns the isotropic viscosity associated with `closure`.
"""
function viscosity end 

"""
    diffusivity(closure, diffusivities, tracer_indedx) where tracer_index

Returns the isotropic diffusivity associated with `closure` and tracer index `c_idx`.
"""
function diffusivity end 

#####
##### Viscosity-stress products
#####

"""
    ν_σᶜᶜᶜ(i, j, k, grid, ν, σᶜᶜᶜ, u, v, w)

Multiply the array `ν` located at `ᶜᶜᶜ` by a function

    `σᶜᶜᶜ(i, j, k, grid, u, v, w)`

at index `i, j, k` and location `ᶜᶜᶜ`.
"""
@inline ν_σᶜᶜᶜ(i, j, k, grid, clock, ν, σᶜᶜᶜ, args...) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * σᶜᶜᶜ(i, j, k, grid, args...)
@inline ν_σᶠᶠᶜ(i, j, k, grid, clock, ν, σᶠᶠᶜ, args...) = νᶠᶠᶜ(i, j, k, grid, clock, ν) * σᶠᶠᶜ(i, j, k, grid, args...)
@inline ν_σᶠᶜᶠ(i, j, k, grid, clock, ν, σᶠᶜᶠ, args...) = νᶠᶜᶠ(i, j, k, grid, clock, ν) * σᶠᶜᶠ(i, j, k, grid, args...)
@inline ν_σᶜᶠᶠ(i, j, k, grid, clock, ν, σᶜᶠᶠ, args...) = νᶜᶠᶠ(i, j, k, grid, clock, ν) * σᶜᶠᶠ(i, j, k, grid, args...)

#####
##### Stress divergences
#####

const AID = AbstractIsotropicDiffusivity

@inline viscous_flux_ux(i, j, k, grid, closure::AID, clock, U, K, C, buoyancy) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, K), Σ₁₁, U.u, U.v, U.w)
@inline viscous_flux_uy(i, j, k, grid, closure::AID, clock, U, K, C, buoyancy) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, K), Σ₁₂, U.u, U.v, U.w)
@inline viscous_flux_uz(i, j, k, grid, closure::AID, clock, U, K, C, buoyancy) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, K), Σ₁₃, U.u, U.v, U.w)

@inline viscous_flux_vx(i, j, k, grid, closure::AID, clock, U, K, C, buoyancy) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, K), Σ₂₁, U.u, U.v, U.w)
@inline viscous_flux_vy(i, j, k, grid, closure::AID, clock, U, K, C, buoyancy) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, K), Σ₂₂, U.u, U.v, U.w)
@inline viscous_flux_vz(i, j, k, grid, closure::AID, clock, U, K, C, buoyancy) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, K), Σ₂₃, U.u, U.v, U.w)

@inline viscous_flux_wx(i, j, k, grid, closure::AID, clock, U, K, C, buoyancy) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, K), Σ₃₁, U.u, U.v, U.w)
@inline viscous_flux_wy(i, j, k, grid, closure::AID, clock, U, K, C, buoyancy) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, K), Σ₃₂, U.u, U.v, U.w)
@inline viscous_flux_wz(i, j, k, grid, closure::AID, clock, U, K, C, buoyancy) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, K), Σ₃₃, U.u, U.v, U.w)

#####
##### Diffusive fluxes
#####

@inline diffusive_flux_x(i, j, k, grid, closure::AID, c, c_idx, clock, K, C, buoyancy, U) = diffusive_flux_x(i, j, k, grid, clock, diffusivity(closure, c_idx, K), c)
@inline diffusive_flux_y(i, j, k, grid, closure::AID, c, c_idx, clock, K, C, buoyancy, U) = diffusive_flux_y(i, j, k, grid, clock, diffusivity(closure, c_idx, K), c)
@inline diffusive_flux_z(i, j, k, grid, closure::AID, c, c_idx, clock, K, C, buoyancy, U) = diffusive_flux_z(i, j, k, grid, clock, diffusivity(closure, c_idx, K), c)

#####
##### Support for VerticallyImplicitTimeDiscretization
#####

const VITD = VerticallyImplicitTimeDiscretization
const VerticallyBoundedGrid{FT} = AbstractGrid{FT, <:Any, <:Any, <:Bounded}

@inline z_viscosity(closure::AID, args...) = viscosity(closure, args...)
@inline z_diffusivity(closure::AID, c_idx, args...) = diffusivity(closure, c_idx, args...)

@inline ivd_viscous_flux_uz(i, j, k, grid, closure, clock, U, args...) = - ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, args...), ∂xᶠᶜᵃ, U.w)
@inline ivd_viscous_flux_vz(i, j, k, grid, closure, clock, U, args...) = - ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, args...), ∂yᶜᶠᵃ, U.v)

# General functions (eg for vertically periodic)
@inline viscous_flux_uz(i, j, k, grid,  ::VITD, closure::AID, args...) = ivd_viscous_flux_uz(i, j, k, grid, closure, args...)
@inline viscous_flux_vz(i, j, k, grid,  ::VITD, closure::AID, args...) = ivd_viscous_flux_vz(i, j, k, grid, closure, args...)
@inline viscous_flux_wz(i, j, k, grid,  ::VITD, closure::AID, args...) = zero(eltype(grid))
@inline diffusive_flux_z(i, j, k, grid, ::VITD, closure::AID, clock, args...) = zero(eltype(grid))
                  
# Vertically bounded grids
#
# For vertically bounded grids, we calculate _explicit_ fluxes on the boundaries, 
# and elide the implicit vertical flux component everywhere else. This is consistent
# with the formulation of the tridiagonal solver, which requires explicit treatment
# of boundary contributions (eg boundary contributions must be moved to the right
# hand side of the tridiagonal system).
@inline function viscous_flux_uz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AID, args...)
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_uz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  ivd_viscous_flux_uz(i, j, k, grid, closure, args...))
end

@inline function viscous_flux_vz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AID, args...)
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_vz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  ivd_viscous_flux_vz(i, j, k, grid, closure, args...))
end

@inline function viscous_flux_wz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::AID, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_wz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  zero(FT))
end

@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::AID, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  zero(FT))
end
