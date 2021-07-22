using Oceananigans.Operators: Δy_uᶠᶜᵃ, Δx_vᶜᶠᵃ, Δx_uᶠᶜᵃ, Δy_vᶜᶠᵃ, ℑxyᶠᶠᵃ, ℑxzᶠᵃᶠ, ℑyzᵃᶠᶠ
using Oceananigans.BoundaryConditions: BoundaryCondition

const UVW = NamedTuple{(:u, :v, :w)}

# Throw away immersed boundary condition when calculating "intrinsic" fluxes
@inline _viscous_flux_ux(i, j, k, grid, disc, closure, immersed_bcs::UVW, args...) = viscous_flux_ux(i, j, k, grid, disc, closure, args...)
@inline _viscous_flux_uy(i, j, k, grid, disc, closure, immersed_bcs::UVW, args...) = viscous_flux_uy(i, j, k, grid, disc, closure, args...)
@inline _viscous_flux_uz(i, j, k, grid, disc, closure, immersed_bcs::UVW, args...) = viscous_flux_uz(i, j, k, grid, disc, closure, args...)
@inline _viscous_flux_vx(i, j, k, grid, disc, closure, immersed_bcs::UVW, args...) = viscous_flux_vx(i, j, k, grid, disc, closure, args...)
@inline _viscous_flux_vy(i, j, k, grid, disc, closure, immersed_bcs::UVW, args...) = viscous_flux_vy(i, j, k, grid, disc, closure, args...)
@inline _viscous_flux_vz(i, j, k, grid, disc, closure, immersed_bcs::UVW, args...) = viscous_flux_vz(i, j, k, grid, disc, closure, args...)
@inline _viscous_flux_wx(i, j, k, grid, disc, closure, immersed_bcs::UVW, args...) = viscous_flux_wx(i, j, k, grid, disc, closure, args...)
@inline _viscous_flux_wy(i, j, k, grid, disc, closure, immersed_bcs::UVW, args...) = viscous_flux_wy(i, j, k, grid, disc, closure, args...)
@inline _viscous_flux_wz(i, j, k, grid, disc, closure, immersed_bcs::UVW, args...) = viscous_flux_wz(i, j, k, grid, disc, closure, args...)

# Functions for models that do not support immersed boundaries
@inline _viscous_flux_ux(args...) = viscous_flux_ux(args...)
@inline _viscous_flux_uy(args...) = viscous_flux_uy(args...)
@inline _viscous_flux_uz(args...) = viscous_flux_uz(args...)
@inline _viscous_flux_vx(args...) = viscous_flux_vx(args...)
@inline _viscous_flux_vy(args...) = viscous_flux_vy(args...)
@inline _viscous_flux_vz(args...) = viscous_flux_vz(args...)
@inline _viscous_flux_wx(args...) = viscous_flux_wx(args...)
@inline _viscous_flux_wy(args...) = viscous_flux_wy(args...)
@inline _viscous_flux_wz(args...) = viscous_flux_wz(args...)

#####                                                            
##### Viscous flux divergences
#####

@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_cᶜᶜᶜ, _viscous_flux_ux, disc, closure, args...) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_ζᶠᶠᶜ, _viscous_flux_uy, disc, closure, args...) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_ηᶠᶜᵃ, _viscous_flux_uz, disc, closure, args...))
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_ζᶠᶠᶜ, _viscous_flux_vx, disc, closure, args...) +
                                    δyᵃᶠᵃ(i, j, k, grid, Ay_cᶜᶜᶜ, _viscous_flux_vy, disc, closure, args...) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_ξᶜᶠᵃ, _viscous_flux_vz, disc, closure, args...))
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1/Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_ηᶠᶜᶠ, _viscous_flux_wx, disc, closure, args...) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_ξᶜᶠᶠ, _viscous_flux_wy, disc, closure, args...) +
                                    δzᵃᵃᶠ(i, j, k, grid, Az_cᶜᶜᵃ, _viscous_flux_wz, disc, closure, args...))
end

#####
##### Viscosities at different locations
#####

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::Number) = ν

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::AbstractArray) = @inbounds ν[i, j, k]
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::AbstractArray) = @inbounds ℑxyᶠᶠᵃ(i, j, k, grid, ν)
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::AbstractArray) = @inbounds ℑxzᶠᵃᶠ(i, j, k, grid, ν)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::AbstractArray) = @inbounds ℑyzᵃᶠᶠ(i, j, k, grid, ν)

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::F) where F<:Function = ν(xnode(Center(), i, grid), ynode(Center(), j, grid), znode(Center(), k, grid), clock.time)
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::F) where F<:Function = ν(xnode(Face(),   i, grid), ynode(Face(),   j, grid), znode(Center(), k, grid), clock.time)
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::F) where F<:Function = ν(xnode(Face(),   i, grid), ynode(Center(), j, grid), znode(Face(),   k, grid), clock.time)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::F) where F<:Function = ν(xnode(Center(), i, grid), ynode(Face(),   j, grid), znode(Face(),   k, grid), clock.time)

#####
##### Convenience "vanilla" viscous flux functions
#####

@inline viscous_flux_ux(i, j, k, grid, clock, ν, u) = - νᶜᶜᶜ(i, j, k, grid, clock, ν) * ∂xᶜᶜᵃ(i, j, k, grid, u)
@inline viscous_flux_uy(i, j, k, grid, clock, ν, u) = - νᶠᶠᶜ(i, j, k, grid, clock, ν) * ∂yᶠᶠᵃ(i, j, k, grid, u)
@inline viscous_flux_uz(i, j, k, grid, clock, ν, u) = - νᶠᶜᶠ(i, j, k, grid, clock, ν) * ∂zᵃᵃᶠ(i, j, k, grid, u)

@inline viscous_flux_vx(i, j, k, grid, clock, ν, v) = - νᶠᶠᶜ(i, j, k, grid, clock, ν) * ∂xᶠᶠᵃ(i, j, k, grid, v)
@inline viscous_flux_vy(i, j, k, grid, clock, ν, v) = - νᶜᶜᶜ(i, j, k, grid, clock, ν) * ∂yᶜᶜᵃ(i, j, k, grid, v)
@inline viscous_flux_vz(i, j, k, grid, clock, ν, v) = - νᶜᶠᶠ(i, j, k, grid, clock, ν) * ∂zᵃᵃᶠ(i, j, k, grid, v)

@inline viscous_flux_wx(i, j, k, grid, clock, ν, w) = - νᶠᶜᶠ(i, j, k, grid, clock, ν) * ∂xᶠᶜᵃ(i, j, k, grid, w)
@inline viscous_flux_wy(i, j, k, grid, clock, ν, w) = - νᶜᶠᶠ(i, j, k, grid, clock, ν) * ∂yᶜᶠᵃ(i, j, k, grid, w)
@inline viscous_flux_wz(i, j, k, grid, clock, ν, w) = - νᶜᶜᶜ(i, j, k, grid, clock, ν) * ∂zᵃᵃᶜ(i, j, k, grid, w)

#####
##### Products of viscosity and divergence, vorticity, vertical momentum gradients
#####

@inline ν_δᶜᶜᶜ(i, j, k, grid, clock, ν, u, v) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * div_xyᶜᶜᵃ(i, j, k, grid, u, v)
@inline ν_ζᶠᶠᶜ(i, j, k, grid, clock, ν, u, v) = νᶠᶠᶜ(i, j, k, grid, clock, ν) * ζ₃ᶠᶠᵃ(i, j, k, grid, u, v)

@inline ν_uzᶠᶜᶠ(i, j, k, grid, clock, ν, u) = νᶠᶜᶠ(i, j, k, grid, clock, ν) * ∂zᵃᵃᶠ(i, j, k, grid, u)
@inline ν_vzᶜᶠᶠ(i, j, k, grid, clock, ν, v) = νᶜᶠᶠ(i, j, k, grid, clock, ν) * ∂zᵃᵃᶠ(i, j, k, grid, v)

@inline ν_uzzzᶠᶜᶠ(i, j, k, grid, clock, ν, u) = νᶠᶜᶠ(i, j, k, grid, clock, ν) * ∂³zᵃᵃᶠ(i, j, k, grid, u)
@inline ν_vzzzᶜᶠᶠ(i, j, k, grid, clock, ν, v) = νᶜᶠᶠ(i, j, k, grid, clock, ν) * ∂³zᵃᵃᶠ(i, j, k, grid, v)

# See https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-dissipation
@inline function δ★ᶜᶜᶜ(i, j, k, grid, u, v)

    # These closures seem to be needed to help the compiler infer types
    # (either of u and v or of the function arguments)
    @inline Δy_∇²u(i, j, k, grid, u) = Δy_uᶠᶜᵃ(i, j, k, grid, ∇²hᶠᶜᶜ, u)
    @inline Δx_∇²v(i, j, k, grid, v) = Δx_vᶜᶠᵃ(i, j, k, grid, ∇²hᶜᶠᶜ, v)

    return 1 / Azᶜᶜᵃ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, Δy_∇²u, u) +
                                       δyᵃᶜᵃ(i, j, k, Δx_∇²v, v))
end

@inline function ζ★ᶠᶠᶜ(i, j, k, grid, u, v)

    # These closures seem to be needed to help the compiler infer types
    # (either of u and v or of the function arguments)
    @inline Δy_∇²v(i, j, k, grid, v) = Δy_vᶜᶠᵃ(i, j, k, grid, ∇²hᶜᶠᶜ, v)
    @inline Δx_∇²u(i, j, k, grid, u) = Δx_uᶠᶜᵃ(i, j, k, grid, ∇²hᶠᶜᶜ, u)

    return 1 / Azᶠᶠᵃ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, Δy_∇²v, v) -
                                       δyᵃᶠᵃ(i, j, k, Δx_∇²u, u))
end

@inline ν_δ★ᶜᶜᶜ(i, j, k, grid, clock, ν, u, v) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * δ★ᶜᶜᶜ(i, j, k, grid, u, v)
@inline ν_ζ★ᶠᶠᶜ(i, j, k, grid, clock, ν, u, v) = νᶠᶠᶜ(i, j, k, grid, clock, ν) * ζ★ᶠᶠᶜ(i, j, k, grid, u, v)
