using Oceananigans.Advection

#####
##### Legend:
##### ρ  -> density fields
##### ρũ -> momentum fields
#####  ũ -> velocity fields
##### ρc̃ -> conservative tracer fields
#####  c̃ -> tracer fields
#####  K̃ -> diffusivity fields
#####

#####
##### Convert conservative variables to primitive variables
#####

@inline U_over_ρ(i, j, k, grid, ρ, ρu) = @inbounds ρu[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, ρ)
@inline V_over_ρ(i, j, k, grid, ρ, ρv) = @inbounds ρv[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, ρ)
@inline W_over_ρ(i, j, k, grid, ρ, ρw) = @inbounds ρw[i, j, k] / ℑzᵃᵃᶠ(i, j, k, grid, ρ)
@inline C_over_ρ(i, j, k, grid, ρ, ρc) = @inbounds ρc[i, j, k] / ρ[i, j, k]

#####
##### Kinetic energy
#####

@inline kinetic_energy(i, j, k, grid::AbstractGrid{FT}, ρ, ρũ) where FT =
    @inbounds FT(0.5) * (  (ℑxᶜᵃᵃ(i, j, k, grid, ρũ.ρu) / ρ[i, j, k])^2
                         + (ℑyᵃᶜᵃ(i, j, k, grid, ρũ.ρv) / ρ[i, j, k])^2
                         + (ℑzᵃᵃᶜ(i, j, k, grid, ρũ.ρw) / ρ[i, j, k])^2)

#####
##### Pressure work
#####

@inline p∂u∂x(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃) =
       (  ℑxᶠᵃᵃ(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃)
        * Axᵃᵃᶠ(i, j, k, grid) * U_over_ρ(i, j, k, grid, ρ, ρũ.ρu))

@inline p∂v∂y(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃) =
       (  ℑyᵃᶠᵃ(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃)
        * Ayᵃᵃᶠ(i, j, k, grid) * V_over_ρ(i, j, k, grid, ρ, ρũ.ρv))

@inline p∂w∂z(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃) =
       (  ℑzᵃᵃᶠ(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃)
        * Azᵃᵃᵃ(i, j, k, grid) * W_over_ρ(i, j, k, grid, ρ, ρũ.ρw))

@inline ∂ⱼpuⱼ(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃) =
    (1/Vᵃᵃᶜ(i, j, k, grid)
        * (  δxᶜᵃᵃ(i, j, k, grid, p∂u∂x, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃)
           + δyᵃᶜᵃ(i, j, k, grid, p∂v∂y, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃)
           + δzᵃᵃᶜ(i, j, k, grid, p∂w∂z, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃)))

#####
##### Tracer advection
#####

@inline tracer_flux_ρuc(i, j, k, grid, scheme, ρ, ρu, ρc) = advective_tracer_flux_x(i, j, k, grid, scheme, ρu, ρc) / ℑxᶠᵃᵃ(i, j, k, grid, ρ)
@inline tracer_flux_ρvc(i, j, k, grid, scheme, ρ, ρv, ρc) = advective_tracer_flux_y(i, j, k, grid, scheme, ρv, ρc) / ℑyᵃᶠᵃ(i, j, k, grid, ρ)
@inline tracer_flux_ρwc(i, j, k, grid, scheme, ρ, ρw, ρc) = advective_tracer_flux_z(i, j, k, grid, scheme, ρw, ρc) / ℑzᵃᵃᶠ(i, j, k, grid, ρ)

@inline div_ρUc(i, j, k, grid, scheme, density, momenta, ρc) =
    (1/Vᵃᵃᶜ(i, j, k, grid)
        * (  δxᶜᵃᵃ(i, j, k, grid, tracer_flux_ρuc, scheme, density, momenta.ρu, ρc)
           + δyᵃᶜᵃ(i, j, k, grid, tracer_flux_ρvc, scheme, density, momenta.ρv, ρc)
           + δzᵃᵃᶜ(i, j, k, grid, tracer_flux_ρwc, scheme, density, momenta.ρw, ρc)))

#####
##### Diffusion
##### FIXME: need a better way to enforce boundary conditions on diffusive fluxes at rigid boundaries
#####

@inline diffusive_flux_x(i, j, k, grid, κᶠᶜᶜ, ρ, ρc) =
    ℑxᶠᵃᵃ(i, j, k, grid, ρ) * κᶠᶜᶜ * Axᵃᵃᶠ(i, j, k, grid) * ∂xᶠᵃᵃ(i, j, k, grid, C_over_ρ, ρ, ρc)

@inline diffusive_flux_y(i, j, k, grid, κᶜᶠᶜ, ρ, ρc) =
    ℑyᵃᶠᵃ(i, j, k, grid, ρ) * κᶜᶠᶜ * Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶠᵃ(i, j, k, grid, C_over_ρ, ρ, ρc)

@inline diffusive_flux_z(i, j, k, grid, κᶜᶜᶠ, ρ, ρc) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ) * κᶜᶜᶠ * Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, C_over_ρ, ρ, ρc)

@inline diffusive_tvar_flux_x(i, j, k, grid, κᶠᶜᶜ, tvar_diag, tracer_index, tvar, gases, gravity, ρ, ρũ, ρc̃, ρc) =
    (ℑxᶠᵃᵃ(i, j, k, grid, tvar_diag, tracer_index, tvar, gases, gravity, ρ, ρũ, ρc̃)
     * κᶠᶜᶜ * Axᵃᵃᶠ(i, j, k, grid) * ∂xᶠᵃᵃ(i, j, k, grid, C_over_ρ, ρ, ρc))

@inline diffusive_tvar_flux_y(i, j, k, grid, κᶜᶠᶜ, tvar_diag, tracer_index, tvar, gases, gravity, ρ, ρũ, ρc̃, ρc) =
    (ℑyᵃᶠᵃ(i, j, k, grid, tvar_diag, tracer_index, tvar, gases, gravity, ρ, ρũ, ρc̃)
     * κᶜᶠᶜ * Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶠᵃ(i, j, k, grid, C_over_ρ, ρ, ρc))

@inline diffusive_tvar_flux_z(i, j, k, grid, κᶜᶜᶠ, tvar_diag, tracer_index, tvar, gases, gravity, ρ, ρũ, ρc̃, ρc) =
    (ℑzᵃᵃᶠ(i, j, k, grid, tvar_diag, tracer_index, tvar, gases, gravity, ρ, ρũ, ρc̃)
     * κᶜᶜᶠ * Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, C_over_ρ, ρ, ρc))

@inline diffusive_pressure_flux_x(i, j, k, grid, κᶠᶜᶜ, p_over_ρ_diag, tvar, gases, gravity, ρ, ρũ, ρc̃) =
    (ℑxᶠᵃᵃ(i, j, k, grid, ρ) * κᶠᶜᶜ * Axᵃᵃᶠ(i, j, k, grid)
     * ∂xᶠᵃᵃ(i, j, k, grid, p_over_ρ_diag, tvar, gases, gravity, ρ, ρũ, ρc̃))

@inline diffusive_pressure_flux_y(i, j, k, grid, κᶜᶠᶜ, p_over_ρ_diag, tvar, gases, gravity, ρ, ρũ, ρc̃) =
    (ℑyᵃᶠᵃ(i, j, k, grid, ρ) * κᶜᶠᶜ * Ayᵃᵃᶠ(i, j, k, grid)
     * ∂yᵃᶠᵃ(i, j, k, grid, p_over_ρ_diag, tvar, gases, gravity, ρ, ρũ, ρc̃))

@inline function diffusive_pressure_flux_z(i, j, k, grid::AbstractGrid{FT}, κᶜᶜᶠ, p_over_ρ_diag, tvar, gases, gravity, ρ, ρũ, ρc̃) where FT
    (k <= 1 || k > grid.Nz) && return zero(FT)
    return (ℑzᵃᵃᶠ(i, j, k, grid, ρ) * κᶜᶜᶠ * Azᵃᵃᵃ(i, j, k, grid)
            * ∂zᵃᵃᶠ(i, j, k, grid, p_over_ρ_diag, tvar, gases, gravity, ρ, ρũ, ρc̃))
end

@inline function ∂ⱼDᶜⱼ(i, j, k, grid, closure::IsotropicDiffusivity, tracer_index, ρ, ρc, args...)
    @inbounds κ = closure.κ[tracer_index]
    return (1/Vᵃᵃᶜ(i, j, k, grid)
            * (  δxᶜᵃᵃ(i, j, k, grid, diffusive_flux_x, κ, ρ, ρc)
               + δyᵃᶜᵃ(i, j, k, grid, diffusive_flux_y, κ, ρ, ρc)
               + δzᵃᵃᶜ(i, j, k, grid, diffusive_flux_z, κ, ρ, ρc)))
end

@inline function ∂ⱼtᶜDᶜⱼ(i, j, k, grid, closure::IsotropicDiffusivity, tvar_diag,
                         tracer_index, tvar, gases, gravity, ρ, ρũ, ρc̃, ρc, args...)

    @inbounds κ = closure.κ[tracer_index]
    return (1/Vᵃᵃᶜ(i, j, k, grid)
               * (  δxᶜᵃᵃ(i, j, k, grid, diffusive_tvar_flux_x, κ, tvar_diag, tracer_index, tvar, gases, gravity, ρ, ρũ, ρc̃, ρc)
                  + δyᵃᶜᵃ(i, j, k, grid, diffusive_tvar_flux_y, κ, tvar_diag, tracer_index, tvar, gases, gravity, ρ, ρũ, ρc̃, ρc)
                  + δzᵃᵃᶜ(i, j, k, grid, diffusive_tvar_flux_z, κ, tvar_diag, tracer_index, tvar, gases, gravity, ρ, ρũ, ρc̃, ρc)))
end

@inline function ∂ⱼDᵖⱼ(i, j, k, grid, closure::IsotropicDiffusivity, tracer_index,
                       p_over_ρ_diag, tvar, gases, gravity, ρ, ρũ, ρc̃, args ...)

    @inbounds κ = closure.κ[tracer_index]
    return (1/Vᵃᵃᶜ(i, j, k, grid)
               * (  δxᶜᵃᵃ(i, j, k, grid, diffusive_pressure_flux_x, κ, p_over_ρ_diag, tvar, gases, gravity, ρ, ρũ, ρc̃)
                  + δyᵃᶜᵃ(i, j, k, grid, diffusive_pressure_flux_y, κ, p_over_ρ_diag, tvar, gases, gravity, ρ, ρũ, ρc̃)
                  + δzᵃᵃᶜ(i, j, k, grid, diffusive_pressure_flux_z, κ, p_over_ρ_diag, tvar, gases, gravity, ρ, ρũ, ρc̃)))
end

#####
##### Momentum advection
##### Note the convention "momentum_flux_Ua" corresponds to the advection _of_ a _by_ U.
#####

@inline momentum_flux_ρuu(i, j, k, grid, scheme, ρ, ρu) = @inbounds momentum_flux_uu(i, j, k, grid, scheme, ρu, ρu) / ρ[i, j, k]
@inline momentum_flux_ρvv(i, j, k, grid, scheme, ρ, ρv) = @inbounds momentum_flux_vv(i, j, k, grid, scheme, ρv, ρv) / ρ[i, j, k]
@inline momentum_flux_ρww(i, j, k, grid, scheme, ρ, ρw) = @inbounds momentum_flux_ww(i, j, k, grid, scheme, ρw, ρw) / ρ[i, j, k]

@inline momentum_flux_ρuv(i, j, k, grid, scheme, ρ, ρu, ρv) = momentum_flux_uv(i, j, k, grid, scheme, ρv, ρu) / ℑxyᶠᶠᵃ(i, j, k, grid, ρ)
@inline momentum_flux_ρuw(i, j, k, grid, scheme, ρ, ρu, ρw) = momentum_flux_uw(i, j, k, grid, scheme, ρw, ρu) / ℑxzᶠᵃᶠ(i, j, k, grid, ρ)
@inline momentum_flux_ρvu(i, j, k, grid, scheme, ρ, ρu, ρv) = momentum_flux_vu(i, j, k, grid, scheme, ρu, ρv) / ℑxyᶠᶠᵃ(i, j, k, grid, ρ)
@inline momentum_flux_ρvw(i, j, k, grid, scheme, ρ, ρv, ρw) = momentum_flux_vw(i, j, k, grid, scheme, ρw, ρv) / ℑyzᵃᶠᶠ(i, j, k, grid, ρ)
@inline momentum_flux_ρwu(i, j, k, grid, scheme, ρ, ρu, ρw) = momentum_flux_wu(i, j, k, grid, scheme, ρu, ρw) / ℑxzᶠᵃᶠ(i, j, k, grid, ρ)
@inline momentum_flux_ρwv(i, j, k, grid, scheme, ρ, ρv, ρw) = momentum_flux_wv(i, j, k, grid, scheme, ρv, ρw) / ℑyzᵃᶠᶠ(i, j, k, grid, ρ)

@inline div_ρuũ(i, j, k, grid, scheme, ρ, ρũ) =
    (1/Vᵃᵃᶜ(i, j, k, grid)
        * (  δxᶠᵃᵃ(i, j, k, grid, momentum_flux_ρuu, scheme, ρ, ρũ.ρu)
           + δyᵃᶜᵃ(i, j, k, grid, momentum_flux_ρuv, scheme, ρ, ρũ.ρu, ρũ.ρv)
           + δzᵃᵃᶜ(i, j, k, grid, momentum_flux_ρuw, scheme, ρ, ρũ.ρu, ρũ.ρw)))

@inline div_ρvũ(i, j, k, grid, scheme, ρ, ρũ) =
    (1/Vᵃᵃᶜ(i, j, k, grid)
        * (  δxᶜᵃᵃ(i, j, k, grid, momentum_flux_ρvu, scheme, ρ, ρũ.ρu, ρũ.ρv)
           + δyᵃᶠᵃ(i, j, k, grid, momentum_flux_ρvv, scheme, ρ, ρũ.ρv)
           + δzᵃᵃᶜ(i, j, k, grid, momentum_flux_ρvw, scheme, ρ, ρũ.ρv, ρũ.ρw)))

@inline div_ρwũ(i, j, k, grid, scheme, ρ, ρũ) =
    (1/Vᵃᵃᶠ(i, j, k, grid)
        * (  δxᶜᵃᵃ(i, j, k, grid, momentum_flux_ρwu, scheme, ρ, ρũ.ρu, ρũ.ρw)
           + δyᵃᶜᵃ(i, j, k, grid, momentum_flux_ρwv, scheme, ρ, ρũ.ρv, ρũ.ρw)
           + δzᵃᵃᶠ(i, j, k, grid, momentum_flux_ρww, scheme, ρ, ρũ.ρw)))

#####
##### Viscous dissipation
#####

@inline strain_rate_tensor_ux(i, j, k, grid::AbstractGrid{FT}, ρ, ρu) where FT = FT(1/3) * ∂xᶜᵃᵃ(i, j, k, grid, U_over_ρ, ρ, ρu)
@inline strain_rate_tensor_vy(i, j, k, grid::AbstractGrid{FT}, ρ, ρv) where FT = FT(1/3) * ∂yᵃᶜᵃ(i, j, k, grid, V_over_ρ, ρ, ρv)
@inline strain_rate_tensor_wz(i, j, k, grid::AbstractGrid{FT}, ρ, ρw) where FT = FT(1/3) * ∂zᵃᵃᶜ(i, j, k, grid, W_over_ρ, ρ, ρw)

@inline strain_rate_tensor_uy(i, j, k, grid, ρ, ρu, ρv) = ∂yᵃᶠᵃ(i, j, k, grid, U_over_ρ, ρ, ρu) + ∂xᶠᵃᵃ(i, j, k, grid, V_over_ρ, ρ, ρv)
@inline strain_rate_tensor_uz(i, j, k, grid, ρ, ρu, ρw) = ∂zᵃᵃᶠ(i, j, k, grid, U_over_ρ, ρ, ρu) + ∂xᶠᵃᵃ(i, j, k, grid, W_over_ρ, ρ, ρw)
@inline strain_rate_tensor_vx(i, j, k, grid, ρ, ρv, ρu) = ∂xᶠᵃᵃ(i, j, k, grid, V_over_ρ, ρ, ρv) + ∂yᵃᶠᵃ(i, j, k, grid, U_over_ρ, ρ, ρu)
@inline strain_rate_tensor_vz(i, j, k, grid, ρ, ρv, ρw) = ∂zᵃᵃᶠ(i, j, k, grid, V_over_ρ, ρ, ρv) + ∂yᵃᶠᵃ(i, j, k, grid, W_over_ρ, ρ, ρw)
@inline strain_rate_tensor_wx(i, j, k, grid, ρ, ρw, ρu) = ∂xᶠᵃᵃ(i, j, k, grid, W_over_ρ, ρ, ρw) + ∂zᵃᵃᶠ(i, j, k, grid, U_over_ρ, ρ, ρu)
@inline strain_rate_tensor_wy(i, j, k, grid, ρ, ρw, ρv) = ∂yᵃᶠᵃ(i, j, k, grid, W_over_ρ, ρ, ρw) + ∂zᵃᵃᶠ(i, j, k, grid, V_over_ρ, ρ, ρv)

@inline viscous_flux_ux(i, j, k, grid, νᶜᶜᶜ, ρ, ρu) = @inbounds ρ[i, j, k] * νᶜᶜᶜ * Axᵃᵃᶜ(i, j, k, grid) * strain_rate_tensor_ux(i, j, k, grid, ρ, ρu)
@inline viscous_flux_vy(i, j, k, grid, νᶜᶜᶜ, ρ, ρv) = @inbounds ρ[i, j, k] * νᶜᶜᶜ * Ayᵃᵃᶜ(i, j, k, grid) * strain_rate_tensor_vy(i, j, k, grid, ρ, ρv)
@inline viscous_flux_wz(i, j, k, grid, νᶜᶜᶜ, ρ, ρw) = @inbounds ρ[i, j, k] * νᶜᶜᶜ * Azᵃᵃᵃ(i, j, k, grid) * strain_rate_tensor_wz(i, j, k, grid, ρ, ρw)

@inline viscous_flux_uy(i, j, k, grid, νᶠᶠᶜ, ρ, ρu, ρv) = ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * νᶠᶠᶜ * Ayᵃᵃᶜ(i, j, k, grid) * strain_rate_tensor_uy(i, j, k, grid, ρ, ρu, ρv)
@inline viscous_flux_uz(i, j, k, grid, νᶠᶜᶠ, ρ, ρu, ρw) = ℑxzᶠᵃᶠ(i, j, k, grid, ρ) * νᶠᶜᶠ * Azᵃᵃᵃ(i, j, k, grid) * strain_rate_tensor_uz(i, j, k, grid, ρ, ρu, ρw)
@inline viscous_flux_vx(i, j, k, grid, νᶠᶠᶜ, ρ, ρv, ρu) = ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * νᶠᶠᶜ * Axᵃᵃᶜ(i, j, k, grid) * strain_rate_tensor_vx(i, j, k, grid, ρ, ρv, ρu)
@inline viscous_flux_vz(i, j, k, grid, νᶜᶠᶠ, ρ, ρv, ρw) = ℑyzᵃᶠᶠ(i, j, k, grid, ρ) * νᶜᶠᶠ * Azᵃᵃᵃ(i, j, k, grid) * strain_rate_tensor_vz(i, j, k, grid, ρ, ρv, ρw)
@inline viscous_flux_wx(i, j, k, grid, νᶠᶜᶠ, ρ, ρw, ρu) = ℑxzᶠᵃᶠ(i, j, k, grid, ρ) * νᶠᶜᶠ * Axᵃᵃᶠ(i, j, k, grid) * strain_rate_tensor_wx(i, j, k, grid, ρ, ρw, ρu)
@inline viscous_flux_wy(i, j, k, grid, νᶜᶠᶠ, ρ, ρw, ρv) = ℑyzᵃᶠᶠ(i, j, k, grid, ρ) * νᶜᶠᶠ * Ayᵃᵃᶠ(i, j, k, grid) * strain_rate_tensor_wy(i, j, k, grid, ρ, ρw, ρv)

@inline ∂ⱼτ₁ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, ρ, ρũ, args...) =
    (1/Vᵃᵃᶜ(i, j, k, grid)
        * (  δxᶠᵃᵃ(i, j, k, grid, viscous_flux_ux, closure.ν, ρ, ρũ.ρu)
           + δyᵃᶜᵃ(i, j, k, grid, viscous_flux_uy, closure.ν, ρ, ρũ.ρu, ρũ.ρv)
           + δzᵃᵃᶜ(i, j, k, grid, viscous_flux_uz, closure.ν, ρ, ρũ.ρu, ρũ.ρw)))

@inline ∂ⱼτ₂ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, ρ, ρũ, args...) =
    (1/Vᵃᵃᶜ(i, j, k, grid)
        * (  δxᶜᵃᵃ(i, j, k, grid, viscous_flux_vx, closure.ν, ρ, ρũ.ρv, ρũ.ρu)
           + δyᵃᶠᵃ(i, j, k, grid, viscous_flux_vy, closure.ν, ρ, ρũ.ρv)
           + δzᵃᵃᶜ(i, j, k, grid, viscous_flux_vz, closure.ν, ρ, ρũ.ρv, ρũ.ρw)))

@inline ∂ⱼτ₃ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, ρ, ρũ, args...) =
    (1/Vᵃᵃᶠ(i, j, k, grid)
        * (  δxᶜᵃᵃ(i, j, k, grid, viscous_flux_wx, closure.ν, ρ, ρũ.ρw, ρũ.ρu)
           + δyᵃᶜᵃ(i, j, k, grid, viscous_flux_wy, closure.ν, ρ, ρũ.ρw, ρũ.ρv)
           + δzᵃᵃᶠ(i, j, k, grid, viscous_flux_wz, closure.ν, ρ, ρũ.ρw)))

@inline u₁∂ⱼτ₁ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, ρ, ρũ, args...) =
    U_over_ρ(i, j, k, grid, ρ, ρũ.ρu) * ∂ⱼτ₁ⱼ(i, j, k, grid, closure, ρ, ρũ, args...)

@inline u₂∂ⱼτ₂ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, ρ, ρũ, args...) =
    V_over_ρ(i, j, k, grid, ρ, ρũ.ρv) * ∂ⱼτ₂ⱼ(i, j, k, grid, closure, ρ, ρũ, args...)

@inline u₃∂ⱼτ₃ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, ρ, ρũ, args...) =
    W_over_ρ(i, j, k, grid, ρ, ρũ.ρw) * ∂ⱼτ₃ⱼ(i, j, k, grid, closure, ρ, ρũ, args...)

@inline Q_dissipation(i, j, k, grid, closure::IsotropicDiffusivity, ρ, ρũ, args...) =
    (  ℑxᶜᵃᵃ(i, j, k, grid, u₁∂ⱼτ₁ⱼ, closure, ρ, ρũ, args...)
     + ℑyᵃᶜᵃ(i, j, k, grid, u₂∂ⱼτ₂ⱼ, closure, ρ, ρũ, args...)
     + ℑzᵃᵃᶜ(i, j, k, grid, u₃∂ⱼτ₃ⱼ, closure, ρ, ρũ, args...))
