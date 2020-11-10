#####
##### Viscous fluxes
#####

@inline viscous_flux_ux(i, j, k, grid, clock, νᶜᶜᶜ::Number, u) = νᶜᶜᶜ * ℑxᶜᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * ∂xᶜᵃᵃ(i, j, k, grid, u)
@inline viscous_flux_uy(i, j, k, grid, clock, νᶠᶠᶜ::Number, u) = νᶠᶠᶜ * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * ∂yᵃᶠᵃ(i, j, k, grid, u)
@inline viscous_flux_uz(i, j, k, grid, clock, νᶠᶜᶠ::Number, u) = νᶠᶜᶠ * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ) * ∂zᵃᵃᶠ(i, j, k, grid, u)

@inline viscous_flux_vx(i, j, k, grid, clock, νᶠᶠᶜ::Number, v) = νᶠᶠᶜ * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * ∂xᶠᵃᵃ(i, j, k, grid, v)
@inline viscous_flux_vy(i, j, k, grid, clock, νᶜᶜᶜ::Number, v) = νᶜᶜᶜ * ℑyᵃᶜᵃ(i, j, k, grid, Ayᵃᵃᶜ) * ∂yᵃᶜᵃ(i, j, k, grid, v)
@inline viscous_flux_vz(i, j, k, grid, clock, νᶜᶠᶠ::Number, v) = νᶜᶠᶠ * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ) * ∂zᵃᵃᶠ(i, j, k, grid, v)

@inline viscous_flux_wx(i, j, k, grid, clock, νᶠᶜᶠ::Number, w) = νᶠᶜᶠ * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * ∂xᶠᵃᵃ(i, j, k, grid, w)
@inline viscous_flux_wy(i, j, k, grid, clock, νᶜᶠᶠ::Number, w) = νᶜᶠᶠ * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * ∂yᵃᶠᵃ(i, j, k, grid, w)
@inline viscous_flux_wz(i, j, k, grid, clock, νᶜᶜᶜ::Number, w) = νᶜᶜᶜ * ℑzᵃᵃᶜ(i, j, k, grid, Azᵃᵃᵃ) * ∂zᵃᵃᶜ(i, j, k, grid, w)

# Viscosities-as-functions

@inline viscous_flux_ux(i, j, k, grid, clock, ν::Function, u) =
        viscous_flux_ux(i, j, k, grid, clock, ν(xnode(Cell, i, grid), ynode(Cell, j, grid), znode(Cell, k, grid), clock.time), u)

@inline viscous_flux_uy(i, j, k, grid, clock, ν::Function, u) =
        viscous_flux_uy(i, j, k, grid, clock, ν(xnode(Cell, i, grid), ynode(Face, j, grid), znode(Face, k, grid), clock.time), u)

@inline viscous_flux_uz(i, j, k, grid, clock, ν::Function, u) =
        viscous_flux_uz(i, j, k, grid, clock, ν(xnode(Face, i, grid), ynode(Cell, j, grid), znode(Face, k, grid), clock.time), u)

@inline viscous_flux_vx(i, j, k, grid, clock, ν::Function, v) =
        viscous_flux_vx(i, j, k, grid, clock, ν(xnode(Face, i, grid), ynode(Face, j, grid), znode(Cell, k, grid), clock.time), v)

@inline viscous_flux_vy(i, j, k, grid, clock, ν::Function, v) =
        viscous_flux_vy(i, j, k, grid, clock, ν(xnode(Cell, i, grid), ynode(Cell, j, grid), znode(Cell, k, grid), clock.time), v)

@inline viscous_flux_vz(i, j, k, grid, clock, ν::Function, v) =
        viscous_flux_vz(i, j, k, grid, clock, ν(xnode(Cell, i, grid), ynode(Face, j, grid), znode(Face, k, grid), clock.time), v)
                        

@inline viscous_flux_wx(i, j, k, grid, clock, ν::Function, w) =
        viscous_flux_wx(i, j, k, grid, clock, ν(xnode(Face, i, grid), ynode(Cell, j, grid), znode(Face, k, grid), clock.time), w)
                        
@inline viscous_flux_wy(i, j, k, grid, clock, ν::Function, w) =
        viscous_flux_wy(i, j, k, grid, clock, ν(xnode(Cell, i, grid), ynode(Face, j, grid), znode(Face, k, grid), clock.time), w)

@inline viscous_flux_wz(i, j, k, grid, clock, ν::Function, w) =
        viscous_flux_wz(i, j, k, grid, clock, ν(xnode(Cell, i, grid), ynode(Cell, j, grid), znode(Cell, k, grid), clock.time), w)
                        
# Viscosities-as-fields

@inline viscous_flux_ux(i, j, k, grid, clock, ν::AbstractField, u) = @inbounds ν[i, j, k]     * ℑxᶜᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * ∂xᶜᵃᵃ(i, j, k, grid, u)
@inline viscous_flux_uy(i, j, k, grid, clock, ν::AbstractField, u) = ℑxyᶠᶠᵃ(i, j, k, grid, ν) * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * ∂yᵃᶠᵃ(i, j, k, grid, u)
@inline viscous_flux_uz(i, j, k, grid, clock, ν::AbstractField, u) = ℑxzᶠᵃᶠ(i, j, k, grid, ν) * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ) * ∂zᵃᵃᶠ(i, j, k, grid, u)

@inline viscous_flux_vx(i, j, k, grid, clock, ν::AbstractField, v) = ℑxyᶠᶠᵃ(i, j, k, grid, ν) * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * ∂xᶠᵃᵃ(i, j, k, grid, v)
@inline viscous_flux_vy(i, j, k, grid, clock, ν::AbstractField, v) = @inbounds ν[i, j, k]     * ℑyᵃᶜᵃ(i, j, k, grid, Ayᵃᵃᶜ) * ∂yᵃᶜᵃ(i, j, k, grid, v)
@inline viscous_flux_vz(i, j, k, grid, clock, ν::AbstractField, v) = ℑyzᵃᶠᶠ(i, j, k, grid, ν) * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ) * ∂zᵃᵃᶠ(i, j, k, grid, v)

@inline viscous_flux_wx(i, j, k, grid, clock, ν::AbstractField, w) = ℑxzᶠᵃᶠ(i, j, k, grid, ν) * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * ∂xᶠᵃᵃ(i, j, k, grid, w)
@inline viscous_flux_wy(i, j, k, grid, clock, ν::AbstractField, w) = ℑyzᵃᶠᶠ(i, j, k, grid, ν) * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * ∂yᵃᶠᵃ(i, j, k, grid, w)
@inline viscous_flux_wz(i, j, k, grid, clock, ν::AbstractField, w) = @inbounds ν[i, j, k]     * ℑzᵃᵃᶜ(i, j, k, grid, Azᵃᵃᵃ) * ∂zᵃᵃᶜ(i, j, k, grid, w)

#####
##### Viscous dissipation operators
#####

"""
    ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, νˣ, νʸ, νᶻ, u)

Calculates viscous dissipation for the u-velocity via

    1/V * [δxᶠᵃᵃ(νˣ * ℑxᶜᵃᵃ(Ax) * δxᶜᵃᵃ(u)) + δyᵃᶜᵃ(νʸ * ℑyᵃᶠᵃ(Ay) * δyᵃᶠᵃ(u)) + δzᵃᵃᶜ(νᶻ * ℑzᵃᵃᶠ(Az) * δzᵃᵃᶠ(u))]

which will end up at the location `fcc`.
"""
@inline function ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, clock, νˣ, νʸ, νᶻ, u)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, viscous_flux_ux, clock, νˣ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_uy, clock, νʸ, u) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_uz, clock, νᶻ, u))
end

"""
    ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, νˣ, νʸ, νᶻ, v)

Calculates viscous dissipation for the v-velocity via

    1/Vᵛ * [δxᶜᵃᵃ(νˣ * ℑxᶠᵃᵃ(Ax) * δxᶠᵃᵃ(v)) + δyᵃᶠᵃ(νʸ * ℑyᵃᶜᵃ(Ay) * δyᵃᶜᵃ(v)) + δzᵃᵃᶜ(νᶻ * ℑzᵃᵃᶠ(Az) * δzᵃᵃᶠ(v))]

which will end up at the location `cfc`.
"""
@inline function ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, clock, νˣ, νʸ, νᶻ, v)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_vx, clock, νˣ, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, viscous_flux_vy, clock, νʸ, v) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_vz, clock, νᶻ, v))
end

"""
    ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid::Grid, νˣ, νʸ, νᶻ, w)

Calculates viscous dissipation for the w-velocity via

    1/Vʷ * [δxᶜᵃᵃ(νˣ * ℑxᶠᵃᵃ(Ax) * δxᶠᵃᵃ(w)) + δyᵃᶜᵃ(νʸ * ℑyᵃᶠᵃ(Ay) * δyᵃᶠᵃ(w)) + δzᵃᵃᶠ(νᶻ * ℑzᵃᵃᶜ(Az) * δzᵃᵃᶜ(w))]

which will end up at the location `ccf`.
"""
@inline function ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, clock, νˣ, νʸ, νᶻ, w)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_wx, clock, νˣ, w) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_wy, clock, νʸ, w) +
                                    δzᵃᵃᶠ(i, j, k, grid, viscous_flux_wz, clock, νᶻ, w))
end

#####
##### Viscous dissipation for isotropic viscosity
#####

@inline ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, clock, ν, u) = ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, clock, ν, ν, ν, u)
@inline ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, clock, ν, v) = ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, clock, ν, ν, ν, v)
@inline ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, clock, ν, w) = ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, clock, ν, ν, ν, w)
