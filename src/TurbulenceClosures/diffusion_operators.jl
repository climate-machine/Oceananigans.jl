#####
##### Diffusive fluxes
#####

@inline diffusive_flux_x(i, j, k, grid, clock, κᶠᶜᶜ::Number, c) = κᶠᶜᶜ * Axᵃᵃᶠ(i, j, k, grid) * ∂xᶠᵃᵃ(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, clock, κᶜᶠᶜ::Number, c) = κᶜᶠᶜ * Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶠᵃ(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, clock, κᶜᶜᶠ::Number, c) = κᶜᶜᶠ * Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, c)

# Diffusivities-as-functions

@inline diffusive_flux_x(i, j, k, grid, clock, κ::Function, c) =
        diffusive_flux_x(i, j, k, grid, clock, κ(xnode(Face, i, grid), ynode(Cell, j, grid), znode(Cell, k, grid), clock.time), c)

@inline diffusive_flux_y(i, j, k, grid, clock, κ::Function, c) =
        diffusive_flux_y(i, j, k, grid, clock, κ(xnode(Cell, i, grid), ynode(Face, j, grid), znode(Cell, k, grid), clock.time), c)

@inline diffusive_flux_z(i, j, k, grid, clock, κ::Function, c) =
        diffusive_flux_z(i, j, k, grid, clock, κ(xnode(Cell, i, grid), ynode(Cell, j, grid), znode(Face, k, grid), clock.time), c)

@inline diffusive_flux_x(i, j, k, grid, clock, κ::AbstractField, c) = ℑxᶠᵃᵃ(i, j, k, grid, κ) * Axᵃᵃᶠ(i, j, k, grid) * ∂xᶠᵃᵃ(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, clock, κ::AbstractField, c) = ℑyᵃᶠᵃ(i, j, k, grid, κ) * Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶠᵃ(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, clock, κ::AbstractField, c) = ℑzᵃᵃᶠ(i, j, k, grid, κ) * Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, c)
                     
#####
##### Laplacian diffusion operator
#####

"""
    ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, κˣ, κʸ, κᶻ, c)

Calculates diffusion for a tracer c via

    1/V * [δxᶜᵃᵃ(κˣ * Ax * ∂xᶠᵃᵃ(c)) + δyᵃᶜᵃ(κʸ * Ay * ∂yᵃᶠᵃ(c)) + δzᵃᵃᶜ(κᶻ * Az * ∂zᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, clock, κˣ, κʸ, κᶻ, c)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, diffusive_flux_x, clock, κˣ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, diffusive_flux_y, clock, κʸ, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, diffusive_flux_z, clock, κᶻ, c))
end

@inline ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, clock, κ, c) = ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, clock, κ, κ, κ, c)
