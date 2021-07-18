using Oceananigans.Operators: Δy_cᶜᶜᵃ, Δx_cᶜᶜᵃ

#####
##### Diffusivities
#####

@inline κᶠᶜᶜ(i, j, k, grid, clock, κ::Number) = κ
@inline κᶜᶠᶜ(i, j, k, grid, clock, κ::Number) = κ
@inline κᶜᶜᶠ(i, j, k, grid, clock, κ::Number) = κ

@inline κᶠᶜᶜ(i, j, k, grid, clock, κ::F) where F<:Function = κ(xnode(Face(),   i, grid), ynode(Center(), j, grid), znode(Center(), k, grid), clock.time)
@inline κᶜᶠᶜ(i, j, k, grid, clock, κ::F) where F<:Function = κ(xnode(Center(), i, grid), ynode(Face(),   j, grid), znode(Center(), k, grid), clock.time)
@inline κᶜᶜᶠ(i, j, k, grid, clock, κ::F) where F<:Function = κ(xnode(Center(), i, grid), ynode(Center(), j, grid), znode(Face(),   k, grid), clock.time)

# Assumes that `κ` is located at cell centers
@inline κᶠᶜᶜ(i, j, k, grid, clock, κ::AbstractArray) = ℑxᶠᵃᵃ(i, j, k, grid, κ)
@inline κᶜᶠᶜ(i, j, k, grid, clock, κ::AbstractArray) = ℑyᵃᶠᵃ(i, j, k, grid, κ)
@inline κᶜᶜᶠ(i, j, k, grid, clock, κ::AbstractArray) = ℑzᵃᵃᶠ(i, j, k, grid, κ)

#####
##### Convenience diffusive flux function
#####

@inline diffusive_flux_x(i, j, k, grid, clock, κ, c) = - κᶠᶜᶜ(i, j, k, grid, clock, κ) * ∂xᶠᶜᵃ(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, clock, κ, c) = - κᶜᶠᶜ(i, j, k, grid, clock, κ) * ∂yᶜᶠᵃ(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, clock, κ, c) = - κᶜᶜᶠ(i, j, k, grid, clock, κ) * ∂zᵃᵃᶠ(i, j, k, grid, c)

#####
##### Diffusive flux divergence
#####

@inline _diffusive_flux_x(args...) = diffusive_flux_x(args...)
@inline _diffusive_flux_y(args...) = diffusive_flux_y(args...)
@inline _diffusive_flux_z(args...) = diffusive_flux_z(args...)

"""
    ∇_dot_qᶜ(i, j, k, grid, clock, closure::AbstractTurbulenceClosure, c, ::Val{tracer_index}, args...)

Calculates the divergence of the diffusive flux `qᶜ` for a tracer `c` via

    1/V * [δxᶜᵃᵃ(Ax * diffusive_flux_x) + δyᵃᶜᵃ(Ay * diffusive_flux_y) + δzᵃᵃᶜ(Az * diffusive_flux_z)]

which will end up at the location `ccc`.
"""
@inline function ∇_dot_qᶜ(i, j, k, grid, closure::AbstractTurbulenceClosure, c, tracer_index, args...)

    disc = time_discretization(closure)

    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_uᶠᶜᶜ, diffusive_flux_x, disc, closure, c, tracer_index, args...) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_vᶜᶠᶜ, diffusive_flux_y, disc, closure, c, tracer_index, args...) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_wᶜᶜᵃ, diffusive_flux_z, disc, closure, c, tracer_index, args...))
end

#####
##### Gradients of Laplacians
#####

@inline ∂x_∇²h_cᶠᶜᶜ(i, j, k, grid, c) = 1 / Azᶠᶜᵃ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, Δy_cᶜᶜᵃ, ∇²hᶜᶜᶜ, c)
@inline ∂y_∇²h_cᶜᶠᶜ(i, j, k, grid, c) = 1 / Azᶜᶠᵃ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, Δx_cᶜᶜᵃ, ∇²hᶜᶜᶜ, c)

