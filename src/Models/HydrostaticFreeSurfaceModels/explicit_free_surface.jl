using Oceananigans.Architectures: device_event
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: ∂xᶠᶜᵃ, ∂yᶜᶠᵃ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions

using Adapt

struct ExplicitFreeSurface{E, T}
    η :: E
    gravitational_acceleration :: T
end

ExplicitFreeSurface(; gravitational_acceleration=g_Earth) =
    ExplicitFreeSurface(nothing, gravitational_acceleration)

Adapt.adapt_structure(to, free_surface::ExplicitFreeSurface) =
    ExplicitFreeSurface(Adapt.adapt(to, free_surface.η), free_surface.gravitational_acceleration)

#####
##### Interface to HydrostaticFreeSurfaceModel
#####

function FreeSurface(free_surface::ExplicitFreeSurface{Nothing}, velocities, arch, grid)
    η = FreeSurfaceDisplacementField(velocities, free_surface, arch, grid)
    g = convert(eltype(grid), free_surface.gravitational_acceleration)
    return ExplicitFreeSurface(η, g)
end

#####
##### Kernel functions for HydrostaticFreeSurfaceModel
#####

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface::ExplicitFreeSurface) =
    free_surface.gravitational_acceleration * ∂xᶠᶜᵃ(i, j, k, grid, free_surface.η)

@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface::ExplicitFreeSurface) =
    free_surface.gravitational_acceleration * ∂yᶜᶠᵃ(i, j, k, grid, free_surface.η)

#####
##### Time stepping
#####

ab2_step_free_surface!(free_surface::ExplicitFreeSurface, model, Δt, χ, velocities_update) =
    explicit_ab2_step_free_surface!(free_surface, model, Δt, χ)

explicit_ab2_step_free_surface!(free_surface, model, Δt, χ) =
    launch!(model.architecture, model.grid, :xy,
            _explicit_ab2_step_free_surface!, free_surface.η, Δt, χ,
            model.timestepper.Gⁿ.η, model.timestepper.G⁻.η,
            dependencies = device_event(model.architecture))

#####
##### Kernel
#####

@kernel function _explicit_ab2_step_free_surface!(η, Δt, χ::FT, Gηⁿ, Gη⁻) where FT
    i, j = @index(Global, NTuple)

    @inbounds begin
        η[i, j, 1] += Δt * ((FT(1.5) + χ) * Gηⁿ[i, j, 1] - (FT(0.5) + χ) * Gη⁻[i, j, 1])
    end
end
