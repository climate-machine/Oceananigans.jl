using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Fields

using Adapt

struct ImplicitFreeSurface{E, G, B, VF, I}
    η :: E
    gravitational_acceleration :: G
    barotropic_volume_flux :: B
    vertically_integrated_lateral_face_areas :: VF
    implicit_step_solver :: I
end

# User interface to ImplicitFreeSurface
ImplicitFreeSurface(; gravitational_acceleration=g_Earth) =
    ImplicitFreeSurface(nothing, gravitational_acceleration, nothing, nothing, nothing)

### ExplicitFreeSurface(; gravitational_acceleration=g_Earth) =
###     ExplicitFreeSurface(nothing, gravitational_acceleration)

### Adapt.adapt_structure(to, free_surface::ExplicitFreeSurface) =
###     ExplicitFreeSurface(Adapt.adapt(to, free_surface.η), free_surface.gravitational_acceleration)

# Internal function for HydrostaticFreeSurfaceModel
function FreeSurface(free_surface::ImplicitFreeSurface{Nothing}, velocities, arch, grid)
    η = FreeSurfaceDisplacementField(velocities, arch, grid)
    g = convert(eltype(grid), free_surface.gravitational_acceleration)

    barotropic_x_volume_flux = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    barotropic_y_volume_flux = ReducedField(Center, Face, Nothing, arch, grid; dims=3)
    barotropic_volume_flux = (u=barotropic_x_volume_flux, v=barotropic_y_volume_flux)

    Ax_zintegral = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    Ay_zintegral = ReducedField(Center, Face, Nothing, arch, grid; dims=3)
    vertically_integrated_lateral_face_areas = (Ax = Ax_zintegral, Ay=Ay_zintegral)
    compute_vertically_integrated_lateral_face_areas!(vertically_integrated_lateral_face_areas, grid, arch)

    implicit_step_solver = ImplicitFreeSurfaceSolver(arch, η, vertically_integrated_lateral_face_areas; maxit=100)
    ## implicit_step_solver = ImplicitFreeSurfaceSolver(arch, η, vertically_integrated_lateral_face_areas)

    return ImplicitFreeSurface(η, g, 
                               barotropic_volume_flux, 
                               vertically_integrated_lateral_face_areas,
                               implicit_step_solver)
end

### In final form the two functions below will return 0 (these are invoked in hydrostatic_free_surface_tendency_kernel_functions.jl )
explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface::ImplicitFreeSurface) = 0

explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface::ImplicitFreeSurface) = 0

@kernel function _compute_vertically_integrated_lateral_face_areas!(grid, A )
    i, j, k = @index(Global, NTuple)
    # U.w[i, j, 1] = 0 is enforced via halo regions.
    A.Ax[i, j, 1] = 0
    A.Ay[i, j, 1] = 0
    @unroll for k in 1:grid.Nz
        @inbounds A.Ax[i, j, 1] += Δyᶠᶜᵃ(i, j, k, grid)*Δzᵃᵃᶜ(i, j, k, grid)
        @inbounds A.Ay[i, j, 1] += Δxᶜᶠᵃ(i, j, k, grid)*Δzᵃᵃᶜ(i, j, k, grid)
    end
end

function compute_vertically_integrated_lateral_face_areas!(vertically_integrated_lateral_face_areas, grid, arch)

    event = launch!(arch,
                    grid,
                    :xyz,
                    _compute_vertically_integrated_lateral_face_areas!,
                    grid,
                    vertically_integrated_lateral_face_areas,
                    dependencies=Event(device(arch)))

    wait(device(arch), event)

    return
end
