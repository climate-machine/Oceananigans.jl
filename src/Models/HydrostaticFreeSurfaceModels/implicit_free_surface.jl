using Oceananigans.Grids: AbstractGrid
using Oceananigans.Architectures: device
using Oceananigans.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Solvers: solve!
using Oceananigans.Fields
using Oceananigans.Utils: prettytime

using Adapt
using KernelAbstractions: NoneEvent

struct ImplicitFreeSurface{E, G, B, I, M, S}
    η :: E
    gravitational_acceleration :: G
    barotropic_volume_flux :: B
    implicit_step_solver :: I
    solver_method :: M
    solver_settings :: S
end

"""
    ImplicitFreeSurface(; solver=:PreconditionedConjugateGradient, gravitational_acceleration=g_Earth, solver_settings...) =

The implicit free surface equation is

```math
(∇ʰ ⋅ H ∇ʰ - 1 / (g Δt²)) ηⁿ⁺¹ = ∇ʰ ⋅ Q★ / (g Δt) - ηⁿ / (g Δt²)
```

where ``H`` is depth, ``g`` is gravitational acceleration, ``Δt`` is time step, and
``Q★`` is the barotropic volume flux associated with the predictor velocity field.

This can be solved in general using the `PreconditionedConjugateGradientSolver`.

In the case that ``H`` is constant, we divide through to obtain

```math
(∇² - 1 / (g H Δt²)) ηⁿ⁺¹ = 1 / (g H Δt) * (∇ʰ ⋅ Q★ - ηⁿ / Δt)
```

The above can be solved with the `FastFourierTransformPoissonSolver` on grids with regular spacing in x and y.
"""
ImplicitFreeSurface(; solver_method=:Default, gravitational_acceleration=g_Earth, solver_settings...) =
    ImplicitFreeSurface(nothing, gravitational_acceleration, nothing, nothing, solver_method, solver_settings)

Adapt.adapt_structure(to, free_surface::ImplicitFreeSurface) =
    ImplicitFreeSurface(Adapt.adapt(to, free_surface.η), free_surface.gravitational_acceleration,
                        nothing, nothing, nothing, nothing)

# Internal function for HydrostaticFreeSurfaceModel
function FreeSurface(free_surface::ImplicitFreeSurface{Nothing}, velocities, arch, grid)
    η = FreeSurfaceDisplacementField(velocities, free_surface, arch, grid)
    g = convert(eltype(grid), free_surface.gravitational_acceleration)

    # Initialize barotropic volume fluxes
    barotropic_x_volume_flux = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    barotropic_y_volume_flux = ReducedField(Center, Face, Nothing, arch, grid; dims=3)
    barotropic_volume_flux = (u=barotropic_x_volume_flux, v=barotropic_y_volume_flux)

    solver_method = free_surface.solver_method
    solver = build_implicit_step_solver(Val(solver_method), arch, grid, free_surface.solver_settings)

    return ImplicitFreeSurface(η, g,
                               barotropic_volume_flux,
                               solver,
                               solver_method,
                               free_surface.solver_settings)
end

is_horizontally_regular(grid) = false
is_horizontally_regular(::RegularRectilinearGrid) = true
is_horizontally_regular(::VerticallyStretchedRectilinearGrid) = true

function build_implicit_step_solver(::Val{:Default}, arch, grid, settings)
    default_method = is_horizontally_regular(grid) ? :FastFourierTransform : :PreconditionedConjugateGradient
    return build_implicit_step_solver(Val(default_method), arch, grid, settings)
end

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0

"""
Implicitly step forward η.
"""
ab2_step_free_surface!(free_surface::ImplicitFreeSurface, model, Δt, χ, velocities_update) =
    implicit_free_surface_step!(free_surface::ImplicitFreeSurface, model, Δt, χ, velocities_update)

function implicit_free_surface_step!(free_surface::ImplicitFreeSurface, model, Δt, χ, velocities_update)
    η = free_surface.η
    g = free_surface.gravitational_acceleration
    rhs = free_surface.implicit_step_solver.right_hand_side
    ∫ᶻQ = free_surface.barotropic_volume_flux
    solver = free_surface.implicit_step_solver
    arch = model.architecture

    # Compute barotropic volume flux. Blocking.
    compute_vertically_integrated_volume_flux!(∫ᶻQ, model, velocities_update)

    # Compute right hand side of implicit free surface equation
    rhs_event = compute_implicit_free_surface_right_hand_side!(rhs, solver, g, Δt, ∫ᶻQ, η)
    wait(device(arch), rhs_event)

    # Solve for the free surface at tⁿ⁺¹
    start_time = time_ns()

    solve!(η, solver, rhs, g, Δt)

    @debug "Implict step solve took $(prettytime((time_ns() - start_time) * 1e-9))."

    fill_halo_regions!(η, arch)
    
    return NoneEvent()
end

