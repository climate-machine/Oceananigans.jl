import Oceananigans.TimeSteppers: calculate_tendencies!
import Oceananigans: tracer_tendency_kernel_function

using Oceananigans: fields, prognostic_fields
using Oceananigans.Utils: work_layout
using Oceananigans.TurbulenceClosures: TKEBasedVerticalDiffusivity

"""
    calculate_tendencies!(model::NonhydrostaticModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model::HydrostaticFreeSurfaceModel)

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    calculate_hydrostatic_free_surface_interior_tendency_contributions!(model)

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_hydrostatic_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                                           model.architecture,
                                                           model.velocities,
                                                           model.free_surface,
                                                           model.tracers,
                                                           model.clock,
                                                           fields(model))

    return nothing
end

""" Calculate momentum tendencies if momentum is not prescribed. `velocities` argument eases dispatch on `PrescribedVelocityFields`."""
function calculate_hydrostatic_momentum_tendencies!(model, velocities; dependencies = device_event(model))

    arch = model.architecture
    grid = model.grid
    velocities_immersed_boundary_conditions = (u = velocities.u.boundary_conditions.immersed,
                                               v = velocities.v.boundary_conditions.immersed,
                                               w = velocities.w.boundary_conditions.immersed)

    momentum_kernel_args = (grid,
                            model.advection.momentum,
                            model.coriolis,
                            model.closure,
                            velocities,
                            velocities_immersed_boundary_conditions, 
                            model.free_surface,
                            model.tracers,
                            model.buoyancy,
                            model.diffusivities,
                            model.pressure.pHY′,
                            model.auxiliary_fields,
                            model.forcing,
                            model.clock)

    Gu_event = launch!(arch, grid, :xyz,
                       calculate_hydrostatic_free_surface_Gu!, model.timestepper.Gⁿ.u, momentum_kernel_args...;
                       dependencies = dependencies)

    Gv_event = launch!(arch, grid, :xyz,
                       calculate_hydrostatic_free_surface_Gv!, model.timestepper.Gⁿ.v, momentum_kernel_args...;
                       dependencies = dependencies)

    Gη_event = launch!(arch, grid, Val(:xy),
                       calculate_hydrostatic_free_surface_Gη!, model.timestepper.Gⁿ.η,
                       grid,
                       model.velocities,
                       model.free_surface,
                       model.tracers,
                       model.auxiliary_fields,
                       model.forcing,
                       model.clock;
                       dependencies = dependencies)

    events = [Gu_event, Gv_event, Gη_event]

    return events
end

# Fallback
@inline tracer_tendency_kernel_function(model::HydrostaticFreeSurfaceModel, closure, tracer_name) = hydrostatic_free_surface_tracer_tendency
@inline tracer_tendency_kernel_function(model::HydrostaticFreeSurfaceModel, closure::TKEBasedVerticalDiffusivity, ::Val{:e}) = hydrostatic_turbulent_kinetic_energy_tendency

# Hack for closure tuples
tracer_tendency_kernel_function(model::HydrostaticFreeSurfaceModel, closures::Tuple, ::Val{:e}) = tracer_tendency_kernel_function(model, closures[1], Val(:e))

""" Store previous value of the source term and calculate current source term. """
function calculate_hydrostatic_free_surface_interior_tendency_contributions!(model)

    arch = model.architecture
    grid = model.grid

    barrier = device_event(model)

    events = calculate_hydrostatic_momentum_tendencies!(model, model.velocities; dependencies = barrier)

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))
        @inbounds tracer_tendency = model.timestepper.Gⁿ[tracer_name]
        @inbounds tracer_advection = model.advection[tracer_name]
        @inbounds tracer_forcing = model.forcing[tracer_name]
        tracer_kernel_function = tracer_tendency_kernel_function(model, model.closure, Val(tracer_name))
        tracer_immersed_boundary_condition = model.tracers[tracer_name].boundary_conditions.immersed

        Gc_event = launch!(arch, grid, :xyz,
                           calculate_hydrostatic_free_surface_Gc!,
                           tracer_tendency,
                           tracer_kernel_function,
                           grid,
                           Val(tracer_index),
                           tracer_advection,
                           tracer_immersed_boundary_condition,
                           model.closure,
                           model.buoyancy,
                           model.velocities,
                           model.free_surface,
                           model.tracers,
                           model.diffusivities,
                           model.auxiliary_fields,
                           tracer_forcing,
                           model.clock;
                           dependencies = barrier)

        push!(events, Gc_event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Tendency calculators for u, v
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function calculate_hydrostatic_free_surface_Gu!(Gu, grid, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function calculate_hydrostatic_free_surface_Gv!(Gv, grid, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

#####
##### Tendency calculators for tracers
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function calculate_hydrostatic_free_surface_Gc!(Gc, tendency_kernel_function, grid, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = tendency_kernel_function(i, j, k, grid, args...)
end

#####
##### Tendency calculators for an explicit free surface
#####

""" Calculate the right-hand-side of the free surface displacement (η) equation. """
@kernel function calculate_hydrostatic_free_surface_Gη!(Gη, grid, args...)
    i, j = @index(Global, NTuple)
    @inbounds Gη[i, j, 1] = free_surface_tendency(i, j, grid, args...)
end

#####
##### Boundary condributions to hydrostatic free surface model
#####

function apply_flux_bcs!(Gcⁿ, events, c, arch, barrier, clock, model_fields)
    x_bcs_event = apply_x_bcs!(Gcⁿ, c, arch, barrier, clock, model_fields)
    y_bcs_event = apply_y_bcs!(Gcⁿ, c, arch, barrier, clock, model_fields)
    z_bcs_event = apply_z_bcs!(Gcⁿ, c, arch, barrier, clock, model_fields)

    push!(events, x_bcs_event, y_bcs_event, z_bcs_event)

    return nothing
end

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_hydrostatic_boundary_tendency_contributions!(Gⁿ, arch, velocities, free_surface, tracers, clock, model_fields)

    barrier = Event(device(arch))

    events = []

    # Velocity fields
    for i in (:u, :v)
        apply_flux_bcs!(Gⁿ[i], events, velocities[i], arch, barrier, clock, model_fields)
    end

    # Free surface
    apply_flux_bcs!(Gⁿ.η, events, displacement(free_surface), arch, barrier, clock, model_fields)

    # Tracer fields
    for i in propertynames(tracers)
        apply_flux_bcs!(Gⁿ[i], events, tracers[i], arch, barrier, clock, model_fields)
    end

    events = filter(e -> typeof(e) <: Event, events)

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end
