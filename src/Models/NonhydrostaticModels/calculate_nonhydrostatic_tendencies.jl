import Oceananigans.TimeSteppers: calculate_tendencies!

using Oceananigans: fields
using Oceananigans.Utils: work_layout

"""
    calculate_tendencies!(model::NonhydrostaticModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model::NonhydrostaticModel)

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    calculate_interior_tendency_contributions!(model)

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                               model.architecture,
                                               model.velocities,
                                               model.tracers,
                                               model.clock,
                                               prognostic_fields(model))

    return nothing
end

""" Store previous value of the source term and calculate current source term. """
function calculate_interior_tendency_contributions!(model)

    arch = model.architecture
    grid = model.grid
    velocities = model.velocities

    velocities_immersed_boundary_conditions = (u = velocities.u.boundary_conditions.immersed,
                                               v = velocities.v.boundary_conditions.immersed,
                                               w = velocities.w.boundary_conditions.immersed)


    momentum_kernel_args = (model.grid,
                            model.advection,
                            model.coriolis,
                            model.stokes_drift,
                            model.closure,
                            model.buoyancy,
                            model.background_fields,
                            model.velocities,
                            velocities_immersed_boundary_conditions,
                            model.tracers,
                            model.diffusivities,
                            model.forcing,
                            model.pressures.pHY′,
                            model.clock)

    workgroup, worksize = work_layout(grid, :xyz)

    calculate_Gu_kernel! = calculate_Gu!(device(arch), workgroup, worksize)
    calculate_Gv_kernel! = calculate_Gv!(device(arch), workgroup, worksize)
    calculate_Gw_kernel! = calculate_Gw!(device(arch), workgroup, worksize)
    calculate_Gc_kernel! = calculate_Gc!(device(arch), workgroup, worksize)

    barrier = Event(device(arch))

    Gu_event = calculate_Gu_kernel!(model.timestepper.Gⁿ.u, momentum_kernel_args..., dependencies=barrier)
    Gv_event = calculate_Gv_kernel!(model.timestepper.Gⁿ.v, momentum_kernel_args..., dependencies=barrier)
    Gw_event = calculate_Gw_kernel!(model.timestepper.Gⁿ.w, momentum_kernel_args..., dependencies=barrier)

    events = [Gu_event, Gv_event, Gw_event]

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))
        @inbounds tracer_tendency = model.timestepper.Gⁿ[tracer_name]
        @inbounds tracer_forcing = model.forcing[tracer_name]
        tracer_immersed_boundary_condition = model.tracers[tracer_name].boundary_conditions.immersed

        Gc_event = calculate_Gc_kernel!(tracer_tendency,
                                        grid,
                                        Val(tracer_index),
                                        model.advection,
                                        tracer_immersed_boundary_condition,
                                        model.closure,
                                        model.buoyancy,
                                        model.background_fields,
                                        model.velocities,
                                        model.tracers,
                                        model.diffusivities,
                                        tracer_forcing,
                                        model.clock,
                                        dependencies = barrier)

        push!(events, Gc_event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Tendency calculators for u, v, w-velocity
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function calculate_Gu!(Gu, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function calculate_Gv!(Gv, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, args...)
end

""" Calculate the right-hand-side of the w-velocity equation. """
@kernel function calculate_Gw!(Gw, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, args...)
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function calculate_Gc!(Gc, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, args...)
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_boundary_tendency_contributions!(Gⁿ, arch, velocities, tracers, clock, model_fields)

    barrier = Event(device(arch))

    events = []

    # Velocity fields
    for i in 1:3
        x_bcs_event = apply_x_bcs!(Gⁿ[i], velocities[i], arch, barrier, clock, model_fields)
        y_bcs_event = apply_y_bcs!(Gⁿ[i], velocities[i], arch, barrier, clock, model_fields)
        z_bcs_event = apply_z_bcs!(Gⁿ[i], velocities[i], arch, barrier, clock, model_fields)

        push!(events, x_bcs_event, y_bcs_event, z_bcs_event)
    end

    # Tracer fields
    for i in 4:length(Gⁿ)
        x_bcs_event = apply_x_bcs!(Gⁿ[i], tracers[i-3], arch, barrier, clock, model_fields)
        y_bcs_event = apply_y_bcs!(Gⁿ[i], tracers[i-3], arch, barrier, clock, model_fields)
        z_bcs_event = apply_z_bcs!(Gⁿ[i], tracers[i-3], arch, barrier, clock, model_fields)

        push!(events, x_bcs_event, y_bcs_event, z_bcs_event)
    end

    events = filter(e -> typeof(e) <: Event, events)

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end
