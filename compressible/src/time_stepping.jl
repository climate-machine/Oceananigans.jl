using JULES.Operators

using Oceananigans.BoundaryConditions

using Oceananigans.Fields: datatuple
using Oceananigans.TimeSteppers: tick!

import Oceananigans.TimeSteppers: time_step!
import Oceananigans.Simulations: ab2_or_rk3_time_step!

ab2_or_rk3_time_step!(model::CompressibleModel, Δt; euler) = time_step!(model, Δt)

function time_step!(model::CompressibleModel, Δt)
    arch = model.architecture
    total_density  = model.total_density
    momenta = model.momenta
    tracers = model.tracers
    diffusivities  = model.diffusivities

    slow_source_terms   = model.time_stepper.slow_source_terms
    fast_source_terms   = model.time_stepper.fast_source_terms
    intermediate_fields = model.time_stepper.intermediate_fields

    momenta_names = propertynames(momenta)
    tracers_names = propertynames(tracers)

    intermediate_momenta_fields = [getproperty(intermediate_fields, ρu)         for ρu in momenta_names]
    intermediate_tracers_fields = [getproperty(intermediate_fields.tracers, ρc) for ρc in tracers_names]

    intermediate_momenta = NamedTuple{momenta_names}(intermediate_momenta_fields)
    intermediate_tracers = NamedTuple{tracers_names}(intermediate_tracers_fields)

    first_stage_Δt  = Δt / 3
    second_stage_Δt = Δt / 2
    third_stage_Δt  = Δt

    #####
    ##### Compute slow source terms
    #####

    density_update_event =
        launch!(arch, model.grid, :xyz, update_total_density!,
                datatuple(total_density), model.grid, model.gases, datatuple(tracers),
                dependencies=Event(device(arch)))

    wait(device(arch), density_update_event)
    
    fill_halo_regions!(merge((Σρ=total_density,), momenta, tracers), model.architecture, model.clock, nothing)
    fill_halo_regions!(momenta.ρw, model.architecture, model.clock, nothing)
    fill_halo_regions!(intermediate_momenta.ρw, model.architecture, model.clock, nothing)

    compute_slow_source_terms!(
        slow_source_terms, arch, model.grid, model.thermodynamic_variable, model.gases, model.gravity,
        model.coriolis, model.closure, total_density, momenta, tracers, diffusivities, model.forcing, model.clock)

    fill_halo_regions!(slow_source_terms.ρw, model.architecture, model.clock, nothing)

    #####
    ##### Stage 1
    #####

    density_update_event =
        launch!(arch, model.grid, :xyz, update_total_density!,
                datatuple(total_density), model.grid, model.gases, datatuple(tracers),
                dependencies=Event(device(arch)))

    wait(device(arch), density_update_event)

    fill_halo_regions!(merge((Σρ=total_density,), momenta, tracers), model.architecture, model.clock, nothing)    
    fill_halo_regions!(momenta.ρw, model.architecture, model.clock, nothing)
    fill_halo_regions!(intermediate_momenta.ρw, model.architecture, model.clock, nothing)

    compute_fast_source_terms!(
        fast_source_terms, arch, model.grid, model.thermodynamic_variable, model.gases, model.gravity,
        model.advection, total_density, momenta, tracers, slow_source_terms)

    calculate_boundary_tendency_contributions!(fast_source_terms, arch, momenta, tracers, model.clock, nothing)

    advance_state_variables!(intermediate_fields, arch, model.grid, momenta, tracers, fast_source_terms, Δt=first_stage_Δt)

    tick!(model.clock, 0, stage=true)

    #####
    ##### Stage 2
    #####

    density_update_event =
        launch!(arch, model.grid, :xyz, update_total_density!,
                datatuple(total_density), model.grid, model.gases, datatuple(intermediate_tracers),
                dependencies=Event(device(arch)))

    wait(device(arch), density_update_event)

    fill_halo_regions!(merge((Σρ=total_density,), intermediate_momenta, intermediate_tracers), model.architecture, model.clock, nothing)
    fill_halo_regions!(momenta.ρw, model.architecture, model.clock, nothing)
    fill_halo_regions!(intermediate_momenta.ρw, model.architecture, model.clock, nothing)

    compute_fast_source_terms!(
        fast_source_terms, arch, model.grid, model.thermodynamic_variable, model.gases, model.gravity,
        model.advection, total_density, intermediate_momenta, intermediate_tracers, slow_source_terms)

    calculate_boundary_tendency_contributions!(fast_source_terms, arch, intermediate_momenta,
                                               intermediate_tracers, model.clock, nothing)

    advance_state_variables!(intermediate_fields, arch, model.grid, momenta, tracers, fast_source_terms, Δt=second_stage_Δt)

    tick!(model.clock, 0, stage=true)

    #####
    ##### Stage 3
    #####

    density_update_event =
        launch!(arch, model.grid, :xyz, update_total_density!,
                datatuple(total_density), model.grid, model.gases, datatuple(intermediate_tracers),
                dependencies=Event(device(arch)))

    wait(device(arch), density_update_event)

    fill_halo_regions!(merge((Σρ=total_density,), intermediate_momenta, intermediate_tracers), model.architecture, model.clock, nothing)
    fill_halo_regions!(momenta.ρw, model.architecture, model.clock, nothing)
    fill_halo_regions!(intermediate_momenta.ρw, model.architecture, model.clock, nothing)

    compute_fast_source_terms!(
        fast_source_terms, arch, model.grid, model.thermodynamic_variable, model.gases, model.gravity,
        model.advection, total_density, intermediate_momenta, intermediate_tracers, slow_source_terms)

    calculate_boundary_tendency_contributions!(fast_source_terms, arch, intermediate_momenta,
                                               intermediate_tracers, model.clock, nothing)

    state_variables = (momenta..., tracers = tracers)
    advance_state_variables!(state_variables, arch, model.grid, momenta, tracers, fast_source_terms, Δt=third_stage_Δt)

    tick!(model.clock, Δt)

    return nothing
end
