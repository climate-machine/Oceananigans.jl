using KernelAbstractions
using Oceananigans.Utils
using Oceananigans.Architectures: device, @hascuda, CPU, GPU, array_type
using Oceananigans.Fields: datatuple
using Oceananigans.BoundaryConditions: apply_x_bcs!, apply_y_bcs!, apply_z_bcs!

@kernel function update_total_density!(total_density, grid, gases, tracers)
    i, j, k = @index(Global, NTuple)

    @inbounds total_density[i, j, k] = diagnose_density(i, j, k, grid, gases, tracers)
end

# This is for users. Do not use in time_stepping.jl as it doesn't make use of intermediate fields.
function update_total_density!(model)

    density_update_event =
        launch!(model.architecture, model.grid, :xyz, update_total_density!,
                datatuple(model.total_density), model.grid, model.gases, datatuple(model.tracers),
                dependencies=Event(device(model.architecture)))

    wait(device(model.architecture), density_update_event)

    return nothing
end

#####
##### Computing slow source terms (viscous dissipation, diffusion, and Coriolis terms).
#####

function compute_slow_source_terms!(slow_source_terms, arch, grid, thermodynamic_variable, gases, gravity, coriolis, closure, total_density, momenta, tracers, diffusivities, forcing, clock)

    slow_source_terms, total_density, momenta, tracers, diffusivities =
        datatuples(slow_source_terms, total_density, momenta, tracers, diffusivities)

    workgroup, worksize = work_layout(grid, :xyz)
    barrier = Event(device(arch))

    momentum_kernel! = compute_slow_momentum_source_terms!(device(arch), workgroup, worksize)
    tracer_kernel! = compute_slow_tracer_source_terms!(device(arch), workgroup, worksize)
    thermodynamic_variable_kernel! = compute_slow_thermodynamic_variable_source_terms!(device(arch), workgroup, worksize)

    momentum_event = momentum_kernel!(slow_source_terms, grid, coriolis, closure, total_density, momenta, tracers, diffusivities, forcing, clock, dependencies=barrier)

    events = [momentum_event]

    for (tracer_index, ρc_name) in enumerate(propertynames(tracers))
        ρc   = getproperty(tracers, ρc_name)
        S_ρc = getproperty(slow_source_terms.tracers, ρc_name)
        forcing_ρc = getproperty(forcing, ρc_name)

        tracer_event = tracer_kernel!(S_ρc, grid, closure, tracer_index, total_density, ρc, momenta, tracers, diffusivities, forcing_ρc, clock, dependencies=barrier)
        push!(events, tracer_event)
    end

    thermodynamic_variable_event = thermodynamic_variable_kernel!(slow_source_terms.tracers[1], grid, thermodynamic_variable, gases, gravity, closure, total_density, momenta, tracers, diffusivities, dependencies=barrier)
    push!(events, thermodynamic_variable_event)

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

@kernel function compute_slow_momentum_source_terms!(slow_source_terms, grid, coriolis, closure, total_density, momenta, tracers, diffusivities, forcing, clock)
    i, j, k = @index(Global, NTuple)

    @inbounds slow_source_terms.ρu[i, j, k] = ρu_slow_source_term(i, j, k, grid, coriolis, closure, total_density, momenta, diffusivities) + forcing.ρu(i, j, k, grid, clock, merge(momenta, tracers))
    @inbounds slow_source_terms.ρv[i, j, k] = ρv_slow_source_term(i, j, k, grid, coriolis, closure, total_density, momenta, diffusivities) + forcing.ρv(i, j, k, grid, clock, merge(momenta, tracers))
    @inbounds slow_source_terms.ρw[i, j, k] = ρw_slow_source_term(i, j, k, grid, coriolis, closure, total_density, momenta, diffusivities) + forcing.ρw(i, j, k, grid, clock, merge(momenta, tracers))
end

@kernel function compute_slow_tracer_source_terms!(S_ρc, grid, closure, tracer_index, total_density, ρc, momenta, tracers, diffusivities, forcing, clock)
    i, j, k = @index(Global, NTuple)

    @inbounds S_ρc[i, j, k] = ρc_slow_source_term(i, j, k, grid, closure, tracer_index, total_density, ρc, diffusivities) + forcing(i, j, k, grid, clock, merge(momenta, tracers))
end

@kernel function compute_slow_thermodynamic_variable_source_terms!(S_ρt, grid, thermodynamic_variable, gases, gravity, closure, total_density, momenta, tracers, diffusivities)
    i, j, k = @index(Global, NTuple)

    @inbounds S_ρt[i, j, k] += ρt_slow_source_term(i, j, k, grid, closure, thermodynamic_variable, gases, gravity, total_density, momenta, tracers, diffusivities)
end

#####
##### Computing fast source terms (advection, pressure gradient, and buoyancy terms).
#####

function compute_fast_source_terms!(fast_source_terms, arch, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms)

    fast_source_terms, total_density, momenta, tracers, slow_source_terms =
        datatuples(fast_source_terms, total_density, momenta, tracers, slow_source_terms)

    workgroup, worksize = work_layout(grid, :xyz)
    barrier = Event(device(arch))

    momentum_kernel! = compute_fast_momentum_source_terms!(device(arch), workgroup, worksize)
    tracer_kernel! = compute_fast_tracer_source_terms!(device(arch), workgroup, worksize)
    thermodynamic_variable_kernel! = compute_fast_thermodynamic_variable_source_terms!(device(arch), workgroup, worksize)

    momentum_event = momentum_kernel!(fast_source_terms, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms, dependencies=barrier)

    events = [momentum_event]

    for ρc_name in propertynames(tracers)
        ρc   = getproperty(tracers, ρc_name)
        F_ρc = getproperty(fast_source_terms.tracers, ρc_name)
        S_ρc = getproperty(slow_source_terms.tracers, ρc_name)

        tracer_event = tracer_kernel!(F_ρc, grid, advection_scheme, total_density, momenta, ρc, S_ρc, dependencies=barrier)
        push!(events, tracer_event)
    end
    
    thermodynamic_variable_event = thermodynamic_variable_kernel!(fast_source_terms.tracers[1], grid, thermodynamic_variable, gases, gravity, total_density, momenta, tracers, dependencies=barrier)
    push!(events, thermodynamic_variable_event)

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

@kernel function compute_fast_momentum_source_terms!(fast_source_terms, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms)
    i, j, k = @index(Global, NTuple)

    @inbounds fast_source_terms.ρu[i, j, k] = ρu_fast_source_term(i, j, k, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms.ρu)
    @inbounds fast_source_terms.ρv[i, j, k] = ρv_fast_source_term(i, j, k, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms.ρv)
    @inbounds fast_source_terms.ρw[i, j, k] = ρw_fast_source_term(i, j, k, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms.ρw)
end

@kernel function compute_fast_tracer_source_terms!(F_ρc, grid, advection_scheme, total_density, momenta, ρc, S_ρc)
    i, j, k = @index(Global, NTuple)

    @inbounds F_ρc[i, j, k] = ρc_fast_source_term(i, j, k, grid, advection_scheme, total_density, momenta, ρc, S_ρc)
end

@kernel function compute_fast_thermodynamic_variable_source_terms!(F_ρt, grid, thermodynamic_variable, gases, gravity, total_density, momenta, tracers)
    i, j, k = @index(Global, NTuple)

    @inbounds F_ρt[i, j, k] += ρt_fast_source_term(i, j, k, grid, thermodynamic_variable, gases, gravity, total_density, momenta, tracers)
end

#####
##### Calculating boundary tendency contributions
#####

function calculate_boundary_tendency_contributions!(source_terms, arch, momenta, tracers, clock, model_fields)

    barrier = Event(device(arch))

    events = []

    # Momentum fields
    momentum_source_terms = (source_terms.ρu, source_terms.ρv, source_terms.ρw)

    for (ρϕ_source_term, ρϕ) in zip(momentum_source_terms, momenta)
        x_bcs_event = apply_x_bcs!(ρϕ_source_term, ρϕ, arch, barrier, clock, model_fields)
        y_bcs_event = apply_y_bcs!(ρϕ_source_term, ρϕ, arch, barrier, clock, model_fields)
        z_bcs_event = apply_z_bcs!(ρϕ_source_term, ρϕ, arch, barrier, clock, model_fields)

        push!(events, x_bcs_event, y_bcs_event, z_bcs_event)
    end

    # Tracer fields
    for (ρϕ_source_term, ρϕ) in zip(source_terms.tracers, tracers)
        x_bcs_event = apply_x_bcs!(ρϕ_source_term, ρϕ, arch, barrier, clock, model_fields)
        y_bcs_event = apply_y_bcs!(ρϕ_source_term, ρϕ, arch, barrier, clock, model_fields)
        z_bcs_event = apply_z_bcs!(ρϕ_source_term, ρϕ, arch, barrier, clock, model_fields)

        push!(events, x_bcs_event, y_bcs_event, z_bcs_event)
    end

    events = filter(e -> typeof(e) <: Event, events)

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Advancing state variables
#####

function advance_state_variables!(state_variables, arch, grid, momenta, tracers, fast_source_terms; Δt)
    
    state_variables, momenta, tracers, fast_source_terms =
        datatuples(state_variables, momenta, tracers, fast_source_terms )

    workgroup, worksize = work_layout(grid, :xyz)
    barrier = Event(device(arch))

    momentum_kernel! = advance_momentum!(device(arch), workgroup, worksize)
    tracer_kernel! = advance_tracer!(device(arch), workgroup, worksize)

    momentum_event = momentum_kernel!(state_variables, grid, momenta, fast_source_terms, Δt, dependencies=barrier)

    events = [momentum_event]

    for ρc_name in propertynames(tracers)
        ρc   = getproperty(tracers, ρc_name)
        ρc⁺  = getproperty(state_variables.tracers, ρc_name)
        F_ρc = getproperty(fast_source_terms.tracers, ρc_name)
        
        tracer_event = tracer_kernel!(ρc⁺, grid, ρc, F_ρc, Δt, dependencies=barrier)
        push!(events, tracer_event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

@kernel function advance_momentum!(momenta⁺, grid, momenta, fast_source_terms, Δt)
    i, j, k = @index(Global, NTuple)

    @inbounds momenta⁺.ρu[i, j, k] = momenta.ρu[i, j, k] + Δt * fast_source_terms.ρu[i, j, k]
    @inbounds momenta⁺.ρv[i, j, k] = momenta.ρv[i, j, k] + Δt * fast_source_terms.ρv[i, j, k]
    @inbounds momenta⁺.ρw[i, j, k] = momenta.ρw[i, j, k] + Δt * fast_source_terms.ρw[i, j, k]
end

@kernel function advance_tracer!(ρc⁺, grid, ρc, F_ρc, Δt)
    i, j, k = @index(Global, NTuple)

    @inbounds ρc⁺[i, j, k] = ρc[i, j, k] + Δt * F_ρc[i, j, k]
end
