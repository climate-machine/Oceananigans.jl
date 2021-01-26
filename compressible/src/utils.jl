#####
##### Grabbing properties for thermodynamic variables
#####

function thermodynamic_field(model)
    model.thermodynamic_variable isa Energy  && return model.tracers.ρe
    model.thermodynamic_variable isa Entropy && return model.tracers.ρs
end

function intermediate_thermodynamic_field(model)
    model.thermodynamic_variable isa Energy  && return model.time_stepper.intermediate_fields.tracers.ρe
    model.thermodynamic_variable isa Entropy && return model.time_stepper.intermediate_fields.tracers.ρs
end

#####
##### Kernels for diagnosing thermodynamic variables
#####

@kernel function compute_velocities!(velocities, grid, momenta, total_density)
    i, j, k = @index(Global, NTuple)

    @inbounds velocities.u[i, j, k] = momenta.ρu[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, total_density)
    @inbounds velocities.v[i, j, k] = momenta.ρv[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, total_density)
    @inbounds velocities.w[i, j, k] = momenta.ρw[i, j, k] / ℑzᵃᵃᶠ(i, j, k, grid, total_density)
end

@kernel function compute_p_over_ρ!(p_over_ρ, grid, thermodynamic_variable, gases, gravity, total_density, momenta, tracers)
    i, j, k = @index(Global, NTuple)

    @inbounds p_over_ρ[i, j, k] = diagnose_p_over_ρ(i, j, k, grid, thermodynamic_variable, gases, gravity, total_density, momenta, tracers)
end

@kernel function compute_temperature!(temperature, grid, thermodynamic_variable, gases, gravity, total_density, momenta, tracers)
    i, j, k = @index(Global, NTuple)

    @inbounds temperature[i, j, k] = diagnose_temperature(i, j, k, grid, thermodynamic_variable, gases, gravity, total_density, momenta, tracers)
end

function compute_temperature!(model)
    temperature = intermediate_thermodynamic_field(model)

    temperature, total_density, momenta, tracers =
        datatuples(temperature, model.total_density, model.momenta, model.tracers)

    compute_temperature_event =
        launch!(model.architecture, model.grid, :xyz, compute_temperature!,
                temperature, model.grid, model.thermodynamic_variable, model.gases,
                model.gravity, total_density, momenta, tracers,
                dependencies=Event(device(model.architecture)))

    wait(device(model.architecture), compute_temperature_event)

    return temperature
end

#####
##### CFL
#####

function cfl(model, Δt)
    # We will store the velocities in the time stepper's intermediate fields.
    velocities = (
        u = model.time_stepper.intermediate_fields.ρu,
        v = model.time_stepper.intermediate_fields.ρv,
        w = model.time_stepper.intermediate_fields.ρw
    )

    velocities, momenta, total_density =
        datatuples(velocities, model.momenta, model.total_density)

    velocity_event =
        launch!(model.architecture, model.grid, :xyz, compute_velocities!,
                velocities, model.grid, momenta, total_density,
                dependencies=Event(device(model.architecture)))
    
    wait(device(model.architecture), velocity_event)

    u_max = maximum(velocities.u.parent)
    v_max = maximum(velocities.v.parent)
    w_max = maximum(velocities.w.parent)

    Δx, Δy, Δz = model.grid.Δx, model.grid.Δy, model.grid.Δz
    CFL = Δt / min(Δx/u_max, Δy/v_max, Δz/w_max)

    return CFL
end

#####
##### Acoustic CFL
#####

function acoustic_cfl(model, Δt)
    # We will store p/ρ in the time stepper's intermediate field for the thermodynamic variable.
    p_over_ρ = intermediate_thermodynamic_field(model)

    p_over_ρ, total_density, momenta, tracers =
        datatuples(p_over_ρ, model.total_density, model.momenta, model.tracers)

    compute_p_over_ρ_event =
        launch!(model.architecture, model.grid, :xyz, compute_p_over_ρ!,
                p_over_ρ, model.grid, model.thermodynamic_variable, model.gases,
                model.gravity, total_density, momenta, tracers,
                dependencies=Event(device(model.architecture)))

    wait(device(model.architecture), compute_p_over_ρ_event)

    p_over_ρ_max = maximum(p_over_ρ.parent)

    γ_max = maximum(gas.cₚ / gas.cᵥ for gas in model.gases)
    c_max = √(γ_max * p_over_ρ_max)

    Δx, Δy, Δz = model.grid.Δx, model.grid.Δy, model.grid.Δz
    acoustic_CFL = Δt / min(Δx/c_max, Δy/c_max, Δz/c_max)

    return acoustic_CFL
end
