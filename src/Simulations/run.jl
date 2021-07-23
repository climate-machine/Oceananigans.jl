using Glob

using Oceananigans.Utils: initialize_schedule!, align_time_step
using Oceananigans.Fields: set!
using Oceananigans.OutputWriters: WindowedTimeAverage, checkpoint_superprefix
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper, update_state!, next_time, unit_time

using Oceananigans: AbstractModel, run_diagnostic!, write_output!

import Oceananigans.OutputWriters: checkpoint_path, set!

# Simulations are for running

function stop(sim)
    time_before = time()

    for sc in sim.stop_criteria
        if sc(sim)
            time_after = time()
            sim.run_time += time_after - time_before
            return true
        end
    end

    time_after = time()
    sim.run_time += time_after - time_before

    return false
end

function iteration_limit_exceeded(sim)
    if sim.model.clock.iteration >= sim.stop_iteration
          @info "Simulation is stopping. Model iteration $(sim.model.clock.iteration) " *
                "has hit or exceeded simulation stop iteration $(sim.stop_iteration)."
          return true
    end
    return false
end

function stop_time_exceeded(sim)
    if sim.model.clock.time >= sim.stop_time
          @info "Simulation is stopping. Model time $(prettytime(sim.model.clock.time)) " *
                "has hit or exceeded simulation stop time $(prettytime(sim.stop_time))."
          return true
    end
    return false
end

function wall_time_limit_exceeded(sim)
    if sim.run_time >= sim.wall_time_limit
          @info "Simulation is stopping. Simulation run time $(prettytime(sim.run_time)) " *
                "has hit or exceeded simulation wall time limit $(prettytime(sim.wall_time_limit))."
          return true
    end
    return false
end

add_dependency!(diagnostics, output) = nothing # fallback
add_dependency!(diags, wta::WindowedTimeAverage) = wta ∈ values(diags) || push!(diags, wta)

add_dependencies!(diags, writer) = [add_dependency!(diags, out) for out in values(writer.outputs)]
add_dependencies!(sim, ::Checkpointer) = nothing # Checkpointer does not have "outputs"

get_Δt(Δt) = Δt
get_Δt(wizard::TimeStepWizard) = wizard.Δt
get_Δt(simulation::Simulation) = get_Δt(simulation.Δt)

ab2_or_rk3_time_step!(model::AbstractModel{<:QuasiAdamsBashforth2TimeStepper}, Δt; euler) = time_step!(model, Δt, euler=euler)
ab2_or_rk3_time_step!(model::AbstractModel{<:RungeKutta3TimeStepper}, Δt; euler) = time_step!(model, Δt)

we_want_to_pickup(pickup::Bool) = pickup
we_want_to_pickup(pickup) = true

"""
    aligned_time_step(sim)

Returns a time step Δt that is aligned with the output writer schedules and stop time of the simulation `sim`.
The purpose of aligning the time step is to ensure simulations do not time step beyond the `sim.stop_time` and
to ensure that output is written at the exact time specified by the output writer schedules.
"""
function aligned_time_step(sim)
    clock = sim.model.clock

    # Align time step with output writing
    Δt = get_Δt(sim)
    for writer in values(sim.output_writers)
        Δt = align_time_step(writer.schedule, clock, Δt)
    end

    # Align time step with simulation stop time
    if next_time(clock, Δt) > sim.stop_time
        Δt = unit_time(sim.stop_time - clock.time)
    end

    return Δt
end

"""
    run!(simulation; pickup=false)

Run a `simulation` until one of `simulation.stop_criteria` evaluates `true`.
The simulation will then stop.

# Picking simulations up from a checkpoint

Simulations will be "picked up" from a checkpoint if `pickup` is either `true`, a `String`,
or an `Integer` greater than 0.

Picking up a simulation sets field and tendency data to the specified checkpoint,
leaving all other model properties unchanged.

Possible values for `pickup` are:

    * `pickup=true` will pick a simulation up from the latest checkpoint associated with
      the `Checkpointer` in simulation.output_writers`.

    * `pickup=iteration::Int` will pick a simulation up from the checkpointed file associated
       with `iteration` and the `Checkpointer` in simulation.output_writers`.

    * `pickup=filepath::String` will pick a simulation up from checkpointer data in `filepath`.

Note that `pickup=true` and `pickup=iteration` will fail if `simulation.output_writers` contains
more than one checkpointer.
"""
function run!(sim; pickup=false, callbacks=[])

    model = sim.model
    clock = model.clock

    add_callbacks!(sim, callbacks)

    if we_want_to_pickup(pickup)
        checkpointers = filter(writer -> writer isa Checkpointer, collect(values(sim.output_writers)))
        checkpoint_filepath = checkpoint_path(pickup, checkpointers)

        # https://github.com/CliMA/Oceananigans.jl/issues/1159
        if pickup isa Bool && isnothing(checkpoint_filepath)
            @warn "pickup=true but no checkpoints were found. Simulation will run without picking up."
        else
            set!(model, checkpoint_filepath)
        end
    end

    # Conservatively initialize the model state
    update_state!(model)

    # Output and diagnostics initialization
    for writer in values(sim.output_writers)
        initialize_schedule!(writer.schedule)
        add_dependencies!(sim.diagnostics, writer)
    end

    [initialize_schedule!(diag.schedule) for diag in values(sim.diagnostics)]

    while !stop(sim)
        time_before = time()

        # Evaluate all diagnostics, and then write all output at first iteration
        if clock.iteration == 0
            [run_diagnostic!(diag, sim.model) for diag in values(sim.diagnostics)]
            [write_output!(writer, sim.model) for writer in values(sim.output_writers)]
            [callback(sim) for callback in values(sim.callbacks)]
        end

        # Ensure that the simulation doesn't iterate past `stop_iteration`.
        iterations = min(sim.iteration_interval, sim.stop_iteration - clock.iteration)

        for n in 1:iterations
            clock.time >= sim.stop_time && break

            # Temporary fix for https://github.com/CliMA/Oceananigans.jl/issues/1280
            aligned_Δt = aligned_time_step(sim)
            if aligned_Δt <= 0
                Δt = get_Δt(sim)
            else
                Δt = min(get_Δt(sim), aligned_Δt)
            end

            euler = clock.iteration == 0 || (sim.Δt isa TimeStepWizard && n == 1)
            ab2_or_rk3_time_step!(model, Δt, euler=euler)

            # Run diagnostics, then write output
            [  diag.schedule(model)   && run_diagnostic!(diag, sim.model) for diag in values(sim.diagnostics)]
            [writer.schedule(model)   && write_output!(writer, sim.model) for writer in values(sim.output_writers)]
            [callback.schedule(model) && callback(sim)                    for callback in values(sim.callbacks)]
        end

        sim.progress(sim)

        sim.Δt isa TimeStepWizard && update_Δt!(sim.Δt, model)

        time_after = time()
        sim.run_time += time_after - time_before
    end

    return nothing
end
