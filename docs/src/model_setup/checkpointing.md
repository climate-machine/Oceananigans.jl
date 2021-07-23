# Checkpointing

A checkpointer can be used to serialize the entire model state to a file from which the model can be restored at any
time. This is useful if you'd like to periodically checkpoint when running long simulations in case of crashes or
cluster time limits, but also if you'd like to restore from a checkpoint and try out multiple scenarios.

For example, to periodically checkpoint the model state to disk every 1,000,000 seconds of simulation time to files of
the form `model_checkpoint_iteration12500.jld2` where `12500` is the iteration number (automatically filled in)

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest
julia> using Oceananigans, Oceananigans.Units

julia> model = NonhydrostaticModel(grid=RegularRectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1)));

julia> simulation = Simulation(model, Δt=1, stop_iteration=1);

julia> simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(5years), prefix="model_checkpoint")
Checkpointer{TimeInterval, Vector{Symbol}}(TimeInterval(1.5768e8, 0.0), ".", "model_checkpoint", [:architecture, :grid, :clock, :coriolis, :buoyancy, :closure, :velocities, :tracers, :timestepper, :particles], false, false, false)
```

The default options should provide checkpoint files that are easy to restore from in most cases. For more advanced
options and features, see [`Checkpointer`](@ref).

## Picking up a simulation from a checkpoint file

Picking up a simulation from a checkpoint requires the original script that was
used to generate the checkpoint data. Change the first instance of `[run!](@ref)` in the script
to take `pickup=true`:

```julia
run!(simulation, pickup=true)
```

which finds the latest checkpoint file in the current working directory,
loads prognostic fields and their tendencies from file, resets the model clock and iteration, and
updates the model auxiliary state before starting the time-stepping loop.

Use `pickup=iteration` to pick up from a specific iteration, or `pickup=filepath` to pickup
from a specific file.
