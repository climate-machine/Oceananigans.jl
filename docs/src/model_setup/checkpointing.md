# Checkpointing

A checkpointer can be used to save the model state to a file from which a model or simulation can be picked up from
at any time. This is useful if you'd like to periodically checkpoint when running long simulations in case of crashes or
cluster time limits, but also if you'd like to restore from a checkpoint and try out multiple scenarios.

For example, to periodically checkpoint the model state to disk every 5 years of simulation time to files with names
like `model_checkpoint_iteration12500.jld2` where `12500` is the iteration number (automatically filled in), use

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest
julia> using Oceananigans, Oceananigans.Units

julia> model = IncompressibleModel(grid=RegularRectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1)));

julia> checkpointer = Checkpointer(model, schedule=TimeInterval(5years), prefix="model_checkpoint")
Checkpointer{TimeInterval, Vector{Symbol}}(TimeInterval(1.5768e8, 0.0), ".", "model_checkpoint", [:architecture, :grid, :clock, :coriolis, :buoyancy, :closure, :velocities, :tracers, :timestepper, :particles], false, false, false)
```

The default options should provide checkpoint files that are easy to restore from in most cases. For more advanced
options and features, see [`Checkpointer`](@ref).

## Picking up a simulation from a checkpoint

To restore the model from a specific checkpoint file, for example `model_checkpoint_12345.jld2`, simply call

```julia
set!(model, "model_checkpoint_12345.jld2")
```

that sets all the checkpointed data in `model_checkpoint_12345.jld2` to `model`, including tendencies, and synchronizes the
model clock and iteration with the checkpointed clock and iteration.

To automatically pickup from a checkpoint, you can use the `pickup` keyword argument for `run!`. To pickup from the latest
checkpoint, use

```julia
run!(simulation, pickup=true)
```

or if you want to pickup from a specific iteration number, use

```julia
run!(simulation, pickup=n)
```

where `n` is an integer that refers to an iteration number of a checkpointer file to be "picked up" and run from,
or if you want to pickup from a specific checkpoint file use

```julia
run!(simulation, pickup=filepath)
```

where `filepath` is a string that indicates the path to checkpoint data.

Note that `simulation.output_writers` must include a `Checkpointer` for the `pickup` keyword argument to work.
