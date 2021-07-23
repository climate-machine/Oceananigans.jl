# It's not a model --- its a simulation!

default_progress(simulation) = nothing

mutable struct Simulation{M, Δ, C, I, T, W, R, P, F, Π}
                 model :: M
                    Δt :: Δ
         stop_criteria :: C
        stop_iteration :: I
             stop_time :: T
       wall_time_limit :: W
              run_time :: R
           diagnostics :: OrderedDict{Symbol, AbstractDiagnostic}
        output_writers :: OrderedDict{Symbol, AbstractOutputWriter}
             callbacks :: OrderedDict{Symbol, Callback}
              progress :: P
    iteration_interval :: F
            parameters :: Π
end

"""
    Simulation(model; Δt,
               stop_criteria = Function[iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded],
               stop_iteration = Inf,
               stop_time = Inf,
               wall_time_limit = Inf,
               diagnostics = OrderedDict{Symbol, AbstractDiagnostic}(),
               output_writers = OrderedDict{Symbol, AbstractOutputWriter}(),
               callback = OrderedDict{Symbol, Callback}(),
               progress = nothing,
               iteration_interval = 1,
               parameters = nothing)

Construct an Oceananigans.jl `Simulation` for a `model` with time step `Δt`.

Keyword arguments
=================
    - `Δt`: Required keyword argument specifying the simulation time step. Can be a `Number`
      for constant time steps or a `TimeStepWizard` for adaptive time-stepping.

    - `stop_criteria`: A list of functions or callable objects (each taking a single argument,
      the `simulation`). If any of the functions return `true` when the stop criteria is
      evaluated the simulation will stop.

    - `stop_iteration`: Stop the simulation after this many iterations.

    - `stop_time`: Stop the simulation once this much model clock time has passed.

    - `wall_time_limit`: Stop the simulation if it's been running for longer than this many
       seconds of wall clock time.

    - `progress`: A function with a single argument, the `simulation`. Will be called every
      `iteration_interval` iterations. Useful for logging simulation health.

    - `iteration_interval`: How often to update the time step, check stop criteria, and call
      `progress` function (in number of iterations).

    - `parameters`: Parameters that can be accessed in the `progress` function.
"""
function Simulation(model; Δt,
                    stop_criteria = Any[iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded],
                    stop_iteration = Inf,
                    stop_time = Inf,
                    wall_time_limit = Inf,
                    diagnostics = OrderedDict{Symbol, AbstractDiagnostic}(),
                    output_writers = OrderedDict{Symbol, AbstractOutputWriter}(),
                    callbacks = OrderedDict{Symbol, Callback}(),
                    progress = default_progress,
                    iteration_interval = 1,
                    parameters = nothing)

   if stop_iteration == Inf && stop_time == Inf && wall_time_limit == Inf
       @warn "This simulation will run forever as stop iteration = stop time " *
             "= wall time limit = Inf."
   end

   if Δt isa TimeStepWizard && iteration_interval == 1
       @warn "You have used the default iteration_interval=1. This simulation will " *
             "recalculate the time step every iteration which can be slow."
   end

   # Check for NaNs in the model's first field.
   model_fields = fields(model)
   field_to_check_nans = NamedTuple{keys(model_fields) |> first |> tuple}(first(model_fields) |> tuple)
   diagnostics[:nan_checker] = NaNChecker(fields=field_to_check_nans,
                                          schedule=IterationInterval(iteration_interval))

   run_time = 0.0

   return Simulation(model, Δt, stop_criteria, stop_iteration, stop_time, wall_time_limit, run_time,
                     diagnostics, output_writers, callbacks, progress, iteration_interval, parameters)
end

Base.show(io::IO, s::Simulation) =
    print(io, "Simulation{$(typeof(s.model).name){$(Base.typename(typeof(s.model.architecture))), $(eltype(s.model.grid))}}\n",
            "├── Model clock: time = $(prettytime(s.model.clock.time)), iteration = $(s.model.clock.iteration) \n",
            "├── Next time step ($(typeof(s.Δt))): $(prettytime(get_Δt(s.Δt))) \n",
            "├── Iteration interval: $(s.iteration_interval)\n",
            "├── Stop criteria: $(s.stop_criteria)\n",
            "├── Run time: $(prettytime(s.run_time)), wall time limit: $(s.wall_time_limit)\n",
            "├── Stop time: $(prettytime(s.stop_time)), stop iteration: $(s.stop_iteration)\n",
            "├── Diagnostics: $(ordered_dict_show(s.diagnostics, "│"))\n",
            "└── Output writers: $(ordered_dict_show(s.output_writers, "│"))")

#####
##### Utilities
#####

Base.time(sim::Simulation) = sim.model.clock.time
iteration(sim::Simulation) = sim.model.clock.iteration
