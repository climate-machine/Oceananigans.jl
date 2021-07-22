module TimeSteppers

export
    QuasiAdamsBashforth2TimeStepper,
    RungeKutta3TimeStepper,
    time_step!,
    Clock,
    tendencies

using CUDA
using KernelAbstractions
using Oceananigans: AbstractModel, prognostic_fields
using Oceananigans.Architectures: device
using Oceananigans.Fields: TendencyFields
using Oceananigans.LagrangianParticleTracking: update_particle_properties!
using Oceananigans.Utils: work_layout

"""
    AbstractTimeStepper

Abstract supertype for time steppers.
"""
abstract type AbstractTimeStepper end

"""
    TimeStepper(name, args...)

Returns a timestepper with name `name`, instantiated with `args...`.

Example
=======

julia> stepper = TimeStepper(:QuasiAdamsBashforth2, CPU(), grid, tracernames)
"""
function TimeStepper(name::Symbol, args...; kwargs...)
    fullname = Symbol(name, :TimeStepper)
    return @eval $fullname($args...; $kwargs...)
end

# Fallback
TimeStepper(stepper::AbstractTimeStepper, args...; kwargs...) = stepper

function update_state! end
function calculate_tendencies! end

calculate_pressure_correction!(model, Δt) = nothing
pressure_correct_velocities!(model, Δt) = nothing

include("clock.jl")
include("store_tendencies.jl")
include("quasi_adams_bashforth_2.jl")
include("runge_kutta_3.jl")

end # module
