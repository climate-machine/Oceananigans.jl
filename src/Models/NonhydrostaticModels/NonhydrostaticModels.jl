module NonhydrostaticModels

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!

import Oceananigans: fields, prognostic_fields

#####
##### NonhydrostaticModel definition
#####

include("nonhydrostatic_model.jl")
include("show_nonhydrostatic_model.jl")
include("set_nonhydrostatic_model.jl")

#####
##### Time-stepping NonhydrostaticModels
#####

"""
    fields(model::NonhydrostaticModel)

Returns a flattened `NamedTuple` of the fields in `model.velocities` and `model.tracers`.
"""
fields(model::NonhydrostaticModel) = merge(model.velocities, model.tracers)
prognostic_fields(model::NonhydrostaticModel) = fields(model)

include("update_hydrostatic_pressure.jl")
include("update_nonhydrostatic_model_state.jl")
include("pressure_correction.jl")
include("nonhydrostatic_tendency_kernel_functions.jl")
include("calculate_nonhydrostatic_tendencies.jl")

end # module
