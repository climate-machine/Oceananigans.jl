module NonhydrostaticModels

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!
using Oceananigans.Grids
using Oceananigans.Solvers
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

import Oceananigans: fields, prognostic_fields

PressureSolver(arch, grid::RegularRectilinearGrid) = FFTBasedPoissonSolver(arch, grid)
PressureSolver(arch, grid::VerticallyStretchedRectilinearGrid) = FourierTridiagonalPoissonSolver(arch, grid)

# *Evil grin*
PressureSolver(arch, ibg::ImmersedBoundaryGrid) = PressureSolver(arch, ibg.grid)

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

include("solve_for_pressure.jl")
include("update_hydrostatic_pressure.jl")
include("update_nonhydrostatic_model_state.jl")
include("pressure_correction.jl")
include("velocity_and_tracer_tendencies.jl")
include("calculate_tendencies.jl")
include("correct_nonhydrostatic_immersed_tendencies.jl")

end # module
