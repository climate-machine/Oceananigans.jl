using Oceananigans.BoundaryConditions: fill_halo_regions!

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model::ShallowWaterModel)

Fill halo regions for `model.solution` and `model.tracers`.
"""
function update_state!(model::ShallowWaterModel)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.solution, model.tracers),
                       model.clock,
                       fields(model))

    return nothing
end

