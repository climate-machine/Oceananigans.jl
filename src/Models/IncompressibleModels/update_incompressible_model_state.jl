using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.TurbulenceClosures: calculate_diffusivities!

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model::IncompressibleModel)

Update peripheral aspects of the model (halo regions, diffusivities, hydrostatic pressure) to the current model state.
"""
function update_state!(model::IncompressibleModel)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.velocities, model.tracers), model.clock, fields(model))

    # Calculate diffusivities
    calculate_diffusivities!(model.diffusivities, model.architecture, model.grid, model.closure,
                             model.buoyancy, model.velocities, model.tracers)

    fill_halo_regions!(model.diffusivities, model.clock, fields(model))

    # Calculate hydrostatic pressure
    pressure_calculation = launch!(model.architecture, model.grid, :xy, update_hydrostatic_pressure!,
                                   model.pressures.pHY′, model.grid, model.buoyancy, model.tracers,
                                   dependencies=Event(device(model.architecture)))

    # Fill halo regions for pressure
    wait(device(model.architecture), pressure_calculation)

    fill_halo_regions!(model.pressures.pHY′)

    return nothing
end

