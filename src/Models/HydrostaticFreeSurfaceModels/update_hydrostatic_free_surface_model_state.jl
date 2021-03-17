using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.TurbulenceClosures: calculate_diffusivities!
using Oceananigans.Models.IncompressibleModels: update_hydrostatic_pressure!

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model::HydrostaticFreeSurfaceModel)

Update peripheral aspects of the model (halo regions, diffusivities, hydrostatic pressure) to the current model state.
"""
function update_state!(model::HydrostaticFreeSurfaceModel)

    # Fill halos for velocities and tracers
    fill_halo_regions!(fields(model), model.clock, fields(model))

    compute_w_from_continuity!(model)

    fill_halo_regions!(model.velocities.w, model.clock, fields(model))

    # Calculate diffusivities
    calculate_diffusivities!(model.diffusivities, model.architecture, model.grid, model.closure,
                             model.buoyancy, model.velocities, model.tracers)

    fill_halo_regions!(model.diffusivities, model.clock, fields(model))

    # Calculate hydrostatic pressure
    pressure_calculation = launch!(model.architecture, model.grid, :xy, update_hydrostatic_pressure!,
                                   model.pressure.pHY′, model.grid, model.buoyancy, model.tracers,
                                   dependencies=Event(device(model.architecture)))

    # Fill halo regions for pressure
    wait(device(model.architecture), pressure_calculation)

    fill_halo_regions!(model.pressure.pHY′)

    return nothing
end
