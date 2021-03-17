using Oceananigans.Solvers

import Oceananigans.TimeSteppers: calculate_pressure_correction!, pressure_correct_velocities!

"""
    calculate_pressure_correction!(model::IncompressibleModel, Δt)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function calculate_pressure_correction!(model::IncompressibleModel, Δt)

    fill_halo_regions!(model.velocities, model.clock, fields(model))

    solve_for_pressure!(model.pressures.pNHS, model.pressure_solver, model.architecture, model.grid, Δt, model.velocities)

    fill_halo_regions!(model.pressures.pNHS)

    return nothing
end

#####
##### Fractional and time stepping
#####

"""
Update the predictor velocities u, v, and w with the non-hydrostatic pressure via

    `u^{n+1} = u^n - δₓp_{NH} / Δx * Δt`
"""
@kernel function _pressure_correct_velocities!(U, grid, Δt, pNHS)
    i, j, k = @index(Global, NTuple)

    @inbounds U.u[i, j, k] -= ∂xᶠᵃᵃ(i, j, k, grid, pNHS) * Δt
    @inbounds U.v[i, j, k] -= ∂yᵃᶠᵃ(i, j, k, grid, pNHS) * Δt
    @inbounds U.w[i, j, k] -= ∂zᵃᵃᶠ(i, j, k, grid, pNHS) * Δt
end

"Update the solution variables (velocities and tracers)."
function pressure_correct_velocities!(model::IncompressibleModel, Δt)

    event = launch!(model.architecture, model.grid, :xyz,
                    _pressure_correct_velocities!,
                    model.velocities,
                    model.grid,
                    Δt,
                    model.pressures.pNHS,
                    dependencies=Event(device(model.architecture))) 

    wait(device(model.architecture), event)

    return nothing
end
