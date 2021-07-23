using Oceananigans.Models.NonhydrostaticModels: calculate_pressure_source_term_fft_based_solver!
using Oceananigans.Architectures: device_event
import Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!

function solve_for_pressure!(pressure, solver::DistributedFFTBasedPoissonSolver, Δt, U★)
    rhs = first(solver.storage)
    arch = solver.architecture
    grid = solver.my_grid

    rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term_fft_based_solver!,
                        rhs, grid, Δt, U★, dependencies = device_event(arch))

    wait(device(arch), rhs_event)

    # Solve pressure Poisson equation for pressure, given rhs
    solve!(pressure, solver)

    return pressure
end
