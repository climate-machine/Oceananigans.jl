using Oceananigans.Operators
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!
using Oceananigans.Architectures: device_event

#####
##### Calculate the right-hand-side of the non-hydrostatic pressure Poisson equation.
#####

@kernel function calculate_pressure_source_term_fft_based_solver!(rhs, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@kernel function calculate_pressure_source_term_fourier_tridiagonal_solver!(rhs, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = Δzᵃᵃᶜ(i, j, k, grid) * divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

#####
##### Solve for pressure
#####

function solve_for_pressure!(pressure, solver::FFTBasedPoissonSolver, Δt, U★)

    # Calculate right hand side:
    rhs = solver.storage
    arch = solver.architecture
    grid = solver.grid

    rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term_fft_based_solver!,
                        rhs, grid, Δt, U★, dependencies = device_event(arch))

    wait(device(arch), rhs_event)

    # Solve pressure Poisson given for pressure, given rhs
    solve!(pressure, solver, rhs)

    return nothing
end

function solve_for_pressure!(pressure, solver::FourierTridiagonalPoissonSolver, Δt, U★)

    # Calculate right hand side:
    rhs = solver.source_term
    arch = solver.architecture
    grid = solver.grid

    rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term_fourier_tridiagonal_solver!,
                        rhs, grid, Δt, U★, dependencies = device_event(arch))

    wait(device(arch), rhs_event)

    # Pressure Poisson rhs, scaled by Δzᵃᵃᶜ, is stored in solver.source_term:
    solve!(pressure, solver)

    return nothing
end

