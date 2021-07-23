using Oceananigans.Grids
using Oceananigans.Grids: x_domain, y_domain
using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.Architectures
using Oceananigans.Fields: ReducedField
using Statistics

import Oceananigans.Solvers: solve!

struct FFTImplicitFreeSurfaceSolver{S, G3, G2, R}
    fft_based_poisson_solver :: S
    three_dimensional_grid :: G3
    horizontal_grid :: G2
    right_hand_side :: R
end

"""
    function FFTImplicitFreeSurfaceSolver(arch, grid, settings)

Return a solver based on the fast Fourier transform for the elliptic equation
    
```math
[∇² - Az / (g H Δt²)] ηⁿ⁺¹ = (∇ʰ ⋅ Q★ - Az ηⁿ / Δt) / (g H Δt)
```

representing an implicit time discretization of the linear free surface evolution equation
for a fluid with constant depth `H`, horizontal areas `Az`, barotropic volume flux `Q★`, time
step `Δt`, gravitational acceleration `g`, and free surface at time-step `n` `ηⁿ`.
"""
function FFTImplicitFreeSurfaceSolver(arch, grid, settings)

    grid isa RegularRectilinearGrid || grid isa VerticallyStretchedRectilinearGrid ||
        throw(ArgumentError("FFTImplicitFreeSurfaceSolver requires horizontally-regular rectilinear grids."))

    # Construct a "horizontal grid". We support either x or y being Flat, but not both.
    TX, TY, TZ = topology(grid)
    sz = Nx, Ny = (grid.Nx, grid.Ny)
    halo = (grid.Hx, grid.Hy)

    domain = (x = x_domain(grid),
              y = y_domain(grid))

    # Reduce kwargs.
    # Either [1, 2], [1], or [2]
    nonflat_dims = findall(T -> !(T() isa Flat), (TX, TY))

    sz = Tuple(sz[i] for i in nonflat_dims)
    halo = Tuple(halo[i] for i in nonflat_dims)
    domain = NamedTuple((:x, :y)[i] => domain[i] for i in nonflat_dims)

    # Build a "horizontal grid" with a Flat vertical direction.
    # Even if the three dimensional grid is vertically stretched, we can only use
    # FFTImplicitFreeSurfaceSolver with grids that are regularly spaced in the
    # horizontal direction.
    #
    horizontal_grid = RegularRectilinearGrid(; topology = (TX, TY, Flat),
                                               size = sz,
                                               halo = halo,
                                               domain...)

    solver = FFTBasedPoissonSolver(arch, horizontal_grid)
    right_hand_side = solver.storage

    return FFTImplicitFreeSurfaceSolver(solver, grid, horizontal_grid, right_hand_side)
end

build_implicit_step_solver(::Val{:FastFourierTransform}, arch, grid, settings) =
    FFTImplicitFreeSurfaceSolver(arch, grid, settings)

#####
##### Solve...
#####

function solve!(η, implicit_free_surface_solver::FFTImplicitFreeSurfaceSolver, rhs, g, Δt)
    solver = implicit_free_surface_solver.fft_based_poisson_solver
    grid = implicit_free_surface_solver.three_dimensional_grid
    H = grid.Lz

    m = - 1 / (g * H * Δt^2) # units L⁻²

    # solve! is blocking:
    solve!(η, solver, rhs, m)
    
    return nothing
end

function compute_implicit_free_surface_right_hand_side!(rhs,
                                                        implicit_solver::FFTImplicitFreeSurfaceSolver,
                                                        g, Δt, ∫ᶻQ, η)

    solver = implicit_solver.fft_based_poisson_solver
    arch = solver.architecture
    grid = implicit_solver.three_dimensional_grid
    H = grid.Lz

    event = launch!(arch, grid, :xy,
                    fft_based_implicit_free_surface_right_hand_side!,
                    rhs, grid, g, H, Δt, ∫ᶻQ, η)

    return event
end

@kernel function fft_based_implicit_free_surface_right_hand_side!(rhs, grid, g, H, Δt, ∫ᶻQ, η)
    i, j = @index(Global, NTuple)
    Az = Azᶜᶜᵃ(i, j, 1, grid)
    δ_Q = flux_div_xyᶜᶜᵃ(i, j, 1, grid, ∫ᶻQ.u, ∫ᶻQ.v)
    @inbounds rhs[i, j, 1] = (δ_Q - Az * η[i, j, 1] / Δt) / (g * H * Δt * Az)
end
