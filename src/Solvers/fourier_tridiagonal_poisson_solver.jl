using Oceananigans.Operators: Δzᵃᵃᶜ, Δzᵃᵃᶠ
using Oceananigans.Architectures: device_event

struct FourierTridiagonalPoissonSolver{A, G, B, R, S, β, T}
                  architecture :: A
                          grid :: G
    batched_tridiagonal_solver :: B
                   source_term :: R
                       storage :: S
                        buffer :: β
                    transforms :: T
end

@kernel function compute_main_diagonals!(D, grid, λx, λy)
    i, j = @index(Global, NTuple)
    Nz = grid.Nz

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    D[i, j, 1] = -1 / Δzᵃᵃᶠ(i, j, 2, grid) - Δzᵃᵃᶜ(i, j, 1, grid) * (λx[i] + λy[j])

    @unroll for k in 2:Nz-1
        D[i, j, k] = - (1 / Δzᵃᵃᶠ(i, j, k+1, grid) + 1 / Δzᵃᵃᶠ(i, j, k, grid)) - Δzᵃᵃᶜ(i, j, k, grid) * (λx[i] + λy[j])
    end

    D[i, j, Nz] = -1 / Δzᵃᵃᶠ(i, j, Nz, grid) - Δzᵃᵃᶜ(i, j, Nz, grid) * (λx[i] + λy[j])
end

function FourierTridiagonalPoissonSolver(arch, grid, planner_flag=FFTW.PATIENT)
    TX, TY, TZ = topology(grid)
    TZ != Bounded && error("FourierTridiagonalPoissonSolver can only be used with a Bounded z topology.")

    Nx, Ny, Nz = size(grid)

    # Compute discrete Poisson eigenvalues
    λx = poisson_eigenvalues(grid.Nx, grid.Lx, 1, TX())
    λy = poisson_eigenvalues(grid.Ny, grid.Ly, 2, TY())

    λx = arch_array(arch, λx)
    λy = arch_array(arch, λy)

    # Plan required transforms for x and y
    sol_storage = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))
    transforms = plan_transforms(arch, grid, sol_storage, planner_flag)

    # Lower and upper diagonals are the same
    lower_diagonal = CUDA.@allowscalar [1 / Δzᵃᵃᶠ(1, 1, k, grid) for k in 2:Nz]
    lower_diagonal = arch_array(arch, lower_diagonal)
    upper_diagonal = lower_diagonal

    # Compute diagonal coefficients for each grid point
    diagonal = arch_array(arch, zeros(Nx, Ny, Nz))
    event = launch!(arch, grid, :xy, compute_main_diagonals!, diagonal, grid, λx, λy, dependencies=device_event(arch))
    wait(device(arch), event)

    # Set up batched tridiagonal solver
    btsolver = BatchedTridiagonalSolver(arch, grid;
                                        lower_diagonal = lower_diagonal,
                                              diagonal = diagonal,
                                        upper_diagonal = upper_diagonal)

    # Need buffer for index permutations and transposes.
    buffer_needed = arch isa GPU && Bounded in (TX, TY)
    buffer = buffer_needed ? similar(sol_storage) : nothing

    # Storage space for right hand side of Poisson equation
    rhs = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))

    return FourierTridiagonalPoissonSolver(arch, grid, btsolver, rhs, sol_storage, buffer, transforms)
end

function solve!(x, solver::FourierTridiagonalPoissonSolver, b=nothing)
    !isnothing(b) && set_source_term!(solver, b) # otherwise, assume source term is set correctly

    arch = solver.architecture
    ϕ = solver.storage

    # Apply forward transforms in order
    [transform!(solver.source_term, solver.buffer) for transform! in solver.transforms.forward]

    # Solve tridiagonal system of linear equations in z at every column.
    solve!(ϕ, solver.batched_tridiagonal_solver, solver.source_term)

    # Apply backward transforms in order
    [transform!(ϕ, solver.buffer) for transform! in solver.transforms.backward]

    ϕ .= real.(ϕ)

    # Set the volume mean of the solution to be zero.
    # Solutions to Poisson's equation are only unique up to a constant (the global mean
    # of the solution), so we need to pick a constant. We choose the constant to be zero
    # so that the solution has zero-mean.
    ϕ .= ϕ .- mean(ϕ)

    copy_event = launch!(arch, solver.grid, :xyz, copy_real_component!, x, ϕ, dependencies=device_event(arch))
    wait(device(arch), copy_event)

    return nothing
end

"""
    set_source_term!(solver, source_term)

Sets the source term in the discrete Poisson equation `solver`
to `source_term` by multiplying it by the vertical grid spacing at z cell centers.
"""
function set_source_term!(solver::FourierTridiagonalPoissonSolver, source_term)
    grid = solver.grid
    arch = solver.architecture
    solver.source_term .= source_term

    event = launch!(arch, grid, :xyz, multiply_by_Δzᵃᵃᶜ!, solver.source_term, grid, dependencies=Event(device(arch)))
    wait(device(arch), event)

    return nothing
end

@kernel function multiply_by_Δzᵃᵃᶜ!(a, grid)
    i, j, k = @index(Global, NTuple)
    a[i, j, k] *= Δzᵃᵃᶜ(i, j, k, grid)
end
