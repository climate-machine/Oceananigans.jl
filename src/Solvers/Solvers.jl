module Solvers

export
    BatchedTridiagonalSolver, solve!,
    FFTBasedPoissonSolver,
    FourierTridiagonalPoissonSolver,
    PreconditionedConjugateGradientSolver

using Statistics
using FFTW
using CUDA
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Architectures: device, CPU, GPU, array_type, arch_array
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.BoundaryConditions
using Oceananigans.Fields

using Oceananigans.Grids: unpack_grid

"""
    ω(M, k)

Return the `M`th root of unity raised to the `k`th power.
"""
@inline ω(M, k) = exp(-2im*π*k/M)

reshaped_size(N, dim) = dim == 1 ? (N, 1, 1) :
                        dim == 2 ? (1, N, 1) :
                        dim == 3 ? (1, 1, N) : nothing

include("batched_tridiagonal_solver.jl")
include("poisson_eigenvalues.jl")
include("index_permutations.jl")
include("discrete_transforms.jl")
include("plan_transforms.jl")
include("fft_based_poisson_solver.jl")
include("fourier_tridiagonal_poisson_solver.jl")
include("preconditioned_conjugate_gradient_solver.jl")

end
