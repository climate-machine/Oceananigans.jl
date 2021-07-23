module Distributed

export
    MultiCPU, MultiGPU, child_architecture,
    HaloCommunication, HaloCommunicationBC,
    inject_halo_communication_boundary_conditions,
    DistributedFFTBasedPoissonSolver,
    DistributedNonhydrostaticModel, DistributedShallowWaterModel

using MPI

using Oceananigans.Models
using Oceananigans.Utils

include("distributed_utils.jl")
include("multi_architectures.jl")
include("distributed_kernel_launching.jl")
include("halo_communication_bcs.jl")
include("halo_communication.jl")
include("distributed_fields.jl")
include("distributed_fft_based_poisson_solver.jl")
include("distributed_solve_for_pressure.jl")
include("distributed_nonhydrostatic_model.jl")
include("distributed_shallow_water_model.jl")

end # module
