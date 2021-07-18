using CUDA
using Oceananigans.Architectures: AbstractCPUArchitecture, AbstractGPUArchitecture

import Base: zeros

zeros(FT, ::AbstractCPUArchitecture, N...) = zeros(FT, N...)
zeros(FT, ::AbstractGPUArchitecture, N...) = CUDA.zeros(FT, N...)

zeros(arch::AbstractArchitecture, grid, N...) = zeros(eltype(grid), arch, N...)
