using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks

using Oceananigans.BoundaryConditions: NormalFlow

# Benchmark function

function benchmark_time_stepper(Arch, N, immersed_boundary)
    topo = (Periodic, Bounded, Bounded)
    grid = RegularRectilinearGrid(topology=topo, size=(N, N, 1), extent=(1, 1, 1))

    v_bcs = VVelocityBoundaryConditions(grid,
                                        north = BoundaryCondition(NormalFlow,1.0),
                                        south = BoundaryCondition(NormalFlow,1.0))

    model = IncompressibleModel(
        architecture=Arch(),
        grid=grid,
        timestepper=:RungeKutta3,
        boundary_conditions=(v=v_bcs,),
        immersed_boundary=immersed_boundary)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
Ns = [256]

const R = 1
const vCent = Float64[30, 20, 0]
dist_cylinder(v) = sqrt((vCent[1] - v[1])^2 + (vCent[2] - v[2])^2) - R

boundaries = [nothing, dist_cylinder]

# Run and summarize benchmarks

print_system_info()
suite = run_benchmarks(benchmark_time_stepper; Architectures, Ns, boundaries)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :Ns], by=(string, identity))
benchmarks_pretty_table(df, title="Immersed boundary benchmarks")
