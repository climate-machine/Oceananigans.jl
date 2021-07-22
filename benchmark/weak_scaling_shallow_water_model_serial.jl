push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Logging
using JLD2
using BenchmarkTools
using Benchmarks

using Oceananigans
using Oceananigans.Models
using CUDA

Logging.global_logger(OceananigansLogger())

Nx = parse(Int, ARGS[1])
Ny = parse(Int, ARGS[2])

T = Threads.nthreads()

@info "Setting up serial shallow water model with N=($Nx, $Ny) grid points and $T threads..."

topo = (Periodic, Periodic, Flat)   # Use Flat
grid = RegularRectilinearGrid(topology=topo, size=(Nx, Ny), extent=(1, 1))
model = ShallowWaterModel(architecture=CPU(), grid=grid, gravitational_acceleration=1.0)
set!(model, h=1.0)

@info "Warming up serial shallow water model..."

time_step!(model, 1) # warmup

@info "Benchmarking serial shallow water model..."

trial = @benchmark begin
    CUDA.@sync blocking=true time_step!($model, 1)
end samples=10 evals=1

t_median = BenchmarkTools.prettytime(median(trial).time)
@info "Done benchmarking. Median time: $t_median"

jldopen("weak_scaling_shallow_water_model_threads$T.jld2", "w") do file
    file["trial"] = trial
end
