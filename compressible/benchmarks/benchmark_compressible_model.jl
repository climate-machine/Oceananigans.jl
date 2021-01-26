using Printf

using BenchmarkTools
using CUDA
using DataFrames
using PrettyTables

using Oceananigans
using Oceananigans.Architectures
using JULES

using BenchmarkTools: prettytime, prettymemory

Archs = [CPU]
@hascuda Archs = [CPU, GPU]

Ns = [32, 192]
Tvars = [Energy, Entropy]
Gases = [DryEarth, DryEarth3]

suite = BenchmarkGroup()

sync_step!(model) = time_step!(model, 1)
sync_step!(model::CompressibleModel{GPU}) = CUDA.@sync time_step!(model, 1)

for Arch in Archs, N in Ns, Gas in Gases, Tvar in Tvars
    @info "Running static atmosphere benchmark [$Arch, N=$N, $Tvar, $Gas]..."

    grid = RegularCartesianGrid(size=(N, N, N), halo=(2, 2, 2), extent=(1, 1, 1))
    model = CompressibleModel(architecture=Arch(), grid=grid, thermodynamic_variable=Tvar(), gases=Gas())

    sync_step!(model) # warmup

    b = @benchmark sync_step!($model) samples=10
    display(b)

    key = (Arch, N, Gas, Tvar)
    suite[key] = b
end

function benchmarks_to_dataframe(suite)
    df = DataFrame(architecture=[], size=[], gases=[],
                   thermodynamic_variable=[], min=[], median=[],
                   mean=[], max=[], memory=[], allocs=[])
    
    for Arch in Archs, N in Ns, Gas in Gases, Tvar in Tvars
        b = suite[Arch, N, Gas, Tvar]

        d = Dict(
            "architecture" => Arch,
            "size" => "$(N)³",
            "gases" => Gas,
            "thermodynamic_variable" => Tvar,
            "min" => minimum(b.times) |> prettytime,
            "median" => median(b.times) |> prettytime,
            "mean" => mean(b.times) |> prettytime,
            "max" => maximum(b.times) |> prettytime,
            "memory" => prettymemory(b.memory),
            "allocs" => b.allocs
        )

        push!(df, d)
    end

    return df
end

header = ["Arch" "Size" "Gases" "ThermoVar" "min" "median" "mean" "max" "memory" "allocs"]

df = benchmarks_to_dataframe(suite)
pretty_table(df, header, title="Static atmosphere benchmarks", nosubheader=true)

function gpu_speedups(suite)
    df = DataFrame(size=[], gases=[], thermodynamic_variable=[], speedup=[])

    for N in Ns, Gas in Gases, Tvar in Tvars
        b_cpu = suite[CPU, N, Gas, Tvar] |> median
        b_gpu = suite[GPU, N, Gas, Tvar] |> median
        b_ratio = ratio(b_cpu, b_gpu)

        d = Dict(
            "size" => "$(N)³",
            "gases" => Gas,
            "thermodynamic_variable" => Tvar,
            "speedup" => @sprintf("%.3fx", b_ratio.time)
        )

        push!(df, d)
    end

    return df
end

header = ["Size" "Gases" "ThermoVar" "speedup"]

df = gpu_speedups(suite)
pretty_table(df, header, title="Static atmosphere speedups", nosubheader=true)
