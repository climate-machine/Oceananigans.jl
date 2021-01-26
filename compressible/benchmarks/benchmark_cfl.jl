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

Ns = [32, 128]
Tvars = [Energy, Entropy]
Gases = [DryEarth, DryEarth3]

tags = ["arch", "N", "gases", "tvar"]

suite = BenchmarkGroup(
    "cfl" => BenchmarkGroup(),
    "acoustic_cfl" => BenchmarkGroup()
)

_cfl(model) = cfl(model, 1)
_cfl(model::CompressibleModel{GPU}) = CUDA.@sync cfl(model, 1)
_acoustic_cfl(model) = acoustic_cfl(model, 1)
_acoustic_cfl(model::CompressibleModel{GPU}) = CUDA.@sync acoustic_cfl(model, 1)

for Arch in Archs, N in Ns, Gas in Gases, Tvar in Tvars
    @info "Running CFL benchmark [$Arch, N=$N, $Tvar, $Gas]..."

    grid = RegularCartesianGrid(size=(N, N, N), extent=(1, 1, 1))
    model = CompressibleModel(architecture=Arch(), grid=grid, thermodynamic_variable=Tvar(),
                              gases=Gas())

    # warmup
    _cfl(model)
    _acoustic_cfl(model)

    b_cfl = @benchmark _cfl($model) samples=10
    display(b_cfl)

    b_acfl = @benchmark _acoustic_cfl($model) samples=10
    display(b_acfl)

    key = (Arch, N, Gas, Tvar)
    suite["cfl"][key] = b_cfl
    suite["acoustic_cfl"][key] = b_acfl
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

df = benchmarks_to_dataframe(suite["cfl"])
pretty_table(df, header, title="CFL benchmarks", nosubheader=true)

df = benchmarks_to_dataframe(suite["acoustic_cfl"])
pretty_table(df, header, title="Acoustic CFL benchmarks", nosubheader=true)

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

df = gpu_speedups(suite["cfl"])
pretty_table(df, header, title="CFL speedups", nosubheader=true)

df = gpu_speedups(suite["acoustic_cfl"])
pretty_table(df, header, title="Acoustic CFL speedups", nosubheader=true)
