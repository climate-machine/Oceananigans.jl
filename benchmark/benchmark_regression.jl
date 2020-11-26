using PkgBenchmark
using Oceananigans
using Benchmarks

baseline = BenchmarkConfig(id="master")
script = joinpath(@__DIR__, "benchmarkable_incompressible_model.jl")
resultfile = joinpath(@__DIR__, "regression_benchmarks.json")

print_system_info()

judgement = judge(Oceananigans, baseline, script=script, resultfile=resultfile, verbose=true)
results = PkgBenchmark.benchmarkgroup(judgement)

for (case, trial) in results
    println("Results for $case")
    display(trial)
end

