push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using JLD2
using BenchmarkTools
using Benchmarks
using Plots
pyplot()

threads = (1, 2, 4, 8, 16, 32)

grid_size(T) = (8192, 512T)

# Run benchmarks

print_system_info()

for T in threads
    Nx, Ny = grid_size(T)
    @info "Benchmarking serial shallow water model weak scaling with threading [N=($Nx, $Ny), T=$T]..."
    julia = Base.julia_cmd()
    run(`$julia --threads $T --project weak_scaling_shallow_water_model_serial.jl $Nx $Ny`)
end

# Collect and merge benchmarks from all ranks

suite = BenchmarkGroup(["size", "threads"])

for T in threads
    Nx, Ny = grid_size(T)
    case = ((Nx, Ny), T)

    filename = string("weak_scaling_shallow_water_model_threads$(T).jld2")
    file = jldopen(filename, "r")
    suite[case] = file["trial"]
end

plot_keys = collect(keys(suite))
sort!(plot_keys, by = v -> v[2])
plot_num = length(plot_keys)
thread_num = zeros(Int64, plot_num)
run_times = zeros(Float64, plot_num)
eff_ratio = zeros(Float64, plot_num)
for i in 1:plot_num
    thread_num[i] = plot_keys[i][2]
    run_times[i] = mean(suite[plot_keys[i]].times) / 1.0e6
    eff_ratio[i] = median(suite[plot_keys[1]].times) / median(suite[plot_keys[i]].times)
end

plt = plot(thread_num, run_times, lw=4, xaxis=:log2, legend=:none,
          xlabel="Threads", ylabel="Times (ms)", title="Weak Scaling Shallow Water Threaded Times")
display(plt)
savefig(plt, "ws_shallow_water_thread_times.png")


plt2 = plot(thread_num, eff_ratio, lw=4, xaxis=:log2, legend=:none, ylims=(0,1.1),
            xlabel="Threads", ylabel="Efficiency", title="Weak Scaling Shallow Water Threaded Efficiency")
display(plt2)
savefig(plt2, "ws_shallow_water_threaded_efficiency.png")

# Summarize benchmarks

df = benchmarks_dataframe(suite)
sort!(df, :threads)
benchmarks_pretty_table(df, title="Shallow water model weak scaling with multithreading benchmark")

base_case = (grid_size(1), threads[1])
suite_Δ = speedups_suite(suite, base_case=base_case)
df_Δ = speedups_dataframe(suite_Δ, slowdown=true, efficiency=:weak, base_case=base_case)
sort!(df_Δ, :threads)
benchmarks_pretty_table(df_Δ, title="Shallow water model weak multithreading scaling speedup")
