# # Hello, ocean!
#
# > How inappropriate to call this planet Earth, when it is quite clearly _Ocean_.
#
#   --Arthur C. Clark

using Oceananigans, Oceananigans.Grids, Plots

grid = RegularCartesianGrid(size = (64, 1, 64),
                            x = (-5, 5), y = (-5, 5), z = (-3, 1),
                            topology = (Periodic, Periodic, Bounded))

model = IncompressibleModel(grid = grid,
                            architecture = CPU(),
                            advection = Oceananigans.Advection.WENO5(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = IsotropicDiffusivity(ν=1e-3, κ=1e-3))

@info "Simulating the ocean with" model

Σ(ξ) = (1 - tanh(ξ)) / 2
Π(ξ, δ=1) = (Σ(ξ - δ) - Σ(ξ + δ)) / 2

set!(model,
     u = (x, y, z) -> Σ(-z),
     b = (x, y, z) -> - Π(4x, 1) * Σ(32z))

run!(Simulation(model, Δt=0.01, stop_iteration=200))

# Analyze the data

b = model.tracers.b

@show maximum(abs, b.data)

plt = contourf(xnodes(b), znodes(b), interior(b)[:, 1, :]',
               xlabel = "x", ylabel = "z", title = "Hello, ocean!",
               xlim = (grid.xF[1], grid.xF[end]), ylim = (grid.zF[1], grid.zF[end]),
               aspectratio = :equal, linewidth = 0)

display(plt) # hide
