using Printf
using Plots
using JULES
using Oceananigans
using Oceananigans.Advection

using Oceananigans.Grids: Cell, xnodes

ENV["GKSwstype"] = "100"

N = 512
L = 1
T = 0.25
Δt = 1e-3
Nt = Int(T/Δt)

topo = (Bounded, Periodic, Periodic)
domain = (x=(0, L), y=(0, 1), z=(0, 1))
grid = RegularCartesianGrid(topology=topo, size=(N, 1, 1), halo=(4, 4, 4); domain...)

model = CompressibleModel(
                      grid = grid,
                 advection = WENO5(),
                     gases = DryEarth(),
    thermodynamic_variable = Entropy(),
                   closure = IsotropicDiffusivity(ν=0, κ=0)
)

gas = model.gases.ρ
R, cₚ, cᵥ = gas.R, gas.cₚ, gas.cᵥ
u₀₀, T₀₀, ρ₀₀, s₀₀ = gas.u₀, gas.T₀, gas.ρ₀, gas.s₀
g  = model.gravity

ρₗ, ρᵣ = 1.0, 0.125
pₗ, pᵣ = 1.0, 0.1
uₗ, uᵣ = 0.0, 0.0

ρ₀(x, y, z) = x < 0.5 ? ρₗ : ρᵣ
p₀(x, y, z) = x < 0.5 ? pₗ : pᵣ
u₀(x, y, z) = x < 0.5 ? uₗ : uᵣ

T₀(x, y, z) = p₀(x, y, z) / (R*ρ₀(x, y, z))
ρs₀(x, y, z) = ρ₀(x, y, z) * (s₀₀ + cᵥ * log(T₀(x, y, z)/T₀₀) - R * log(ρ₀(x, y, z)/ρ₀₀))

set!(model.tracers.ρ, ρ₀)
set!(model.tracers.ρs, ρs₀)
update_total_density!(model)

anim = @animate for n in 1:Nt
    @info "iteration $n/$Nt"
    time_step!(model, Δt)

    title = @sprintf("Shock tube t=%.3f", model.clock.time)

    x = xnodes(Cell, grid)
    ρ = interior(model.total_density)[:]
    plot(x, ρ, lw=2, label="", title=title, xlims=(0, 1), ylims=(0, 1), dpi=200)
end

mp4(anim, "shock_tube.mp4", fps=60)
