using Printf
using Plots

using Oceananigans
using Oceananigans.Advection
using JULES

using JULES: IdealGas
using Oceananigans.Grids: Cell, xnodes

ENV["GKSwstype"] = "100"

N = 512
L = 1
T = 1e-1
Δt = 2e-4
Nt = Int(T/Δt)

ideal_gas = IdealGas(Float64, JULES.R⁰, 1; T₀=1, p₀=1, s₀=1, u₀=0)

topo = (Periodic, Periodic, Periodic)
domain = (x=(0, L), y=(0, 1), z=(0, 1))
grid = RegularCartesianGrid(topology=topo, size=(N, 1, 1), halo=(4, 4, 4); domain...)

model = CompressibleModel(
                      grid = grid,
                 advection = WENO5(),
                     gases = (ρ=ideal_gas,),
    thermodynamic_variable = Energy(),
                   closure = IsotropicDiffusivity(ν=0, κ=0),
                   gravity = 0.0
)

g  = model.gravity
gas = model.gases.ρ
R, cₚ, cᵥ = gas.R, gas.cₚ, gas.cᵥ
u₀₀, T₀₀, ρ₀₀, s₀₀ = gas.u₀, gas.T₀, gas.ρ₀, gas.s₀

ρ₀(x, y, z) = 1
p₀(x, y, z) = 1
T₀(x, y, z) = p₀(x, y, z) / (R*ρ₀(x, y, z))
ρe₀(x, y, z) = ρ₀(x, y, z) * (u₀₀ + cᵥ * (T₀(x, y, z) - T₀₀) + g*z)

@inline x′(x, t, L, U) = mod(x + L/2 - U * t, L) - L/2
@inline ϕ_Gaussian(x, t; L, U, a=1, c=1/8) = a * exp(-x′(x, t, L, U)^2 / (2c^2))
@inline ϕ_Square(x, t; L, U, w=0.15) = -w <= x′(x, t, L, U) <= w ? 1.0 : 0.0
@inline ϕ_Sine(x, t; L, U) = sin(2π * x)
ρu₀(x, y, z) = ϕ_Sine(x, 0, L=L, U=1)

set!(model.tracers.ρ, ρ₀)
set!(model.tracers.ρe, ρe₀)
set!(model.momenta.ρu, ρu₀)
update_total_density!(model)

anim = @animate for n in 1:Nt
    @info "iteration $n/$Nt"
    time_step!(model, Δt)

    title = @sprintf("Burgers equation t=%.3f", model.clock.time)

    x = xnodes(Cell, grid)
    ρu = interior(model.momenta.ρu)[:]
    plot(x, ρu, lw=2, label="", title=title, xlims=(0, L), ylims=(-2, 2), dpi=200)
end

mp4(anim, "burgers_equation.mp4", fps=60)
