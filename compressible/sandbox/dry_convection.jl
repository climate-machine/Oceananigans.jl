using Logging
using Printf
using Statistics
using NCDatasets
using CUDA

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.OutputWriters
using Oceananigans.Utils
using JULES

using Oceananigans.Fields: cpudata

Logging.global_logger(OceananigansLogger())

const km = kilometers
const hPa = 100.0

#####
##### Hydrostatic base state
#####

pₛ = 1000hPa
Tₛ = 300.0

θ₀(x, y, z) = Tₛ
p₀(x, y, z) = pₛ * (1 - g*z / (cₚ*Tₛ))^(cₚ/R)
T₀(x, y, z) = Tₛ * (p₀(x, y, z)/pₛ)^(R/cₚ)
ρ₀(x, y, z) = p₀(x, y, z) / (R*T₀(x, y, z))

#####
##### Model setup
#####

L = 2.56km
Δ = 40meters
N = Int(L/Δ)

topo = (Periodic, Periodic, Bounded)
domain = (x=(0, L), y=(0, L), z=(0, L))
grid = RegularCartesianGrid(topology=topo, size=(N, N, N), halo=(3, 3, 3); domain...)

gas = DryEarth()

# Sponge layer at the top with exponential decay length scale γ.
@inline damping_profile(z, L, γ) = exp(- (L - z) / γ)

forcing_params = (λ = 1/(10seconds), L=grid.Lz, γ=0.2grid.Lz)

@inline rayleigh_damping_ρu(i, j, k, grid, clock, fields, p) =
    @inbounds - p.λ * damping_profile(grid.zC[k], p.L, p.γ) * fields.ρu[i, j, k]

@inline rayleigh_damping_ρv(i, j, k, grid, clock, fields, p) =
    @inbounds - p.λ * damping_profile(grid.zC[k], p.L, p.γ) * fields.ρv[i, j, k]

@inline rayleigh_damping_ρw(i, j, k, grid, clock, fields, p) =
    @inbounds - p.λ * damping_profile(grid.zC[k], p.L, p.γ) * fields.ρw[i, j, k]

ρu_forcing = Forcing(rayleigh_damping_ρu, discrete_form=true, parameters=forcing_params)
ρv_forcing = Forcing(rayleigh_damping_ρv, discrete_form=true, parameters=forcing_params)
ρw_forcing = Forcing(rayleigh_damping_ρw, discrete_form=true, parameters=forcing_params)

# Exponentially decaying radiative forcing
@inline radiative_profile(z, Q₀, ℓ) = Q₀ * exp(-z/ℓ)

radiative_params = (Q₀=100.0, ℓ=100.0)

@inline radiative_forcing_ρe(i, j, k, grid, clock, fields, p) =
    @inbounds (radiative_profile(grid.zC[k], p.Q₀, p.ℓ) - radiative_profile(grid.zC[k+1], p.Q₀, p.ℓ)) / grid.Δz

ρe_forcing = Forcing(radiative_forcing_ρe, discrete_form=true, parameters=radiative_params)

model = CompressibleModel(
              architecture = GPU(),
                      grid = grid,
                 advection = WENO5(),
                     gases = gas,
    thermodynamic_variable = Energy(),
                   closure = IsotropicDiffusivity(ν=1e-2, κ=1e-2),
                   forcing = (ρu=ρu_forcing, ρv=ρv_forcing, ρw=ρw_forcing, ρe=ρe_forcing)
)

g  = model.gravity
R, cₚ, cᵥ = gas.ρ.R, gas.ρ.cₚ, gas.ρ.cᵥ
uᵣ, Tᵣ, ρᵣ, sᵣ = gas.ρ.u₀, gas.ρ.T₀, gas.ρ.ρ₀, gas.ρ.s₀
ρe₀(x, y, z) = ρ₀(x, y, z) * (uᵣ + cᵥ * (T₀(x, y, z) - Tᵣ) + g*z) + randn()

# Set initial state
set!(model.tracers.ρ, ρ₀)
set!(model.tracers.ρe, ρe₀)
update_total_density!(model)

function print_progress(simulation)
    model, Δt = simulation.model, simulation.Δt
    tvar = model.thermodynamic_variable

    progress = 100 * model.clock.time / simulation.stop_time
    message = @sprintf("[%05.2f%%] iteration = %d, time = %s, CFL = %.4e, acoustic CFL = %.4e",
                       progress, model.clock.iteration, prettytime(model.clock.time), cfl(model, Δt),
                       acoustic_cfl(model, Δt))

    @info message

    return nothing
end

simulation = Simulation(model, Δt=0.02second, stop_time=1hour, iteration_interval=20,
                        progress=print_progress)

fields = Dict(
    "ρ"  => model.total_density,
    "ρu" => model.momenta.ρu,
    "ρv" => model.momenta.ρv,
    "ρw" => model.momenta.ρw,
    "ρe" => model.tracers.ρe
)
    
simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, fields, filepath="dry_convection.nc", time_interval=10seconds)

# Save base state to NetCDF.
ds = simulation.output_writers[:fields].dataset
ds_ρ = defVar(ds, "ρ₀", Float32, ("xC", "yC", "zC"))
ds_ρe = defVar(ds, "ρe₀", Float32, ("xC", "yC", "zC"))

x, y, z = nodes((Cell, Cell, Cell), grid, reshape=true)
ds_ρ[:, :, :] = ρ₀.(x, y, z)
ds_ρe[:, :, :] = ρe₀.(x, y, z)

run!(simulation)
