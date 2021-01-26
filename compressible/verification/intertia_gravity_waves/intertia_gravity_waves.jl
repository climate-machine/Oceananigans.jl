using Printf
using PyPlot

using Oceananigans
using Oceananigans.Advection
using JULES

const km = 1000.0
const hPa = 100.0

L= 300km
H = 10km
Δ = 0.5km  # grid spacing [m]

Nx = Int(L/Δ)
Ny = 1
Nz = Int(H/Δ)

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), halo=(4, 4, 4),
                            x=(0, L), y=(0, L), z=(0, H))

model = CompressibleModel(
                      grid = grid,
                     gases = DryEarth(),
    thermodynamic_variable = Entropy(),
                   closure = IsotropicDiffusivity(ν=75.0, κ=75.0)
)

#####
##### Dry thermal bubble perturbation
#####

gas = model.gases.ρ
R, cₚ, cᵥ = gas.R, gas.cₚ, gas.cᵥ
g  = model.gravity
pₛ = 1000hPa
θₛ = 300.0
N² = 1e-4
Γ  = - N² * θₛ / g
U  = 0

# Define an approximately hydrostatic background state
θ₀(x, y, z) = θₛ + Γ * z
p₀(x, y, z) = pₛ * (1 - g / (cₚ*Γ) * log(θ₀(x, y, z)/θₛ))^(cₚ/R)
T₀(x, y, z) = θₛ * (p₀(x, y, z)/pₛ)^(R/cₚ)
ρ₀(x, y, z) = p₀(x, y, z) / (R*T₀(x, y, z))

# Define both energy and entropy
uᵣ, Tᵣ, ρᵣ, sᵣ = gas.u₀, gas.T₀, gas.ρ₀, gas.s₀  # Reference values
ρe₀(x, y, z) = ρ₀(x, y, z) * (uᵣ + cᵥ * (T₀(x, y, z) - Tᵣ) + g*z)
ρs₀(x, y, z) = ρ₀(x, y, z) * (sᵣ + cᵥ * log(T₀(x, y, z)/Tᵣ) - R * log(ρ₀(x, y, z)/ρᵣ))

# Define the initial density perturbation
Δθ₀ = 1e-2
xᶜ = 100km
a = 5km

θ′(x, y, z) = Δθ₀ * sin(π*z/H) / (1 + (x-xᶜ)^2/a^2)
ρ′(x, y, z) = ρ₀(x, y, z) * θ′(x, y, z) / θ₀(x, y, z)

# Define initial state
ρᵢ(x, y, z) = ρ₀(x, y, z) + ρ′(x, y, z)
pᵢ(x, y, z) = p₀(x, y, z)
Tᵢ(x, y, z) = pᵢ(x, y, z) / (R * ρᵢ(x, y, z))

ρuᵢ(x, y, z) = ρᵢ(x, y, z) * U
ρeᵢ(x, y, z) = ρᵢ(x, y, z) * (uᵣ + cᵥ * (Tᵢ(x, y, z) - Tᵣ) + g*z)
ρsᵢ(x, y, z) = ρᵢ(x, y, z) * (sᵣ + cᵥ * log(Tᵢ(x, y, z)/Tᵣ) - R * log(ρᵢ(x, y, z)/ρᵣ))

# Set initial state (which includes the thermal perturbation)
set!(model.tracers.ρ, ρᵢ)
set!(model.momenta.ρu, ρuᵢ)
set!(model.tracers.ρs, ρsᵢ)
update_total_density!(model)

T = 1000
Δt = 0.1
Nt = Int(T/Δt)
for n in 1:Nt
    @info "iteration = $n/$Nt, ρ=$(model.total_density[200, 1, 10])"
    time_step!(model, Δt)
end
