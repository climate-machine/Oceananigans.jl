# # Freely decaying barotropic turbulence on a latitude-longitude strip

using Oceananigans
using Oceananigans.Grids

using Oceananigans.BoundaryConditions: fill_halo_regions!

using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    VerticalVorticityField,
    VectorInvariant,
    ExplicitFreeSurface,
    ImplicitFreeSurface

using Oceananigans.TurbulenceClosures:
    HorizontallyCurvilinearAnisotropicDiffusivity,
    HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity

using Oceananigans.Utils: prettytime, hours, day, days, years, year
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval
using Oceananigans.AbstractOperations: KernelFunctionOperation

using Statistics
using JLD2
using Printf

import Oceananigans.Architectures: architecture
architecture(ibg::ImmersedBoundaryGrid) = architecture(ibg.grid)

#####
##### Grid
#####

latitude = (-30, 30)
Δφ = latitude[2] - latitude[1]

resolution = 4 # degree
Nx = round(Int, 360 / resolution)
Ny = round(Int, Δφ / resolution)
Nz = 10

# A spherical domain
underlying_grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, Nz),
                                          longitude = (-180, 180),
                                          latitude = latitude,
                                          halo = (2, 2, 2),
                                          z = (-3000, 0))


const ridge_height = 500.0    # [m]
const ridge_base   = -3000.0  # [m]
@inline height(x, y, L) = ridge_base + ridge_height * exp(-L * (x/180)^2)
@inline ridge(x, y, z) = z < height(x, y, 30) 

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(ridge))
# grid = underlying_grid

#####
##### Physics and model setup
#####

free_surface = ExplicitFreeSurface(gravitational_acceleration=1.0)

equatorial_Δx = grid.radius * deg2rad(grid.Δλ)
diffusive_time_scale = 120days

@show const νh₂ = 1e2 * equatorial_Δx^2 / diffusive_time_scale
@show const νh₄ = 1e-5 * equatorial_Δx^4 / diffusive_time_scale

closure = HorizontallyCurvilinearAnisotropicDiffusivity(νh=νh₂)
# closure = HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity(νh=νh₄)

coriolis = HydrostaticSphericalCoriolis()
Ω = coriolis.rotation_rate / 20
coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    architecture = CPU(),
                                    momentum_advection = VectorInvariant(),
                                    free_surface = free_surface,
                                    coriolis = coriolis,
                                    tracers = nothing,
                                    buoyancy = nothing,
                                    closure = closure)

#####
##### Initial condition: two streamfunction
#####

# Random noise
ψ★ = Field(Face, Face, Center, model.architecture, model.grid)
set!(ψ★, (x, y, z) -> rand())
fill_halo_regions!(ψ★, model.architecture)

# Zonal wind
step(x, d, c) = 1/2 * (1 + tanh((x - c) / d))
polar_mask(y) = step(y, -5, 60) * step(y, 5, -60)
zonal_ψ(y) = (cosd(4y)^3 + 0.5 * exp(-y^2 / 200)) * polar_mask(y)

ψ̄ = Field(Face, Face, Center, model.architecture, model.grid)
set!(ψ̄, (x, y, z) -> zonal_ψ(y))
fill_halo_regions!(ψ̄, model.architecture)

ψ_total = 40 * ψ★ + Ny * ψ̄

u, v, w = model.velocities
η = model.free_surface.η

if !isnothing(model.coriolis)
    using Oceananigans.Coriolis: fᶠᶠᵃ
    f = KernelFunctionOperation{Face, Face, Center}(fᶠᶠᵃ, model.grid, parameters=model.coriolis)
    g = model.free_surface.gravitational_acceleration
    η .= f * ψ_total / g
end

u .= - ∂y(ψ_total)
v .= + ∂x(ψ_total)

#####
##### Shenanigans for rescaling the velocity field to
#####   1. Have a magnitude (ish) that's a fixed fraction of
#####      the surface gravity wave speed;
#####   2. Zero volume mean on the curvilinear RegularLatitudeLongitudeGrid.
#####

# Time-scale for gravity wave propagation across the smallest grid cell
g = model.free_surface.gravitational_acceleration
gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

minimum_Δx = grid.radius * cosd(maximum(abs, grid.φᵃᶜᵃ)) * deg2rad(grid.Δλ)
minimum_Δy = grid.radius * deg2rad(grid.Δφ)
wave_propagation_time_scale = min(minimum_Δx, minimum_Δy) / gravity_wave_speed

@info "Max speeds prior to rescaling:"
@show max_u = maximum(u)
@show max_v = maximum(v)
max_speed_ish = sqrt(max_u^2 + max_v^2)

target_speed = 0.5 * gravity_wave_speed
u .*= target_speed / max_speed_ish
v .*= target_speed / max_speed_ish

# Zero out mean motion
using Oceananigans.AbstractOperations: volume

u_cpu = XFaceField(CPU(), grid)
v_cpu = YFaceField(CPU(), grid)
set!(u_cpu, u)
set!(v_cpu, v)

@show max_u = maximum(u)
@show max_v = maximum(v)

u_dV = u_cpu * volume
u_reduced = AveragedField(u_dV, dims=(1, 2, 3))
compute!(u_reduced)
mean!(u_reduced, u_dV)
integrated_u = u_reduced[1, 1, 1]

v_dV = v_cpu * volume
v_reduced = AveragedField(v_dV, dims=(1, 2, 3))
mean!(v_reduced, v_dV)
integrated_v = v_reduced[1, 1, 1]

# Calculate total volume
u_cpu .= 1
v_cpu .= 1
compute!(u_reduced)
compute!(v_reduced)

u_volume = u_reduced[1, 1, 1]
v_volume = v_reduced[1, 1, 1]

@info "Max speeds prior zeroing out volume mean:"
@show maximum(u)
@show maximum(v)

u .-= integrated_u / u_volume
v .-= integrated_v / v_volume

@info "Initial max speeds:"
@show maximum(u)
@show maximum(v)

#####
##### Simulation setup
#####

ζ = VerticalVorticityField(model)
compute!(ζ)
Δt = 0.2 * minimum_Δx / gravity_wave_speed

@info """
    Maximum vertical vorticity: $(maximum(ζ))
    Inverse maximum vertical vorticity: $(prettytime(1/maximum(ζ)))
    Minimum wave propagation time scale: $(prettytime(wave_propagation_time_scale))
    Time step: $(prettytime(Δt))
"""

mutable struct Progress; interval_start_time::Float64; end

function (p::Progress)(sim)
    wall_time = (time_ns() - p.interval_start_time) * 1e-9

    compute!(ζ)

    @info @sprintf("Time: %s, iteration: %d, max(|ζ|): %.2e s⁻¹, wall time: %s",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   maximum(abs, ζ),
                   prettytime(wall_time))

    p.interval_start_time = time_ns()

    return nothing
end

simulation = Simulation(model,
                        Δt = Δt,
                        stop_time = 100year,
                        iteration_interval = 1000,
                        progress = Progress(time_ns()))

output_fields = merge(model.velocities, (η=model.free_surface.η, ζ=ζ))

output_prefix = "rotating_freely_decaying_barotropic_turbulence_Nx$(grid.Nx)_Ny$(grid.Ny)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (ζ = ζ,),
                                                      schedule = TimeInterval(10day),
                                                      prefix = output_prefix,
                                                      force = true)

# Let's goo!
run!(simulation)
