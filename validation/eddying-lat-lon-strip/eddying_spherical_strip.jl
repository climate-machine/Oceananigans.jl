# # Barotropic gyre

using Oceananigans
using Oceananigans.Grids

using Oceananigans.Fields: FunctionField

using Oceananigans.Coriolis:
    HydrostaticSphericalCoriolis,
    VectorInvariantEnergyConserving,
    VectorInvariantEnstrophyConserving

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    VerticalVorticityField,
    VectorInvariant,
    ExplicitFreeSurface,
    ImplicitFreeSurface

using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity
using Oceananigans.Utils: prettytime, hours, day, days, years, year
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

using Statistics
using JLD2
using Printf

using Oceananigans.AbstractOperations: AbstractGridMetric, _unary_operation

latitude = (-60, 60)
Δφ = latitude[2] - latitude[1]

resolution = 1/6 # degree
Nx = round(Int, 360 / resolution)
Ny = round(Int, Δφ / resolution)

# A spherical domain
@show grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                          longitude = (-180, 180),
                                          latitude = latitude,
                                          z = (-100, 0))

#free_surface = ImplicitFreeSurface(gravitational_acceleration=0.1)
free_surface = ExplicitFreeSurface(gravitational_acceleration=0.2)

coriolis = HydrostaticSphericalCoriolis(scheme = VectorInvariantEnstrophyConserving())

@show const νh₀ = 5e3 * (60 / grid.Nx)^2
@inline νh(λ, φ, z, t) = νh₀ * cos(π * φ / 180)
variable_horizontal_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=νh)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    architecture = GPU(),
                                    momentum_advection = VectorInvariant(),
                                    free_surface = free_surface,
                                    coriolis = nothing, # coriolis,
                                    tracers = nothing,
                                    buoyancy = nothing,
                                    closure = variable_horizontal_diffusivity)

g = model.free_surface.gravitational_acceleration

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

# Time-scale for gravity wave propagation across the smallest grid cell
wave_propagation_time_scale = min(grid.radius * cosd(maximum(abs, grid.φᵃᶜᵃ)) * deg2rad(grid.Δλ),
                                  grid.radius * deg2rad(grid.Δφ)) / gravity_wave_speed

set!(model,
     u = (x, y, z) -> sqrt(abs(sin(π * y / 180))) * rand(),
     v = (x, y, z) -> sqrt(abs(sin(π * y / 180))) * rand())

# Zero out mean motion
model.velocities.u .-= mean(model.velocities.u)
model.velocities.v .-= mean(model.velocities.v)

# Set target velocity to fraction of free surface velocity
max_u = maximum(model.velocities.u)
max_v = maximum(model.velocities.v)
max_speed = sqrt(max_u^2 + max_v^2)

target_speed = 0.5 * gravity_wave_speed
model.velocities.u ./= target_speed / max_speed
model.velocities.v ./= target_speed / max_speed

mutable struct Progress; interval_start_time::Float64; end

function (p::Progress)(sim)
    wall_time = (time_ns() - p.interval_start_time) * 1e-9

    @info @sprintf("Time: %s, iteration: %d, max(u): %.2e m s⁻¹, wall time: %s",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   maximum(sim.model.velocities.u),
                   prettytime(wall_time))

    p.interval_start_time = time_ns()

    return nothing
end

ζ = VerticalVorticityField(model)
compute!(ζ)

#=
@inline f_func(λ, φ, z, Ω) = ifelse(φ == 0, Inf, 2Ω * sin(π * φ / 180))
f = FunctionField{Face, Face, Center}(f_func, grid, parameters=model.coriolis.rotation_rate)
Ro = ComputedField(ζ / f)
compute!(Ro)
=#

#=
Δt = TimeStepWizard(cfl = 0.2,
                    max_Δt = 0.2wave_propagation_time_scale, 
                    Δt = 0.2wave_propagation_time_scale,
                    cell_advection_timescale = Oceananigans.Diagnostics.accurate_cell_advection_timescale)
=#

Δt = 0.2wave_propagation_time_scale

# Max Rossby number: $(maximum(abs, Ro))

@info """
    Maximum vertical vorticity: $(maximum(ζ))
    Inverse maximum vertical vorticity: $(prettytime(1/maximum(ζ)))
    Minimum wave propagation time scale: $(prettytime(wave_propagation_time_scale))
    Time step: $(prettytime(Δt))
"""

simulation = Simulation(model,
                        Δt = Δt,
                        stop_time = 10year,
                        iteration_interval = 100,
                        progress = Progress(time_ns()))

output_fields = merge(model.velocities, (η=model.free_surface.η, ζ=ζ))

output_prefix = "eddying_strip_$(grid.Nx)_Ny$(grid.Ny)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (ζ = ζ,),
                                                      schedule = TimeInterval(60day),
                                                      prefix = output_prefix,
                                                      force = true)

run!(simulation)

#####
##### Animation!
#####

#include("visualize.jl")
#visualize_plots(simulation.output_writers[:fields].filepath)