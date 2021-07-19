# # Barotropic gyre

using Oceananigans
using Oceananigans.Grids
using Oceananigans.BoundaryConditions

using Oceananigans.Coriolis:
    HydrostaticSphericalCoriolis,
    VectorInvariantEnergyConserving,
    VectorInvariantEnstrophyConserving

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    VectorInvariant,
    ExplicitFreeSurface,
    ImplicitFreeSurface

using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity
using Oceananigans.Utils: prettytime, hours, day, days, years
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, RasterDepthMask

using Statistics
using JLD2
using Printf

Nx = 60
Ny = 60

# A spherical domain
underlying_grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                               longitude = (-30, 30),
                                               latitude = (15, 75),
                                               z = (-4000, 0))

@inline raster_depth(i, j) = 30 < i < 35 && 42 < j < 48

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(raster_depth, mask_type=RasterDepthMask()))

free_surface = ImplicitFreeSurface(gravitational_acceleration=0.1)
# free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1)

coriolis = HydrostaticSphericalCoriolis(scheme = VectorInvariantEnstrophyConserving())

@show surface_wind_stress_parameters = (τ₀ = 1e-4,
                                        Lφ = grid.Ly,
                                        φ₀ = 15)

surface_wind_stress(λ, φ, t, p) = p.τ₀ * cos(2π * (φ - p.φ₀) / p.Lφ)

surface_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress,
                                               parameters = surface_wind_stress_parameters)

μ = 1 / 60days

@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag,
                                         discrete_form = true,
                                         parameters = μ)

v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag,
                                         discrete_form = true,
                                         parameters = μ)

u_bcs = FieldBoundaryConditions(top = surface_wind_stress_bc,
                                bottom = u_bottom_drag_bc)

v_bcs = FieldBoundaryConditions(bottom = v_bottom_drag_bc)

@show const νh₀ = 5e3 * (60 / grid.Nx)^2

@inline νh(λ, φ, z, t) = νh₀ * cos(π * φ / 180)

variable_horizontal_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=νh)
constant_horizontal_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=νh₀)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    architecture = CPU(),
                                    momentum_advection = VectorInvariant(),
                                    free_surface = free_surface,
                                    coriolis = coriolis,
                                    boundary_conditions = (u=u_bcs, v=v_bcs),
                                    closure = constant_horizontal_diffusivity,
                                    #closure = variable_horizontal_diffusivity,
                                    tracers = nothing,
                                    buoyancy = nothing)

g = model.free_surface.gravitational_acceleration

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

# Time-scale for gravity wave propagation across the smallest grid cell
wave_propagation_time_scale = min(grid.radius * cosd(maximum(abs, grid.φᵃᶜᵃ)) * deg2rad(grid.Δλ),
                                  grid.radius * deg2rad(grid.Δφ)) / gravity_wave_speed

mutable struct Progress
    interval_start_time :: Float64
end

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

simulation = Simulation(model,
                        Δt = 3600,
                        stop_time = 1years,
                        iteration_interval = 100,
                        progress = Progress(time_ns()))

output_fields = merge(model.velocities, (η=model.free_surface.η,))

output_prefix = "barotropic_gyre_Nx$(grid.Nx)_Ny$(grid.Ny)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(10day),
                                                      prefix = output_prefix,
                                                      field_slicer = nothing,
                                                      force = true)

run!(simulation)

#####
##### Animation!
#####

include("visualize_barotropic_gyre.jl")

# visualize_barotropic_gyre(simulation.output_writers[:fields])
