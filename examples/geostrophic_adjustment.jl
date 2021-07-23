# # Geostrophic adjustment using Oceananigans.HydrostaticFreeSurfaceModel
#
# This example demonstrates how to simulate the one-dimensional geostrophic adjustment of a
# free surface using `Oceananigans.HydrostaticFreeSurfaceModel`. Here, we solve the hydrostatic
# Boussinesq equations beneath a free surface with a small-amplitude about rest ``z = 0``,
# with boundary conditions expanded around ``z = 0``, and free surface dynamics linearized under
# the assumption ``η / H \ll 1``, where ``η`` is the free surface displacement, and ``H`` is
# the total depth of the fluid.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, JLD2, Plots"
# ```
# 
# The HydrostaticFreeSurfaceModel is still "experimental". This means some of its features
# are not exported, such as the ImplicitFreeSurface, and must be brought into scope manually:

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface

# ## A one-dimensional domain
#
# We use a one-dimensional domain of geophysical proportions,

grid = RegularRectilinearGrid(size = (128, 1),
                              x = (0, 1000kilometers), z = (-400meters, 0),
                              topology = (Bounded, Flat, Bounded))

# !!! note
#   We always have to include the z-direction for `HydrostaticFreeSurfaceModel`, even if
#   the model is "barotropic" with just one grid point in z.
#
# We deploy a Coriolis parameter appropriate for the mid-latitudes on Earth,

coriolis = FPlane(f=1e-4)

# ## Building a `HydrostaticFreeSurfaceModel`
#
# We use `grid` and `coriolis` to build a simple `HydrostaticFreeSurfaceModel`,

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    coriolis = coriolis,
                                    free_surface = ImplicitFreeSurface())

# ## A geostrophic adjustment initial value problem
#
# We pose a geostrophic adjustment problem that consists of a partially-geostrophic
# Gaussian height field complemented by a geostrophic ``y``-velocity,

gaussian(x, L) = exp(-x^2 / 2L^2)

U = 0.1 # geostrophic velocity
L = grid.Lx / 40 # Gaussian width
x₀ = grid.Lx / 4 # Gaussian center

vᵍ(x, y, z) = - U * (x - x₀) / L * gaussian(x - x₀, L)

g = model.free_surface.gravitational_acceleration

η₀ = coriolis.f * U * L / g # geostrohpic free surface amplitude

ηᵍ(x) = η₀ * gaussian(x - x₀, L)

# We use an initial height field that's twice the geostrophic solution,
# thus superimposing a geostrophic and ageostrophic component in the free
# surface displacement field:

ηⁱ(x, y) = 2 * ηᵍ(x)

# We set the initial condition to ``vᵍ`` and ``ηⁱ``,

set!(model, v=vᵍ, η=ηⁱ)

# ## Running a `Simulation`
#
# We pick a time-step that resolves the surface dynamics,

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

wave_propagation_time_scale = model.grid.Δx / gravity_wave_speed

simulation = Simulation(model, Δt = 0.1wave_propagation_time_scale, stop_iteration = 1000)

# ## Output
#
# We output the velocity field and free surface displacement,

output_fields = merge(model.velocities, (η=model.free_surface.η,))

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = IterationInterval(10),
                                                      prefix = "geostrophic_adjustment",
                                                      field_slicer = nothing,
                                                      force = true)

run!(simulation)

# ## Visualizing the results

using Oceananigans.OutputReaders: FieldTimeSeries
using Plots, Printf

u_timeseries = FieldTimeSeries("geostrophic_adjustment.jld2", "u")
v_timeseries = FieldTimeSeries("geostrophic_adjustment.jld2", "v")
η_timeseries = FieldTimeSeries("geostrophic_adjustment.jld2", "η")

xη = xw = xv = xnodes(v_timeseries)
xu = xnodes(u_timeseries)

t = u_timeseries.times

anim = @animate for i = 1:length(t)

    u = interior(u_timeseries[i])[:, 1, 1]
    v = interior(v_timeseries[i])[:, 1, 1]
    η = interior(η_timeseries[i])[:, 1, 1]

    titlestr = @sprintf("Geostrophic adjustment at t = %.1f hours", t[i] / hours)

    u_plot = plot(xu / kilometers, u, linewidth = 2,
                  label = "", xlabel = "x (km)", ylabel = "u (m s⁻¹)", ylims = (-2e-3, 2e-3))

    v_plot = plot(xv / kilometers, v, linewidth = 2, title = titlestr,
                  label = "", xlabel = "x (km)", ylabel = "v (m s⁻¹)", ylims = (-U, U))

    η_plot = plot(xη / kilometers, η, linewidth = 2,
                  label = "", xlabel = "x (km)", ylabel = "η (m)", ylims = (-η₀/10, 2η₀))

    plot(v_plot, u_plot, η_plot, layout = (3, 1), size = (800, 600))
end

mp4(anim, "geostrophic_adjustment.mp4", fps = 15) # hide
