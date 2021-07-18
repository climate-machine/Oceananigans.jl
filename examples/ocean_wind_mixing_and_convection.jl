# # [Wind- and convection-driven mixing in an ocean surface boundary layer](@id gpu_example)
#
# This example simulates mixing by three-dimensional turbulence in an ocean surface
# boundary layer driven by atmospheric winds and convection. It demonstrates:
#
#   * How to set-up a grid with varying spacing in the vertical direction
#   * How to use the `SeawaterBuoyancy` model for buoyancy with a linear equation of state.
#   * How to use a turbulence closure for large eddy simulation.
#   * How to use a function to impose a boundary condition.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, JLD2, Plots"
# ```

# We start by importing all of the packages and functions that we'll need for this
# example.

using Random
using Printf
using Plots
using JLD2

using Oceananigans
using Oceananigans.Units: minute, minutes, hour

# ## The grid
#
# We use 32³ grid points with 2 m grid spacing in the horizontal and
# varying spacing in the vertical, with higher resolution closer to the
# surface,

σ = 1.1 # stretching factor
Nz = 24
Lz = 32

hyperbolically_spaced_faces(k) = - Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ))

grid = VerticallyStretchedRectilinearGrid(size = (32, 32, Nz), 
                                          x = (0, 64),
                                          y = (0, 64),
                                          z_faces = hyperbolically_spaced_faces)

# We plot vertical spacing versus depth to inspect the prescribed grid stretching:

plot(grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz],
      marker = :circle,
      ylabel = "Depth (m)",
      xlabel = "Vertical spacing (m)",
      legend = nothing)

# ## Buoyancy that depends on temperature and salinity
#
# We use the `SeawaterBuoyancy` model with a linear equation of state,

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4, β=8e-4))

# where $α$ and $β$ are the thermal expansion and haline contraction
# coefficients for temperature and salinity.
#
# ## Boundary conditions
#
# We calculate the surface temperature flux associated with surface heating of
# 200 W m⁻², reference density `ρₒ`, and heat capacity `cᴾ`,

Qʰ = 200  # W m⁻², surface _heat_ flux
ρₒ = 1026 # kg m⁻³, average density at the surface of the world ocean
cᴾ = 3991 # J K⁻¹ kg⁻¹, typical heat capacity for seawater

Qᵀ = Qʰ / (ρₒ * cᴾ) # K m s⁻¹, surface _temperature_ flux

# Finally, we impose a temperature gradient `dTdz` both initially and at the
# bottom of the domain, culminating in the boundary conditions on temperature,

dTdz = 0.01 # K m⁻¹

T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵀ),
                                bottom = GradientBoundaryCondition(dTdz))

# Note that a positive temperature flux at the surface of the ocean
# implies cooling. This is because a positive temperature flux implies
# that temperature is fluxed upwards, out of the ocean.
#
# For the velocity field, we imagine a wind blowing over the ocean surface
# with an average velocity at 10 meters `u₁₀`, and use a drag coefficient `cᴰ`
# to estimate the kinematic stress (that is, stress divided by density) exerted
# by the wind on the ocean:

u₁₀ = 10     # m s⁻¹, average wind velocity 10 meters above the ocean
cᴰ = 2.5e-3  # dimensionless drag coefficient
ρₐ = 1.225   # kg m⁻³, average density of air at sea-level

Qᵘ = - ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀) # m² s⁻²

# The boundary conditions on `u` are thus

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

# For salinity, `S`, we impose an evaporative flux of the form

@inline Qˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S # [salinity unit] m s⁻¹
nothing # hide

# where `S` is salinity. We use an evporation rate of 1 millimeter per hour,

evaporation_rate = 1e-3 / hour # m s⁻¹

# We build the `Flux` evaporation `BoundaryCondition` with the function `Qˢ`,
# indicating that `Qˢ` depends on salinity `S` and passing
# the parameter `evaporation_rate`,

evaporation_bc = FluxBoundaryCondition(Qˢ, field_dependencies=:S, parameters=evaporation_rate)

# The full salinity boundary conditions are

S_bcs = FieldBoundaryConditions(top=evaporation_bc)

# ## Model instantiation
#
# We fill in the final details of the model here: upwind-biased 5th-order
# advection for momentum and tracers, 3rd-order Runge-Kutta time-stepping,
# Coriolis forces, and the `AnisotropicMinimumDissipation` closure
# for large eddy simulation to model the effect of turbulent motions at
# scales smaller than the grid scale that we cannot explicitly resolve.

model = IncompressibleModel(architecture = CPU(),
                            advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            grid = grid,
                            tracers = (:T, :S),
                            coriolis = FPlane(f=1e-4),
                            buoyancy = buoyancy,
                            closure = AnisotropicMinimumDissipation(),
                            boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))

# Notes:
#
# * To use the Smagorinsky-Lilly turbulence closure (with a constant model coefficient) rather than
#   `AnisotropicMinimumDissipation`, use `closure = ConstantSmagorinsky()` in the model constructor.
#
# * To change the `architecture` to `GPU`, replace `architecture = CPU()` with
#   `architecture = GPU()`.

# ## Initial conditions
#
# Our initial condition for temperature consists of a linear stratification superposed with
# random noise damped at the walls, while our initial condition for velocity consists
# only of random noise.

## Random noise damped at top and bottom
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise

## Temperature initial condition: a stable density gradient with random noise superposed.
Tᵢ(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 1e-6 * Ξ(z)

## Velocity initial condition: random noise scaled by the friction velocity.
uᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)

## `set!` the `model` fields using functions or constants:
set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=35)

# ## Setting up a simulation
#
# We first build a `TimeStepWizard` to ensure stable time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 1.0.

wizard = TimeStepWizard(cfl=1.0, Δt=10.0, max_change=1.1, max_Δt=1minute)

# Nice progress messaging is helpful:

start_time = time_ns() # so we can print the total elapsed wall time

## Print a progress message
progress_message(sim) =
    @printf("i: %04d, t: %s, Δt: %s, wmax = %.1e ms⁻¹, wall time: %s\n",
            sim.model.clock.iteration, prettytime(model.clock.time),
            prettytime(wizard.Δt), maximum(abs, sim.model.velocities.w),
            prettytime((time_ns() - start_time) * 1e-9))

# We then set up the simulation:

simulation = Simulation(model, Δt=wizard, stop_time=40minutes, iteration_interval=10,
                        progress=progress_message)

# ## Output
#
# We use the `JLD2OutputWriter` to save ``x, z`` slices of the velocity fields,
# tracer fields, and eddy diffusivities. The `prefix` keyword argument
# to `JLD2OutputWriter` indicates that output will be saved in
# `ocean_wind_mixing_and_convection.jld2`.

## Create a NamedTuple with eddy viscosity
eddy_viscosity = (νₑ = model.diffusivities.νₑ,)

simulation.output_writers[:slices] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                           prefix = "ocean_wind_mixing_and_convection",
                     field_slicer = FieldSlicer(j=Int(grid.Ny/2)),
                         schedule = TimeInterval(1minute),
                            force = true)

# We're ready:

run!(simulation)

# ## Turbulence visualization
#
# We animate the data saved in `ocean_wind_mixing_and_convection.jld2`.
# We prepare for animating the flow by creating coordinate arrays,
# opening the file, building a vector of the iterations that we saved
# data at, and defining functions for computing colorbar limits:

## Coordinate arrays
xw, yw, zw = nodes(model.velocities.w)
xT, yT, zT = nodes(model.tracers.T)

## Open the file with our data
file = jldopen(simulation.output_writers[:slices].filepath)

## Extract a vector of iterations
iterations = parse.(Int, keys(file["timeseries/t"]))

""" Returns colorbar levels equispaced between `(-clim, clim)` and encompassing the extrema of `c`. """
function divergent_levels(c, clim, nlevels=21)
    cmax = maximum(abs, c)
    levels = clim > cmax ? range(-clim, stop=clim, length=nlevels) : range(-cmax, stop=cmax, length=nlevels)
    return (levels[1], levels[end]), levels
end

""" Returns colorbar levels equispaced between `clims` and encompassing the extrema of `c`."""
function sequential_levels(c, clims, nlevels=20)
    levels = range(clims[1], stop=clims[2], length=nlevels)
    cmin, cmax = minimum(c), maximum(c)
    cmin < clims[1] && (levels = vcat([cmin], levels))
    cmax > clims[2] && (levels = vcat(levels, [cmax]))
    return clims, levels
end
nothing # hide

# We start the animation at `t = 10minutes` since things are pretty boring till then:

times = [file["timeseries/t/$iter"] for iter in iterations]
intro = searchsortedfirst(times, 10minutes)

anim = @animate for (i, iter) in enumerate(iterations[intro:end])

    @info "Drawing frame $i from iteration $iter..."

    t = file["timeseries/t/$iter"]
    w = file["timeseries/w/$iter"][:, 1, :]
    T = file["timeseries/T/$iter"][:, 1, :]
    S = file["timeseries/S/$iter"][:, 1, :]
    νₑ = file["timeseries/νₑ/$iter"][:, 1, :]

    wlims, wlevels = divergent_levels(w, 2e-2)
    Tlims, Tlevels = sequential_levels(T, (19.7, 19.99))
    Slims, Slevels = sequential_levels(S, (35, 35.005))
    νlims, νlevels = sequential_levels(νₑ, (1e-6, 5e-3))

    kwargs = (linewidth=0, xlabel="x (m)", ylabel="z (m)", aspectratio=1,
              xlims=(0, grid.Lx), ylims=(-grid.Lz, 0))

    w_plot = contourf(xw, zw, w'; color=:balance, clims=wlims, levels=wlevels, kwargs...)
    T_plot = contourf(xT, zT, T'; color=:thermal, clims=Tlims, levels=Tlevels, kwargs...)
    S_plot = contourf(xT, zT, S'; color=:haline,  clims=Slims, levels=Slevels, kwargs...)

    ## We use a heatmap for the eddy viscosity to observe how it varies on the grid scale.
    ν_plot = heatmap(xT, zT, νₑ'; color=:thermal, clims=νlims, levels=νlevels, kwargs...)

    w_title = @sprintf("vertical velocity (m s⁻¹), t = %s", prettytime(t))
    T_title = "temperature (ᵒC)"
    S_title = "salinity (g kg⁻¹)"
    ν_title = "eddy viscosity (m² s⁻¹)"

    ## Arrange the plots side-by-side.
    plot(w_plot, T_plot, S_plot, ν_plot, layout=(2, 2), size=(1200, 600),
         title=[w_title T_title S_title ν_title])

    iter == iterations[end] && close(file)
end

mp4(anim, "ocean_wind_mixing_and_convection.mp4", fps = 8) # hide
