# # Double Gyre
#
# This example simulates a double gyre following:
# https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html

using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.BoundaryConditions
using Oceananigans.Advection
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations

using Oceananigans.Fields: PressureField

using Printf

grid = RegularCartesianGrid(size=(64, 64, 16), x=(-2e5, 2e5), y=(-3e5, 3e5), z=(-1e3, 0),
                            topology=(Bounded, Bounded, Bounded))

# ## Boundary conditions
#
# ### Wind stress

@inline wind_stress(x, y, t, p) = - p.τ * cos(2π * y / p.Ly)

surface_stress_u_bc = BoundaryCondition(Flux, wind_stress, parameters=(τ=1e-4, Ly=grid.Ly))

# ### Bottom drag

@inline bottom_drag_u(x, y, t, u, p) = - p.μ * p.Lz * u
@inline bottom_drag_v(x, y, t, v, p) = - p.μ * p.Lz * v

bottom_drag_u_bc = BoundaryCondition(Flux, bottom_drag_u, field_dependencies=:u, parameters=(μ=1/180day, Lz=grid.Lz))
bottom_drag_v_bc = BoundaryCondition(Flux, bottom_drag_v, field_dependencies=:v, parameters=(μ=1/180day, Lz=grid.Lz))

u_bcs = UVelocityBoundaryConditions(grid, top = surface_stress_u_bc, bottom = bottom_drag_u_bc)
v_bcs = VVelocityBoundaryConditions(grid, bottom = bottom_drag_v_bc)

# ### Buoyancy relaxation

@inline buoyancy_flux(x, y, t, b, p) = - p.μ * (b - p.Δb / p.Ly * y)

buoyancy_flux_bc = BoundaryCondition(Flux, buoyancy_flux,
                                     field_dependencies = :b,
                                     parameters = (μ=1/day, Δb=0.05, Ly=grid.Ly))

b_bcs = TracerBoundaryConditions(grid, top = buoyancy_flux_bc,
                                       bottom = BoundaryCondition(Value, 0))

# ## Turbulence closure
closure = AnisotropicDiffusivity(νh=500, νz=1e-2, κh=100, κz=1e-2)

# ## Model building

model = IncompressibleModel(architecture = CPU(),
                            timestepper = :RungeKutta3, 
                            advection = UpwindBiasedFifthOrder(),
                            grid = grid,
                            coriolis = BetaPlane(latitude=45),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            closure = closure,
                            boundary_conditions = (u=u_bcs, v=v_bcs, b=b_bcs))

# ## Initial conditions

bᵢ(x, y, z) = b_bcs.top.condition.parameters.Δb * (1 + z / grid.Lz)

set!(model, b=bᵢ)

# ## Simulation setup

max_Δt = 1 / 5model.coriolis.f₀

wizard = TimeStepWizard(cfl=1.0, Δt=hour/2, max_change=1.1, max_Δt=max_Δt)

# Finally, we set up and run the the simulation.

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

wall_clock = time_ns()

function print_progress(simulation)
    model = simulation.model

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(wizard.Δt),
                   umax(), vmax(), wmax(),
                   prettytime(1e-9 * (time_ns() - wall_clock))
                  )

    @info msg

    return nothing
end

simulation = Simulation(model, Δt=wizard, stop_time=365days, iteration_interval=10, progress=print_progress)

# ## Output

u, v, w = model.velocities
b = model.tracers.b

speed = ComputedField(u^2 + v^2)
buoyancy_variance = ComputedField(b^2)

outputs = merge(model.velocities, model.tracers, (speed=speed, b²=buoyancy_variance))

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(2days),
                                                      prefix = "double_gyre",
                                                      field_slicer = FieldSlicer(k=model.grid.Nz),
                                                      force = true)

p = PressureField(model)
barotropic_p = AveragedField(p, dims=3)
barotropic_u = AveragedField(model.velocities.u, dims=3)
barotropic_v = AveragedField(model.velocities.v, dims=3)

simulation.output_writers[:barotropic_velocities] =
    JLD2OutputWriter(model, (u=barotropic_u, v=barotropic_v, p=barotropic_p),
                     schedule = AveragedTimeInterval(30days, window=10days),
                     prefix = "double_gyre_circulation",
                     force = true)

run!(simulation)

# # A neat movie

x, y, z = nodes(model.velocities.u)

xlims = (-grid.Lx/2 * 1e-3, grid.Lx/2 * 1e-3)
ylims = (-grid.Ly/2 * 1e-3, grid.Ly/2 * 1e-3)

x_km = x * 1e-3
y_km = y * 1e-3

nothing # hide

# Next, we open the JLD2 file, and extract the iterations we ended up saving at,

using JLD2, Plots

file = jldopen("double_gyre.jld2")

iterations = parse.(Int, keys(file["timeseries/t"]))

# These utilities are handy for calculating nice contour intervals:

""" Returns colorbar levels equispaced from `(-clim, clim)` and encompassing the extrema of `c`. """
function divergent_levels(c, clim, nlevels=21)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    return ((-clim, clim), clim > cmax ? levels : levels = vcat([-cmax], levels, [cmax]))
end

""" Returns colorbar levels equispaced between `clims` and encompassing the extrema of `c`."""
function sequential_levels(c, clims, nlevels=20)
    levels = range(clims[1], stop=clims[2], length=nlevels)
    cmin, cmax = minimum(c), maximum(c)
    cmin < clims[1] && (levels = vcat([cmin], levels))
    cmax > clims[2] && (levels = vcat(levels, [cmax]))
    return clims, levels
end

# Finally, we're ready to animate.

@info "Making an animation from the saved data..."

anim = @animate for (i, iter) in enumerate(iterations)
    
    @info "Drawing frame $i from iteration $iter \n"

    t = file["timeseries/t/$iter"]
    u = file["timeseries/u/$iter"][:, :, 1]
    v = file["timeseries/v/$iter"][:, :, 1]
    s = file["timeseries/speed/$iter"][:, :, 1]

    ulims, ulevels = divergent_levels(u, 1.0)
    slims, slevels = sequential_levels(s, (0.0, 1.5))

    kwargs = (aspectratio=:equal, linewidth=0, xlims=xlims,
              ylims=ylims, xlabel="x (km)", ylabel="y (km)")

    u_plot = contourf(x_km, y_km, u';
                      color = :balance,
                      clims = ulims,
                      levels = ulevels,
                      kwargs...)
                        
    s_plot = contourf(x_km, y_km, s';
                      color = :thermal,
                      clims = slims,
                      levels = slevels,
                      kwargs...)
                             
    plot(u_plot, s_plot, size=(800, 500),
         title = ["u(t="*string(round(t/day, digits=1))*" day)" "speed"])

    iter == iterations[end] && close(file)
end

gif(anim, "double_gyre.gif", fps = 8) # hide

# Plot the barotropic circulation

xv, yv, zv = nodes(model.velocities.v)

xv_km, yv_km = xv * 1e-3, yv * 1e-3

file = jldopen("double_gyre_circulation.jld2")

last_iteration = parse(Int, last(keys(file["timeseries/t"])))

U = file["timeseries/u/$last_iteration"][:, :, 1]
V = file["timeseries/v/$last_iteration"][:, :, 1]

U_plot = contourf(x_km, y_km, U', xlims=xlims, ylims=ylims,
                  linewidth=0, color=:balance, aspectratio=:equal)

V_plot = contourf(xv_km, yv_km, V', xlims=xlims, ylims=ylims,
                  linewidth=0, color=:balance, aspectratio=:equal)

plot(U_plot, V_plot, size=(800, 500),
     title=["Depth- and time-averaged \$ u \$" "Depth- and time-averaged \$ v \$"])

savefig("double_gyre_circulation.png") # hide
