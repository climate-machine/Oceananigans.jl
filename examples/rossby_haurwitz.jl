# # Rossby Haurwitz solutions
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, JLD2, Plots"
# ```

# ## A spherical domain
#
# We use a one-dimensional domain of geophysical proportions,

using Oceananigans
using Oceananigans.Grids: RegularLatitudeLongitudeGrid
using Oceananigans.Utils: prettytime
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface    

h₀ = 8e3

grid = RegularLatitudeLongitudeGrid(size = (360, 160, 1), longitude = (-180, 180), latitude = (-80, 80), z = (-h₀, 0))
#  λ for latitude and ϕ for latitude is
using Oceananigans.Coriolis: VectorInvariantEnergyConserving, HydrostaticSphericalCoriolis

coriolis = HydrostaticSphericalCoriolis(Float64, scheme=VectorInvariantEnergyConserving())

# ## Building a `HydrostaticFreeSurfaceModel`
#
# We use `grid` and `coriolis` to build a simple `HydrostaticFreeSurfaceModel`,

using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity

# closure = HorizontallyCurvilinearAnisotropicDiffusivity(νh=0, κh=0)
# VectorInvariantEnergyConserving()
# HydrostaticSphericalCoriolis(Float64, scheme=VectorInvariantEnergyConserving())
# momentum_advection = VectorInvariant()
model = HydrostaticFreeSurfaceModel(grid = grid,
                                    momentum_advection =  nothing,
                                    tracers = (),
                                    buoyancy = nothing,
                                    coriolis = coriolis,
                                    closure = nothing,
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=90))

# ## The Bickley jet on a sphere
# λ ∈ [-180°, 180°]
# ϕ ∈ [-90°, 90°] # 0° is equator
R = model.grid.radius   # [m]
ω = 0.0 # 7.848e-6            # [s⁻¹]
K = 7.848e-6            # [s⁻¹]
n = 4                   # dimensionless
g = model.free_surface.gravitational_acceleration          # [m/s²]
Ω = coriolis.rotation_rate     # [s⁻¹] 2π/86400
ϵ = 0.0 # perturbation veloctiy # [m/s]

A(θ) = ω/2 * (2 * Ω + ω) * cos(θ)^2 + 1/4 * K^2 * cos(θ)^(2*n) * ((n+1) * cos(θ)^2 + (2 * n^2 - n - 2) - 2 * n^2 * sec(θ)^2 )
B(θ) = 2 * K * (Ω + ω) * ((n+1) * (n+2))^(-1) * cos(θ)^(n) * ( n^2 + 2*n + 2 - (n+1)^2 * cos(θ)^2) # why not  (n+1)^2 sin(θ)^2 + 1
C(θ)  = 1/4 * K^2 * cos(θ)^(2 * n) * ( (n+1) * cos(θ)^2 - (n+2))

# here: θ ∈ [-π/2, π/2] is latitude, ϕ ∈ [0, 2π) is longitude
u(θ, ϕ) =  R * ω * cos(θ) + R * K * cos(θ)^(n-1) * (n * sin(θ)^2 - cos(θ)^2) * cos(n*ϕ) 
v(θ, ϕ) = -n * K * R * cos(θ)^(n-1) * sin(θ) * sin(n*ϕ) 
h(θ, ϕ) =  0*h₀ + R^2/g * (  A(θ)  +  B(θ)  * cos(n * ϕ) + C(θ) * cos(2 * n * ϕ) ) 



# Total initial conditions
# previously: θ ∈ [-π/2, π/2] is latitude, ϕ ∈ [0, 2π) is longitude
# oceanoganigans: ϕ ∈ [-90, 90], λ ∈ [-180, 180], 
rescale¹(λ) = (λ + 180)/ 360 * 2π # λ to θ
rescale²(ϕ) = ϕ / 180 * π # θ to ϕ
# arguments were u(θ, ϕ), λ |-> ϕ, θ |-> ϕ
uᵢ(λ, ϕ, z) = u(rescale²(ϕ), rescale¹(λ))
vᵢ(λ, ϕ, z) = v(rescale²(ϕ), rescale¹(λ))
ηᵢ(λ, ϕ, z) = h(rescale²(ϕ), rescale¹(λ)) # (rescale¹(λ), rescale²(ϕ))

set!(model, u=uᵢ, v=vᵢ, η = ηᵢ) 

# Create Simulation

speed = (n * (3 + n ) * ω - 2*Ω) / ((1+n) * (2+n))
# angles per day speed / π * 180  * 86400
# abs(45 * π / 180 / speed / 86400) days for 45 degree rotation
numdays = abs(45 * π / 180 / speed / 86400)

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

wave_propagation_time_scale = h₀ * model.grid.Δλ / gravity_wave_speed
# numdays*86400
# 20wave_propagation_time_scale,
Δt =  0.5wave_propagation_time_scale
simulation = Simulation(model, Δt = Δt, stop_time = numdays*86400, progress = s -> @info "Time = $(prettytime(s.model.clock.time)) / $(prettytime(s.stop_time))")

output_fields = merge(model.velocities, (η=model.free_surface.η,))

using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(200.0wave_propagation_time_scale),
                                                      prefix = "rh_nonlinear",
                                                      force = true)
##
run!(simulation)
##
using JLD2, Printf, Oceananigans.Grids, GLMakie
using Oceananigans.Utils: hours

λ, ϕ, r = nodes(model.free_surface.η, reshape=true)

λ = λ .+ 180  # Convert to λ ∈ [0°, 360°]
ϕ = 90 .- ϕ   # Convert to ϕ ∈ [0°, 180°] (0° at north pole)
##
file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

iter = Node(0)
plot_title = @lift @sprintf("Rossby-Haurwitz Test: u, v, η @ time = %s", prettytime(file["timeseries/t/" * string($iter)]))
up = @lift file["timeseries/u/" * string($iter)][:, :, 1]
vp = @lift file["timeseries/v/" * string($iter)][:, :, 1]
ηp = @lift file["timeseries/η/" * string($iter)][:, :, 1]

up0 = file["timeseries/u/" * string(0)][:, :, 1]
vp0 = file["timeseries/v/" * string(0)][:, :, 1]
ηp0 = file["timeseries/η/" * string(0)][:, :, 1]

# Plot on the unit sphere to align with the spherical wireframe.
# Multiply by 1.01 so the η field is a bit above the wireframe.
x = @. 1.01 * cosd(λ) * sind(ϕ)
y = @. 1.01 * sind(λ) * sind(ϕ)
z = @. 1.01 * cosd(ϕ) * λ ./ λ

x = x[:, :, 1]
y = y[:, :, 1]
z = z[:, :, 1]

fig = Figure(resolution = (3156, 1074))

clims = [extrema(up0), extrema(vp0), extrema(ηp0)]

statenames = ["u", "v", "η"]
for (n, var) in enumerate([up, vp, ηp])
    ax = fig[3:7, 3n-2:3n] = LScene(fig) # make plot area wider
    wireframe!(ax, Sphere(Point3f0(0), 1f0), show_axis=false)
    surface!(ax, x, y, z, color=var, colormap=:balance, colorrange=clims[n])
    rotate_cam!(ax.scene, (π/2, π/6, 0))
    zoom!(ax.scene, (0, 0, 0), 5, false)
    fig[2, 2 + 3*(n-1)] = Label(fig, statenames[n], textsize = 50) # put names in center
end

supertitle = fig[0, :] = Label(fig, plot_title, textsize=50)
display(fig)
#=
for i in iterations
    iter[] = i
    display(fig)
end
=#
##
iterations = 1:360
record(fig, "RossbyHaurwitzIC.mp4", iterations, framerate=30) do i
    for n in 1:3
        rotate_cam!(fig.scene.children[n], (2π/360, 0, 0))
    end
end

##
record(fig, "RossbyHaurwitzEvolution.mp4", iterations, framerate=30) do i
    iter[] = i
end