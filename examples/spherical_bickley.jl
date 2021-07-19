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


grid = RegularLatitudeLongitudeGrid(size = (90*2, 40*2, 1), longitude = (-180, 180), latitude = (-80, 80), z = (-1, 0), radius = 1.2)
#  λ for latitude and ϕ for latitude is
using Oceananigans.Coriolis: VectorInvariantEnergyConserving, HydrostaticSphericalCoriolis

Ω = 0
coriolis = HydrostaticSphericalCoriolis(Float64, scheme=VectorInvariantEnergyConserving(), rotation_rate = Ω)

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
νh = κh = 1e-6
model = HydrostaticFreeSurfaceModel(grid = grid,
                                    momentum_advection = VectorInvariant(),
                                    buoyancy = nothing,
                                    tracers = :c,
                                    coriolis = coriolis,
                                    closure = HorizontallyCurvilinearAnisotropicDiffusivity(κh=κh, νh =  νh),
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=1e-4),
)

# ## The Bickley jet on a sphere
# θ ∈ [-π/2, π/2]
# ϕ ∈ [-π, π] # 0° is equator
ℓᵐ = 10
ℓ = 20

m = 2
θᵖ = π/2 * 0.05
ϵ = 0.3
vˢ = 5e-4
g = model.free_surface.gravitational_acceleration 
  
# u =   - r⁻¹∂ψ/∂θ
# v =  ( r cos(θ) )⁻¹ ∂ψ/∂ϕ

ψᵐ(θ, ϕ) = tanh(ℓᵐ * θ) 
ψᵖ(θ, ϕ) = exp(-ℓ * (θ - θᵖ)^2) * cos(θ) * cos(2 * (θ - θᵖ)) * sin(m * ϕ)

# here: θ ∈ [-π/2, π/2] is latitude, ϕ ∈ [0, 2π) is longitude
uᵐ(θ, ϕ) =  ℓᵐ * sech(ℓᵐ * θ)^2 
vᵐ(θ, ϕ) =  0.0
hᵐ(θ, ϕ) =  1.0 

u1(θ, ϕ) =  ℓ * 2 * (θ - θᵖ)* exp(-ℓ * (θ - θᵖ)^2) * cos(θ) * cos(2 * (θ - θᵖ)) * sin(m * ϕ)
u2(θ, ϕ) =  exp(-ℓ * (θ - θᵖ)^2) * sin(θ) * cos(2 * (θ - θᵖ)) * sin(m * ϕ)
u3(θ, ϕ) =  2*exp(-ℓ * (θ - θᵖ)^2) * cos(θ) * sin(2 * (θ - θᵖ)) * sin(m * ϕ)
uᵖ(θ, ϕ) =  u1(θ, ϕ) + u2(θ, ϕ) + u3(θ, ϕ)
vᵖ(θ, ϕ) =  m * exp(-ℓ * (θ - θᵖ)^2) * cos(2 * (θ - θᵖ)) * cos(m * ϕ)
hᵖ(θ, ϕ) =  0.0 

u(θ, ϕ) = vˢ * (uᵐ(θ, ϕ) + ϵ * uᵖ(θ, ϕ))
v(θ, ϕ) = vˢ * (vᵐ(θ, ϕ) + ϵ * vᵖ(θ, ϕ))
h(θ, ϕ) = hᵐ(θ, ϕ) + ϵ * hᵖ(θ, ϕ)
tracer(θ, ϕ) = tanh(ℓᵐ * θ)



# Total initial conditions
# previously: θ ∈ [-π/2, π/2] is latitude, ϕ ∈ [0, 2π) is longitude
# oceanoganigans: ϕ ∈ [-90, 90], λ ∈ [-180, 180], 
rescale¹(λ) = (λ + 180)/ 360 * 2π # λ to θ
rescale²(ϕ) = ϕ / 180 * π # θ to ϕ
# arguments were u(θ, ϕ), λ |-> ϕ, θ |-> ϕ
uᵢ(λ, ϕ, z) = u(rescale²(ϕ), rescale¹(λ))
vᵢ(λ, ϕ, z) = v(rescale²(ϕ), rescale¹(λ))
ηᵢ(λ, ϕ, z) = h(rescale²(ϕ), rescale¹(λ)) # (rescale¹(λ), rescale²(ϕ))
tracerᵢ(λ, ϕ, z) = tracer(rescale²(ϕ), rescale¹(λ)) # (rescale¹(λ), rescale²(ϕ))

set!(model, u=uᵢ, v=vᵢ, η = ηᵢ, c = tracerᵢ) 

# Create Simulation

gravity_wave_speed = sqrt(model.free_surface.gravitational_acceleration * grid.Lz) # hydrostatic (shallow water) gravity wave speed

# 20wave_propagation_time_scale,
Δt = grid.Lz * model.grid.Δλ / gravity_wave_speed * 0.001
simulation = Simulation(model, Δt = Δt, stop_time = 4*6000Δt, progress = s -> @info "Time = $(prettytime(s.model.clock.time)) / $(prettytime(s.stop_time))")

output_fields = merge(model.velocities, (η=model.free_surface.η,), model.tracers)

using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(100Δt),
                                                      prefix = "rh",
                                                      force = true)

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
plot_title = @lift @sprintf("Spherical-Bickley Test: u, v, η @ time = %s", prettytime(file["timeseries/t/" * string($iter)]))
up = @lift file["timeseries/u/" * string($iter)][:, :, 1]
vp = @lift file["timeseries/v/" * string($iter)][:, :, 1]
ηp = @lift file["timeseries/η/" * string($iter)][:, :, 1]
cp = @lift file["timeseries/c/" * string($iter)][:, :, 1]

up0 = file["timeseries/u/" * string(0)][:, :, 1]
vp0 = file["timeseries/v/" * string(0)][:, :, 1]
ηp0 = file["timeseries/η/" * string(0)][:, :, 1]
cp0 = file["timeseries/c/" * string(0)][:, :, 1]

# Plot on the unit sphere to align with the spherical wireframe.
# Multiply by 1.01 so the η field is a bit above the wireframe.
x = @. 1.01 * cosd(λ) * sind(ϕ)
y = @. 1.01 * sind(λ) * sind(ϕ)
z = @. 1.01 * cosd(ϕ) * λ ./ λ

x = x[:, :, 1]
y = y[:, :, 1]
z = z[:, :, 1]

fig = Figure(resolution = (3156, 1074))

clims = [(-2e-3, 2e-3), (-2e-3, 2e-3), (0.94,1.01), (-1.0, 1.0)]

statenames = ["u", "v", "η", "c"]
for (n, var) in enumerate([up, vp, ηp, cp])
    ax = fig[3:7, 3n-2:3n] = LScene(fig) # make plot area wider
    wireframe!(ax, Sphere(Point3f0(0), 1f0), show_axis=false)
    surface!(ax, x, y, z, color=var, colormap=:balance, colorrange=clims[n])
    rotate_cam!(ax.scene, (2π/3, π/6, 0))
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
record(fig, "SphericalBickley.mp4", iterations, framerate=30) do i
    for n in 1:3
        rotate_cam!(fig.scene.children[n], (2π/360, 0, 0))
    end
end

##
record(fig, "SphericalBickley.mp4", iterations, framerate=30) do i
    iter[] = i
    
    for n in 1:4
        rotate_cam!(fig.scene.children[n], (2π/360, 0, 0))
    end

end