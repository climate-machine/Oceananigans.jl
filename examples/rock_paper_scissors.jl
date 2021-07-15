# # Rock paper scissors ecological game

using Oceananigans

# # Grid and domain

using Oceananigans.Utils: meters

Nx = Ny = 64
Nz = 32

Lx = Ly = 100meters
Lz = 50meters

topology = (Periodic, Periodic, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(Nx, Ny, Nz),
                            x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

# # Idealized diurnal cycle

using Oceananigans.Utils: hours

N² = 1e-5

Qb(x, y, t, p) = p.Q₀ + p.QΔ * cos(2π * t / p.τ)

b_top_bc = FluxBoundaryCondition(Qb, parameters=(Q₀=0, QΔ=5e-8, τ=24hours))
b_bot_bc = GradientBoundaryCondition(N²)

b_bcs  = TracerBoundaryConditions(grid, top=b_top_bc, bottom=b_bot_bc)

# # Sponge layer

# # Lagrangian microbes

using StructArrays, NearestNeighbors

@enum Species Rock Paper Scissors
display(Species)

struct Microbe{T,S}
          x :: T
          y :: T
          z :: T
    species :: S
end

import Base: convert
convert(::Type{Float32}, s::Species) = Float32(Int(s))

n_microbes = 10000

x₀ = Lx * rand(n_microbes)
y₀ = Ly * rand(n_microbes)
z₀ = -Lz/10 * rand(n_microbes)
s₀ = rand(instances(Species), n_microbes)

microbes = StructArray{Microbe}((x₀, y₀, z₀, s₀))

particles = LagrangianParticles(microbes, 0.5)

function rock_paper_scissors!(microbes, i, j)
    sᵢ, sⱼ = microbes[i].species, microbes[j].species

    if sᵢ == Rock && sⱼ == Paper
        microbes.species[i] = Paper
    elseif sᵢ == Rock && sⱼ == Scissors
        microbes.species[j] = Rock
    elseif sᵢ == Paper && sⱼ == Rock
        microbes.species[j] = Paper
    elseif sᵢ == Paper && sⱼ == Scissors
        microbes.species[i] = Scissors
    elseif sᵢ == Scissors && sⱼ == Rock
        microbes.species[i] = Rock
    elseif sᵢ == Scissors && sⱼ == Paper
        microbes.species[j] = Scissors
    end

    return nothing
end

function microbe_interactions!(microbes; r=1)
    positions = cat(microbes.x', microbes.y', microbes.z', dims=1)
    tree = KDTree(positions)

    pairs = Set{Tuple{Int,Int}}()
    for (m, microbe) in enumerate(microbes)
        p = [microbe.x, microbe.y, microbe.z]
        nearby_microbes_inds = inrange(tree, p, r)

        for n in nearby_microbes_inds
            push!(pairs, minmax(m, n))
        end
    end

    @info "Playing out $(length(pairs)) microbe interactions..."

    for pair in pairs
        rock_paper_scissors!(microbes, pair...)
    end

    return nothing
end

# # Model setup

using Oceananigans.Advection

model = IncompressibleModel(
           architecture = CPU(),
                   grid = grid,
            timestepper = :RungeKutta3,
              advection = UpwindBiasedFifthOrder(),
                tracers = :b,
               buoyancy = BuoyancyTracer(),
               coriolis = FPlane(f=-1e-4),
                closure = AnisotropicMinimumDissipation(),
    boundary_conditions = (b=b_bcs,),
              particles = particles
)

# # Setting initial conditions

b₀(x, y, z) = N²*z + 1e-10 * randn()
set!(model, b=b₀)

# # Simulation setup

using Printf
using Oceananigans.Utils: prettytime, second, seconds, hours, days

function print_progress(simulation)
    model = simulation.model

    microbe_interactions!(microbes, r=0.5)

    @info @sprintf("iteration: %04d, time: %s, Δt: %s",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(wizard.Δt))

    return nothing
end

wizard = TimeStepWizard(cfl=0.7, Δt=1second, min_Δt=0.2seconds, max_Δt=90seconds)

simulation = Simulation(model, Δt=wizard, iteration_interval=1, stop_time=2days, progress=print_progress)

# # Output writing

using Oceananigans: fields

using Oceananigans.OutputWriters

using Oceananigans.Utils: minutes

simulation.output_writers[:fields] = NetCDFOutputWriter(model, fields(model), filepath="idealized_diurnal_cycle.nc",
                                                        schedule=TimeInterval(2minutes))

simulation.output_writers[:particles] = NetCDFOutputWriter(model, model.particles, filepath="lagrangian_microbes.nc",
                                                           schedule=TimeInterval(2minutes))

run!(simulation)

# # Visualizing the solution

using Plots, GeoData, NCDatasets

using GeoData: GeoXDim, GeoYDim, GeoZDim

@dim xC GeoXDim "x"
@dim xF GeoXDim "x"
@dim yC GeoYDim "y"
@dim yF GeoYDim "y"
@dim zC GeoZDim "z"
@dim zF GeoZDim "z"

ds = NCDstack("idealized_diurnal_cycle.nc")

w, b = ds[:w], ds[:b]
times = dims(b)[4]
Nt = length(times)

anim = @animate for n in 1:Nt
    @info "Plotting idealized diurnal cycle frame $n/$Nt..."

    w_plot = plot(w[Ti=n, xC=32], color=:balance, clims=(-0.02, 0.02), aspect_ratio=:auto,
                  title="Idealized diurnal cycle: $(prettytime(times[n]))")

    b_plot = plot(b[Ti=n, xC=32], color=:thermal, clims=(-5e-4, 4e-4), aspect_ratio=:auto, title="")

    plot(w_plot, b_plot, layout=(2, 1), size=(1600, 900))
end

mp4(anim, "idealized_diurnal_cycle.mp4", fps=15)

# # Visualizing the particle trajectories

ds_particles = NCDstack("lagrangian_microbes.nc")
x, y, z = ds_particles[:x], ds_particles[:y], ds_particles[:z]
species = ds_particles[:species] .|> Int .|> Species

function particle_color(s::Species)
    s == Rock     && return "red"
    s == Paper    && return "green"
    s == Scissors && return "blue"
end

P = 300 # particles to animate
T = 20 # particle tail length

anim = @animate for n in 1:Nt
    @info "Plotting particles frame $n/$Nt..."

    xₙ = x[particleid=1:P, Ti=n]
    yₙ = y[particleid=1:P, Ti=n]
    zₙ = z[particleid=1:P, Ti=n]
    sₙ = species[particleid=1:P, Ti=n] .|> Int .|> Species

    s_plot = scatter(xₙ, yₙ, zₙ, color=particle_color.(sₙ), label="",
                     xlim=(0, 100), ylim=(0, 100), zlim=(-50, 0),
                     title="Lagrangian microbe locations: $(prettytime(times[n]))")

    for p in 1:P
        n == 1 && break

        tail_inds = max(1, n-T):n

        x_tail = x[particleid=p, Ti=tail_inds]
        y_tail = y[particleid=p, Ti=tail_inds]
        z_tail = z[particleid=p, Ti=tail_inds]

        Δx = -(extrema(x_tail)...) |> abs
        Δy = -(extrema(y_tail)...) |> abs
        Δz = -(extrema(z_tail)...) |> abs

        (Δx > 10 || Δy > 10 || Δz > 10) && continue

        tail_color = species[particleid=p, Ti=tail_inds].data .|> particle_color .|> color
        tail_color = RGBA.(tail_color, range(0, 1, length=length(tail_color)))

        plot!(s_plot, x_tail, y_tail, z_tail, color=tail_color, linewidth=2, label="")
    end

    h_plot = histogram(z[Ti=n], linewidth=0, orientation=:horizontal, normalize=true, bins=range(-50, 0, length=25),
                       xlabel="p(z)", ylabel="z", label="", title="", xlims=(0, 0.2), ylims=(-50, 0))

    plot(s_plot, h_plot, size=(1600, 900), layout=Plots.grid(1, 2, widths=[0.75, 0.25]))
end

mp4(anim, "particles.mp4", fps=15)

# # Analyzing Lagrangian microbe interactions

n_rocks = [sum(species[Ti=n] .== Rock) for n in 1:Nt]
n_papers = [sum(species[Ti=n] .== Paper) for n in 1:Nt]
n_scissors = [sum(species[Ti=n] .== Scissors) for n in 1:Nt]

p = plot(times[:] / hours, n_rocks, color=:red, label="Rocks",
         xlim=(0, 48), xticks=0:6:24, xlabel="Time (hours)", ylabel="Species count",
         legend=:outertopright, dpi=200)

plot!(p, times[:] / hours, n_papers, color=:green, label="Papers")
plot!(p, times[:] / hours, n_scissors, color=:blue, label="Scissors")

savefig(p, "species_count.png")
