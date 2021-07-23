using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using Statistics

function geostrophic_adjustment_simulation(free_surface)

    grid = RegularRectilinearGrid(size = (64, 1, 1),
                                  x = (0, 1000kilometers), y = (0, 1), z = (-400meters, 0),
                                  topology = (Bounded, Periodic, Bounded))

    coriolis = FPlane(f=1e-4)

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        coriolis = coriolis,
                                        free_surface = free_surface)

    gaussian(x, L) = exp(-x^2 / 2L^2)
    
    U = 0.1 # geostrophic velocity
    L = grid.Lx / 40 # gaussian width
    x₀ = grid.Lx / 4 # gaussian center
    
    vᵍ(x, y, z) = - U * (x - x₀) / L * gaussian(x - x₀, L)
    
    g = model.free_surface.gravitational_acceleration
    
    η₀ = coriolis.f * U * L / g # geostrohpic free surface amplitude
    
    ηᵍ(x) = η₀ * gaussian(x - x₀, L)

    ηⁱ(x, y) = 2 * ηᵍ(x)

    set!(model, v=vᵍ, η=ηⁱ)
    gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
    wave_propagation_time_scale = model.grid.Δx / gravity_wave_speed
    simulation = Simulation(model, Δt=2wave_propagation_time_scale, stop_iteration=2)

    return simulation
end

function run_and_analyze(simulation)
    η = simulation.model.free_surface.η
    u, v, w = simulation.model.velocities

    ηx = ComputedField(∂x(η))
    compute!(ηx)

    u₀ = interior(u)[:, 1, 1]
    v₀ = interior(v)[:, 1, 1]
    η₀ = interior(η)[:, 1, 1]
    ηx₀ = interior(ηx)[:, 1, 1]
    
    run!(simulation)
    
    compute!(ηx)

    u₁ = interior(u)[:, 1, 1]
    v₁ = interior(v)[:, 1, 1]
    η₁ = interior(η)[:, 1, 1]
    ηx₁ = interior(ηx)[:, 1, 1]

    @show mean(η₀)
    @show mean(η₁)
    
    Δη = η₁ .- η₀

    return (; η₀, η₁, Δη, ηx₀, ηx₁, u₀, u₁, v₀, v₁)
end

fft_based_free_surface = ImplicitFreeSurface(solver_method=:FFTBased)
pcg_free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient)

free_surfaces = [fft_based_free_surface, pcg_free_surface]
simulations = [geostrophic_adjustment_simulation(free_surface) for free_surface in free_surfaces]
data = [run_and_analyze(sim) for sim in simulations]

fft_data = data[1]
pcg_data = data[2]

#pcg_p = plot([pcg_data.η₀ pcg_data.η₁ pcg_data.Δη], label=["η₀" "ηᵢ" "Δη"], linewidth=2)
#fft_p = plot([fft_data.η₀ fft_data.η₁ fft_data.Δη], label=["η₀" "ηᵢ" "Δη"], linewidth=2)

pcg_p_η = plot([pcg_data.η₀ pcg_data.η₁], label=["η₀" "ηᵢ"], linewidth=2)
fft_p_η = plot([fft_data.η₀ fft_data.η₁], label=["η₀" "ηᵢ"], linewidth=2)

pcg_p_ηx = plot([pcg_data.ηx₀ pcg_data.ηx₁], label=["ηx₀" "ηxᵢ"], linewidth=2)
fft_p_ηx = plot([fft_data.ηx₀ fft_data.ηx₁], label=["ηx₀" "ηxᵢ"], linewidth=2)

pcg_p_u = plot([pcg_data.u₀ pcg_data.u₁], label=["u₀" "uᵢ"], linewidth=2)
fft_p_u = plot([fft_data.u₀ fft_data.u₁], label=["u₀" "uᵢ"], linewidth=2)

p = plot(pcg_p_η, fft_p_η,
         pcg_p_u, fft_p_u,
         pcg_p_ηx, fft_p_ηx,
         layout=(3, 2), titles = ["PCG η" "FFT η" "PCG u" "FFT u" "PCG η_x" "FFT η_x"])

display(p)
