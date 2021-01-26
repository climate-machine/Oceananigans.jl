using JULES, Oceananigans
using Plots
using Printf

Nx = Ny = 1
Nz = 32
L = 10e3

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), halo=(2, 2, 2),
                            x=(0, L), y=(0, L), z=(0, L))
#####
##### Dry adiabatic atmosphere
#####

gas = IdealGas()
Rᵈ, cₚ, cᵥ = gas.Rᵈ, gas.cₚ, gas.cᵥ

g  = 9.80665
pₛ = 100000
θₛ = 300
πₛ = 1

θ(x, y, z) = θₛ
π(x, y, z) = πₛ - g*z / (cₚ*θₛ)
p(x, y, z) = pₛ * π(x, y, z)^(cₚ/Rᵈ)
ρ(x, y, z) = pₛ / (Rᵈ * θ(x, y, z)) * π(x, y, z)^(cᵥ/Rᵈ)
Θ(x, y, z) = ρ(x, y, z) * θ(x, y, z)

#####
##### Create model
#####

model = CompressibleModel(grid=grid, buoyancy=gas, reference_pressure=pₛ,
                          thermodynamic_variable=ModifiedPotentialTemperature(),
                          tracers=(:Θᵐ,))

#####
##### Set initial conditions
#####

set!(model.density, ρ)
set!(model.tracers.Θᵐ, θ)

# Initial profiles
Θ₀_prof = model.tracers.Θᵐ[1, 1, 1:Nz]
ρ₀_prof = model.density[1, 1, 1:Nz]

# Arrays we will use to store a time series of ρw(z = L/2).
times = [model.clock.time]
ρw_ts = [model.momenta.ρw[1, 1, Int(Nz/2)]]

#####
##### Time step and keep plotting vertical profiles of ρθ′, ρw, and ρ′.
#####

# @animate for i=1:100
while model.clock.time < 500
    time_step!(model; Δt=0.5, Nt=10)

    @show model.clock.time
    Θ_prof = model.tracers.Θᵐ[1, 1, 1:Nz] .- Θ₀_prof
    W_prof = model.momenta.ρw[1, 1, 1:Nz+1]
    ρ_prof = model.density[1, 1, 1:Nz] .- ρ₀_prof

    Θ_plot = plot(Θ_prof, grid.zC, xlim=(-2.5, 2.5), xlabel="Theta_prime", ylabel="z (m)", label="")
    W_plot = plot(W_prof, grid.zF, xlim=(-1.2, 1.2), xlabel="rho*w", label="")
    ρ_plot = plot(ρ_prof, grid.zC, xlim=(-0.01, 0.01), xlabel="rho", label="")

    t_str = @sprintf("t = %d s", model.clock.time)
    display(plot(Θ_plot, W_plot, ρ_plot, title=["" "$t_str" ""], layout=(1, 3), show=true))

    push!(times, model.clock.time)
    push!(ρw_ts, model.momenta.ρw[1, 1, Int(Nz/2)])
end

#####
##### Plot the initial profile and the hydrostatically balanced profile.
#####

ρ∞_prof = model.density[1, 1, 1:Nz]

θ₀_prof = Θ₀_prof ./ ρ₀_prof
θ∞_prof = model.tracers.Θᵐ[1, 1, 1:Nz] ./ ρ∞_prof

θ_plot = plot(θ₀_prof, grid.zC, xlabel="theta (K)", ylabel="z (m)", label="initial", legend=:topleft)
plot!(θ_plot, θ∞_prof, grid.zC, label="balanced")

ρ_plot = plot(ρ₀_prof, grid.zC, xlabel="rho (kg/m³)", ylabel="z (m)", label="initial")
plot!(ρ_plot, ρ∞_prof, grid.zC, label="balanced")

display(plot(θ_plot, ρ_plot, show=true))


#####
##### Plot timeseries of maximum ρw
#####

plot(times, ρw_ts, linewidth=2, xlabel="time (s)", ylabel="rho*w(z=$(Int(L/2)) m)", label="", show=true)

