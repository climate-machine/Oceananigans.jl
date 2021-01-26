using JULES, Oceananigans
using Printf
using Plots
using NLsolve

Nx = Ny = 1
Nz = 32
L = 10e3

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), halo=(2, 2, 2),
                            x=(0, L), y=(0, L), z=(0, L))
Δz = grid.Δz

pₛ = 100000
Tₐ = 293.15
g  = 9.80665

buoyancy = IdealGas()
Rᵈ, cₚ, γ = buoyancy.Rᵈ, buoyancy.cₚ, buoyancy.γ

####
#### Isothermal atmosphere
####

H = Rᵈ * Tₐ / g    # Scale height [m]
ρₛ = pₛ / (Rᵈ*Tₐ)  # Surface density [kg/m³]

p₀(x, y, z) = pₛ * exp(-z/H)
ρ₀(x, y, z) = ρₛ * exp(-z/H)

θ₀(x, y, z) = Tₐ * exp(z/H * Rᵈ/cₚ)
Θ₀(x, y, z) = ρ₀(x, y, z) * θ₀(x, y, z)

θ₀_prof = θ₀.(0, 0, grid.zC)
p₀_prof = p₀.(0, 0, grid.zC)
ρ₀_prof = ρ₀.(0, 0, grid.zC)

####
#### Objective function describing the ideal gas equation of state and hydrostatic balance
#### Set up a nonlinear system of equations F(x) = 0 and solve for pₖ and ρₖ.
####

# First Nz variables are pressure values pₖ, k = 1, 2, …, Nz
# Second Nz variables are density values ρₖ
function f!(F, x)
    @inbounds begin
        p₁, ρ₁, θ₁ = x[1], x[Nz+1], θ₀_prof[1]
        F[1] = p₁ - pₛ * (Rᵈ*ρ₁*θ₁ / pₛ)^γ
        F[Nz+1] = p₁ - pₛ + ρ₁*g*(Δz/2)

        for k in 2:Nz
            pₖ, pₖ₋₁, ρₖ, θₖ = x[k], x[k-1], x[Nz+k], θ₀_prof[k]
            F[k] = pₖ - pₛ * (Rᵈ*ρₖ*θₖ / pₛ)^γ
            F[Nz+k] = pₖ - pₖ₋₁ + ρₖ*g*Δz
        end
    end
end

guess = [p₀_prof; ρ₀_prof]
sol = nlsolve(f!, guess, show_trace=true)

####
#### Plot difference between initial condition and solution.
####

p∞_prof = sol.zero[1:Nz]
ρ∞_prof = sol.zero[Nz+1:2Nz]

p_plot = plot(p₀_prof, grid.zC, xlabel="pressure (Pa)", ylabel="z (m)", label="initial")
plot!(p_plot, p∞_prof, grid.zC, label="balanced")

ρ_plot = plot(ρ₀_prof, grid.zC, xlabel="rho (kg/m³)", ylabel="z (m)", label="initial")
plot!(ρ_plot, ρ∞_prof, grid.zC, label="balanced")

display(plot(p_plot, ρ_plot, show=true))

####
#### Set up compressible model
####

model = CompressibleModel(grid=grid, buoyancy=buoyancy, reference_pressure=pₛ,
                          thermodynamic_variable=ModifiedPotentialTemperature(),
                          tracers=(:Θᵐ,))

Θ∞_prof = ρ∞_prof .* θ₀_prof

set!(model.density, reshape(ρ∞_prof, (Nx, Ny, Nz)))
set!(model.tracers.Θᵐ, reshape(Θ∞_prof, (Nx, Ny, Nz)))

times = [model.clock.time]
ρw_ts = [model.momenta.W[1, 1, Int(Nz/2)]]

while model.clock.time < 50
    time_step!(model; Δt=0.5, Nt=10)

    @show model.clock.time
    Θ_prof = model.tracers.Θᵐ[1, 1, 1:Nz] .- Θ∞_prof
    W_prof = model.momenta.W[1, 1, 1:Nz+1]
    ρ_prof = model.density[1, 1, 1:Nz] .- ρ∞_prof

    Θ_plot = plot(Θ_prof, grid.zC, xlim=(-2.5, 2.5), xlabel="Theta_prime", ylabel="z (m)", label="")
    W_plot = plot(W_prof, grid.zF, xlim=(-1.2, 1.2), xlabel="rho*w", label="")
    ρ_plot = plot(ρ_prof, grid.zC, xlim=(-0.01, 0.01), xlabel="rho", label="")

    t_str = @sprintf("t = %d s", model.clock.time)
    display(plot(Θ_plot, W_plot, ρ_plot, title=["" "$t_str" ""], layout=(1, 3), show=true))

    push!(times, model.clock.time)
    push!(ρw_ts, model.momenta.W[1, 1, Int(Nz/2)])
end
