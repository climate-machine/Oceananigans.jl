using Printf
using Plots

using Oceananigans
using Oceananigans.Advection
using JULES

using JULES: IdealGas
using Oceananigans.Grids: Cell, xnodes

ENV["GKSwstype"] = "100"

@inline x′(x, t, L, U) = mod(x + L/2 - U * t, L) - L/2
@inline ϕ_Square(x, t; L, U, w=0.15) = -w <= x′(x, t, L, U) <= w ? 1.0 : 0.0

function every(n)
      0 < n <= 128 && return 1
    128 < n <= 256 && return 2
    256 < n <= 512 && return 4
    512 < n        && return 8
end

function periodic_advection_verification(N, L, T, U, CFL, advection, solution)
    ideal_gas = IdealGas(Float64, JULES.R⁰, 1; T₀=1, p₀=1, s₀=1, u₀=0)

    topo = (Periodic, Periodic, Periodic)
    domain = (x=(-L/2, L/2), y=(0, 1), z=(0, 1))
    grid = RegularCartesianGrid(topology=topo, size=(N, 1, 1), halo=(4, 4, 4); domain...)

    Δt = CFL * grid.Δx / abs(U)
    Nt = ceil(Int, T/Δt)

    model = CompressibleModel(
                      grid = grid,
                 advection = advection,
                     gases = (ρ=ideal_gas,),
    thermodynamic_variable = Entropy(),
                   closure = IsotropicDiffusivity(ν=0, κ=0),
                   gravity = 0.0,
               tracernames = (:ρ, :ρs, :ρc)
    )

    gas = model.gases.ρ
    R, cₚ, cᵥ = gas.R, gas.cₚ, gas.cᵥ
    u₀₀, T₀₀, ρ₀₀, s₀₀ = gas.u₀, gas.T₀, gas.ρ₀, gas.s₀
    
    ρ₀(x, y, z) = 1
    p₀(x, y, z) = 1
    T₀(x, y, z) = p₀(x, y, z) / (R*ρ₀(x, y, z))
    ρs₀(x, y, z) = ρ₀(x, y, z) * (s₀₀ + cᵥ * log(T₀(x, y, z)/T₀₀) - R * log(ρ₀(x, y, z)/ρ₀₀))

    set!(model.tracers.ρ, ρ₀)
    set!(model.momenta.ρu, U)
    set!(model.tracers.ρs, ρs₀)
    update_total_density!(model)

    initial_condition(x, y, z) = solution(x, 0)
    set!(model.tracers.ρc, initial_condition)
    set!(model.momenta.ρv, initial_condition)
    set!(model.momenta.ρw, initial_condition)

    anim = @animate for n in 1:Nt
        @info "Running periodic advection [N=$N, CFL=$CFL, $(typeof(advection)), U=$U]... iteration $n/$Nt"
        
        time_step!(model, Δt)

        x = xnodes(Cell, grid)
        analytic_solution = solution.(x, model.clock.time)
        ρc = interior(model.tracers.ρc)[:]
        ρv = interior(model.momenta.ρv)[:]
        ρw = interior(model.momenta.ρv)[:]

        title = @sprintf("N=%d, CFL=%.2f %s", N, CFL, typeof(advection))
        plot(x, analytic_solution, lw=2, label="analytic", title=title, xlims=(-L/2, L/2), ylims=(-0.2, 1.2), dpi=200)
        plot!(x, ρc, lw=2, ls=:solid, label="Oceananigans ρc")
        plot!(x, ρv, lw=2, ls=:dash,  label="Oceananigans ρv")
        plot!(x, ρw, lw=2, ls=:dot,   label="Oceananigans ρw")

    end every every(Nt)

    filename = @sprintf("periodic_advection_N%d_CFL%.2f_%s_U%+d.mp4", N, CFL, typeof(advection), U)
    mp4(anim, filename, fps=15)
end

T = 2
L = 1

Ns = [64]
Us = [+1, -1]
CFLs = [0.1]
advection_schemes = [CenteredSecondOrder(), CenteredFourthOrder(), UpwindBiasedThirdOrder(), UpwindBiasedFifthOrder(), WENO5()]

for N in Ns, CFL in CFLs, scheme in advection_schemes, U in Us
    @info "Running periodic advection [N=$N, CFL=$CFL, $(typeof(scheme)), U=$U]..."
    solution(x, t) = ϕ_Square(x, t, L=L, U=U)
    periodic_advection_verification(N, L, T, U, CFL, scheme, solution)
end
