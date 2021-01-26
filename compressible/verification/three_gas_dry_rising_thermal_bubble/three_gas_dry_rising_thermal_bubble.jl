using Logging
using Printf
using Statistics
using NCDatasets
using CUDA

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.OutputWriters
using Oceananigans.Utils
using JULES

using Oceananigans.Fields: cpudata

Logging.global_logger(OceananigansLogger())

const km = kilometers
const hPa = 100.0

function simulate_three_gas_dry_rising_thermal_bubble(; architecture=CPU(), thermodynamic_variable, end_time=1000.0)
    tvar = thermodynamic_variable

    Lx = 20km
    Lz = 10km
    Δ  = 200meters

    Nx = Int(Lx/Δ)
    Ny = 1
    Nz = Int(Lz/Δ)

    topo = (Periodic, Periodic, Bounded)
    domain = (x=(-Lx/2, Lx/2), y=(-Lx/2, Lx/2), z=(0, Lz))
    grid = RegularCartesianGrid(topology=topo, size=(Nx, Ny, Nz), halo=(3, 3, 3); domain...)

    model = CompressibleModel(
                  architecture = architecture,
                          grid = grid,
                         gases = DryEarth3(),
        thermodynamic_variable = tvar,
                       closure = IsotropicDiffusivity(ν=75.0, κ=75.0)
    )

    gas = model.gases.ρ₁
    R, cₚ, cᵥ = gas.R, gas.cₚ, gas.cᵥ
    g  = model.gravity
    pₛ = 1000hPa
    Tₛ = 300.0

    # Define initial mixing ratios
    q₁(z) = exp(-(4z/Lz)^2)
    q₃(z) = exp(-(4*(z - Lz)/Lz)^2)
    q₂(z) = 1 - q₁(z) - q₃(z)

    # Define an approximately hydrostatic background state
    θ₀(x, y, z) = Tₛ
    p₀(x, y, z) = pₛ * (1 - g*z / (cₚ*Tₛ))^(cₚ/R)
    T₀(x, y, z) = Tₛ * (p₀(x, y, z)/pₛ)^(R/cₚ)
    ρ₀(x, y, z) = p₀(x, y, z) / (R*T₀(x, y, z))

    ρ₁₀(x, y, z) = q₁(z) * ρ₀(x, y, z)
    ρ₂₀(x, y, z) = q₂(z) * ρ₀(x, y, z)
    ρ₃₀(x, y, z) = q₃(z) * ρ₀(x, y, z)

    uᵣ, Tᵣ, ρᵣ, sᵣ = gas.u₀, gas.T₀, gas.ρ₀, gas.s₀  # Reference values
    ρe₀(x, y, z) = sum(ρ₀(x, y, z) * (uᵣ + cᵥ * (T₀(x, y, z) - Tᵣ) + g*z)
                       for ρ₀ in (ρ₁₀, ρ₂₀, ρ₃₀))

    function ρs₀(x, y, z)
       ρs = 0.0
       T = T₀(x, y, z)
       for ρ in (ρ₁₀(x, y, z), ρ₂₀(x, y, z), ρ₃₀(x, y, z))
           ρs += ρ > 0 ?  ρ * (sᵣ + cᵥ*log(T/Tᵣ) - R*log(ρ/ρᵣ)) : 0.0
       end
       return ρs
    end

    # Define the initial density perturbation
    θᶜ′ = 2.0
    xᶜ, zᶜ = 0km, 2km
    xʳ, zʳ = 2km, 2km

    L(x, y, z) = sqrt(((x - xᶜ)/xʳ)^2 + ((z - zᶜ)/zʳ)^2)
    θ′(x, y, z) = (L(x, y, z) <= 1) * θᶜ′ * cos(π/2 * L(x, y, z))^2
    ρ′(x, y, z) = -ρ₀(x, y, z) * θ′(x, y, z) / θ₀(x, y, z)

    # Define initial state
    ρᵢ(x, y, z) = ρ₀(x, y, z) + ρ′(x, y, z)
    pᵢ(x, y, z) = p₀(x, y, z)
    Tᵢ(x, y, z) = pᵢ(x, y, z) / (R * ρᵢ(x, y, z))

    ρ₁ᵢ(x, y, z) = q₁(z) * ρᵢ(x, y, z)
    ρ₂ᵢ(x, y, z) = q₂(z) * ρᵢ(x, y, z)
    ρ₃ᵢ(x, y, z) = q₃(z) * ρᵢ(x, y, z)

    ρeᵢ(x, y, z) = sum(ρᵢ(x, y, z) * (uᵣ + cᵥ * (Tᵢ(x, y, z) - Tᵣ) + g*z)
                       for ρᵢ in (ρ₁ᵢ, ρ₂ᵢ, ρ₃ᵢ))

    function ρsᵢ(x, y, z)
        ρs = 0.0
        T = Tᵢ(x, y, z)
        for ρ in (ρ₁ᵢ(x, y, z), ρ₂ᵢ(x, y, z), ρ₃ᵢ(x, y, z))
            ρs += ρ > 0 ?  ρ * (sᵣ + cᵥ*log(T/Tᵣ) - R*log(ρ/ρᵣ)) : 0.0
        end
        return ρs
    end

    # Set initial state (which includes the thermal perturbation)
    set!(model.tracers.ρ₁, ρ₁ᵢ)
    set!(model.tracers.ρ₂, ρ₂ᵢ)
    set!(model.tracers.ρ₃, ρ₃ᵢ)
    tvar isa Energy  && set!(model.tracers.ρe, ρeᵢ)
    tvar isa Entropy && set!(model.tracers.ρs, ρsᵢ)
    update_total_density!(model)

    simulation = Simulation(model, Δt=0.1, stop_time=end_time, iteration_interval=50,
                            progress=print_progress, parameters=(ρᵢ, ρeᵢ, ρsᵢ))

    fields = Dict(
        "ρ"  => model.total_density,
        "ρu" => model.momenta.ρu,
        "ρw" => model.momenta.ρw,
        "ρ₁" => model.tracers.ρ₁,
        "ρ₂" => model.tracers.ρ₂,
        "ρ₃" => model.tracers.ρ₃
    )

    tvar isa Energy  && push!(fields, "ρe" => model.tracers.ρe)
    tvar isa Entropy && push!(fields, "ρs" => model.tracers.ρs)
    
    simulation.output_writers[:fields] =
        NetCDFOutputWriter(model, fields, filepath="three_gas_dry_rising_thermal_bubble_$(typeof(tvar)).nc",
                           time_interval=10seconds)

    # Save base state to NetCDF.
    ds = simulation.output_writers[:fields].dataset
    ds_ρ = defVar(ds, "ρ₀", Float32, ("xC", "yC", "zC"))
    ds_ρ₁ = defVar(ds, "ρ₁₀", Float32, ("xC", "yC", "zC"))
    ds_ρ₂ = defVar(ds, "ρ₂₀", Float32, ("xC", "yC", "zC"))
    ds_ρ₃ = defVar(ds, "ρ₃₀", Float32, ("xC", "yC", "zC"))
    ds_ρe = defVar(ds, "ρe₀", Float32, ("xC", "yC", "zC"))

    x, y, z = nodes((Cell, Cell, Cell), grid, reshape=true)
    ds_ρ[:, :, :] = ρ₀.(x, y, z)
    ds_ρ₁[:, :, :] = ρ₁₀.(x, y, z)
    ds_ρ₂[:, :, :] = ρ₂₀.(x, y, z)
    ds_ρ₃[:, :, :] = ρ₃₀.(x, y, z)
    ds_ρe[:, :, :] = ρe₀.(x, y, z)

    run!(simulation)

    return simulation
end

function print_progress(simulation)
    model, Δt = simulation.model, simulation.Δt
    tvar = model.thermodynamic_variable
    ρᵢ, ρeᵢ, ρsᵢ = simulation.parameters

    zC = znodes(Cell, model.grid)
    ρ̄ᵢ = mean(ρᵢ.(0, 0, zC))
    ρ̄ = mean(cpudata(model.total_density))

    progress = 100 * model.clock.time / simulation.stop_time
    message = @sprintf("[%05.2f%%] iteration = %d, time = %s, CFL = %.4e, acoustic CFL = %.4e, ρ̄ = %.4e (relΔ = %.4e)",
                       progress, model.clock.iteration, prettytime(model.clock.time), cfl(model, Δt),
                       acoustic_cfl(model, Δt), ρ̄, (ρ̄ - ρ̄ᵢ) / ρ̄)

    if tvar isa Energy
        ρ̄ēᵢ = mean(ρeᵢ.(0, 0, zC))
        ρ̄ē = mean(cpudata(model.tracers.ρe))
        message *= @sprintf(", ρ̄ē = %.4e (relΔ = %.4e)", ρ̄ē, (ρ̄ē - ρ̄ēᵢ) / ρ̄ē)
    elseif tvar isa Entropy
        ρ̄s̄ᵢ = mean(ρsᵢ.(0, 0, zC))
        ρ̄s̄ = mean(cpudata(model.tracers.ρs))
        message *= @sprintf(", ρ̄s̄ = %.4e (relΔ = %.4e)", ρ̄s̄, (ρ̄s̄ - ρ̄s̄ᵢ) / ρ̄s̄)
    end

    @info message

    ∂tρ₁ = maximum(model.time_stepper.slow_source_terms.tracers.ρ₁.data.parent)
    ∂tρ₂ = maximum(model.time_stepper.slow_source_terms.tracers.ρ₂.data.parent)
    ∂tρ₃ = maximum(model.time_stepper.slow_source_terms.tracers.ρ₃.data.parent)

    @info @sprintf("[%05.2f%%] Maximum mass tendencies from diffusion: ∂tρ₁ = %.4e, ∂tρ₂ = %.4e, ∂tρ₃ = %.4e",
                   progress, ∂tρ₁, ∂tρ₂, ∂tρ₃)

    return nothing
end
