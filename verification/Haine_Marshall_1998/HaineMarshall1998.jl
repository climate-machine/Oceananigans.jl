using Printf
using Oceananigans
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.OutputWriters
using Oceananigans.Diagnostics
using Oceananigans.Utils

arch = GPU()
FT   = Float64

Lx = 50kilometer
Ly = 30kilometer
Lz = 2kilometer

Δx = Δy = 250meter
Δz = 40meter

Nx = Int(Lx / Δx)
Ny = Int(Ly / Δy)
Nz = Int(Lz / Δz)

topology = (Periodic, Bounded, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

f = 1e-4
coriolis = FPlane(FT, f=f)

α = 2e-4  # Thermal expansion coefficient [K⁻¹]
eos = LinearEquationOfState(FT, α=α, β=0)
buoyancy = BuoyancyTracer()
# buoyancy = SeawaterBuoyancy(FT, equation_of_state=eos, constant_salinity=true)

κh = νh = 5.0   # Horizontal diffusivity and viscosity [m²/s]
κv = νv = 0.02  # Vertical diffusivity and viscosity [m²/s]
closure = ConstantAnisotropicDiffusivity(FT, νh=νh, νv=νv, κh=κh, κv=κv)

bc_params = (
    Ly = Ly,
    B½ = 1.96e-7,    # Buoyancy flux at midchannel [m²/s³]
    Lᶠ = 10kilometer # Characteristic length scale of the forcing [m]
)
buoyancy_flux(x, y, t, p) = p.B½ * (tanh(2 * (y - p.Ly/2) / p.Lᶠ) + 1)  # Surface buoyancy flux [m²/s³]
buoyancy_flux_bf = BoundaryFunction{:z, Cell, Cell}(buoyancy_flux, bc_params)
top_b_bc = FluxBoundaryCondition(buoyancy_flux_bf)
b_bcs = TracerBoundaryConditions(grid, top=top_b_bc)

top_C_bc = ValueBoundaryCondition(1.0)
C_bcs = TracerBoundaryConditions(grid, top=top_C_bc)

model = IncompressibleModel(
           architecture = arch,
             float_type = FT,
                   grid = grid,
               coriolis = coriolis,
               buoyancy = buoyancy,
                closure = closure,
                tracers = (:b,),
    boundary_conditions = (b=b_bcs,)
)

Tₛ  = 12.0     # Surface temperature [°C]
Nₜₕ = 8.37e-4  # Uniform vertical stratification [s⁻¹]
ε(σ) = σ * randn()
B₀(x, y, z) = Nₜₕ^2 * z + ε(1e-8)

set!(model, b=B₀)
# set!(model, b=B₀, C=0.0)

fields = Dict(
    "u" => model.velocities.u,
    "v" => model.velocities.v,
    "w" => model.velocities.w,
    "b" => model.tracers.b
)

surface_output_writer =
    NetCDFOutputWriter(model, fields, filename="HaineMarshall1998_surface.nc",
			           interval=1hour, zC=Nz, zF=Nz)

middepth_output_writer =
    NetCDFOutputWriter(model, fields, filename="HaineMarshall1998_middepth.nc",
                       interval=1hour, zC=Int(Nz/2), zF=Int(Nz/2))

zonal_output_writer =
    NetCDFOutputWriter(model, fields, filename="HaineMarshall1998_zonal.nc",
                       interval=1hour, yC=Int(Ny/2), yF=Int(Ny/2))

meridional_output_writer =
    NetCDFOutputWriter(model, fields, filename="HaineMarshall1998_meridional.nc",
                       interval=1hour, xC=Int(Nx/2), xF=Int(Nx/2))


Δt_wizard = TimeStepWizard(cfl=0.3, Δt=10.0, max_change=1.2, max_Δt=600.0)
cfl = AdvectiveCFL(Δt_wizard)


# Take Ni "intermediate" time steps at a time before printing a progress
# statement and updating the time step.
Ni = 10

function print_progress(simulation)
    model = simulation.model
    i, t = model.clock.iteration, model.clock.time

    progress = 100 * (model.clock.time / end_time)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)

    @printf("[%05.2f%%] i: %d, t: %.2e days, umax: (%6.3e, %6.3e, %6.3e) m/s, CFL: %6.4e, next Δt: %.1e s\n",
    	    progress, i, t / day, umax, vmax, wmax, cfl(model), Δt_wizard.Δt)
end

end_time = 10day
simulation = Simulation(model, Δt=Δt_wizard, stop_time=end_time, progress=print_progress, progress_frequency=Ni)

simulation.output_writers[:surface] = surface_output_writer
simulation.output_writers[:middepth] = middepth_output_writer
simulation.output_writers[:zonal] = zonal_output_writer
simulation.output_writers[:meridional] = meridional_output_writer

run!(simulation)

