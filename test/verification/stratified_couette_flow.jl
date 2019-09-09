using Statistics, Printf

using Oceananigans
using Oceananigans.TurbulenceClosures

""" Friction velocity. See equation (16) of Vreugdenhil & Taylor (2018). """
function uτ(model)
    Nz, Hz, Δz = model.grid.Nz, model.grid.Hz, model.grid.Δz
    ν = model.closure.ν

    Up = HorizontalAverage(model, model.velocities.u; frequency=Inf)
    U = Up(model)[1+Hz:end-Hz]  # Exclude average of halo region.

    # Use a finite difference to calculate dU/dz at the top and bottom walls.
    # The distance between the center of the cell adjacent to the wall and the
    # wall itself is Δz/2.
    uτ²⁺ = ν * abs(U[1] - Uw)    / (Δz/2)  # Top wall    where u = +Uw
    uτ²⁻ = ν * abs(-Uw  - U[Nz]) / (Δz/2)  # Bottom wall where u = -Uw

    uτ⁺, uτ⁻ = √uτ²⁺, √uτ²⁻

    return uτ⁺, uτ⁻
end

""" Heat flux at the wall. See equation (16) of Vreugdenhil & Taylor (2018). """
function q_wall(model)
    Nz, Hz, Δz = model.grid.Nz, model.grid.Hz, model.grid.Δz
    κ = model.closure.κ

    Tp = HorizontalAverage(model, model.tracers.T; frequency=Inf)
    Θ = Tp(model)[1+Hz:end-Hz]  # Exclude average of halo region.

    q_wall⁺ = κ * abs(Θ[1] - Θw)   / (Δz/2)  # Top wall    where Θ = +Θw
    q_wall⁻ = κ * abs(-Θw - Θ[Nz]) / (Δz/2)  # Bottom wall where Θ = -Θw

    return q_wall⁺, q_wall⁻
end

""" Friction temperature. See equation (16) of Vreugdenhil & Taylor (2018). """
function θτ(model)
    uτ⁺, uτ⁻ = uτ(model)
    q_wall⁺, q_wall⁻ = q_wall(model)

    return qw⁺ / uτ⁺, qw⁻ / uτ⁻
end

""" Obukov length scale (assuming a linear equation of state). See equation (17) of Vreugdenhil & Taylor (2018). """
function L_Obukov(model)
    kₘ = 0.4  # Von Kármán constant
    uτ²⁺, uτ²⁻ = uτ²(model)
    qw⁺, qw⁻ = qw(model)
    uτ⁺, uτ⁻ = √uτ²⁺, √uτ²⁻

    uτ⁺^3 / (kₘ * qw⁺), uτ⁻^3 / (kₘ * qw⁻)
end

""" Near wall viscous length scale. See line following equation (18) of Vreugdenhil & Taylor (2018). """
function δᵥ(model)
    ν = model.closure.ν
    uτ²⁺, uτ²⁻ = uτ²(model)
    ν / √uτ²⁺, ν / √uτ²⁻
end

"""
    Ratio of length scales that define when the stratified plane Couette flow is turbulent.
    See equation (18) of Vreugdenhil & Taylor (2018).
"""
function L⁺(model)
    δᵥ⁺, δᵥ⁻ = δᵥ(model)
    L_O⁺, L_O⁻ = L_Obukov(model)
    L_O⁺ / δᵥ⁺, L_O⁻ / δᵥ⁻
end

""" Friction Reynolds number. See equation (20) of Vreugdenhil & Taylor (2018). """
function Reτ(model)
    ν = model.closure.ν
    h = model.grid.Lz / 2
    uτ⁺, uτ⁻ = uτ(model)
    
    return h * uτ⁺ / ν, h * uτ⁻ / ν
end

""" Friction Nusselt number. See equation (20) of Vreugdenhil & Taylor (2018). """
function Nu(model)
    κ = model.closure.κ
    h = model.grid.Lz / 2
    q_wall⁺, q_wall⁻ = q_wall(model)

    return (q_wall⁺ * h)/(κ * Θw), (q_wall⁻ * h)/(κ * Θw)
end

# Non-dimensional parameters chosen to reproduce run 5 from Table 1 of Vreugdenhil & Taylor (2018).
Pr = 0.7   # Prandtl number
Re = 4250  # Reynolds number
Ri = 0.04  # Richardson number
 h = 1.0   # Height
Uw = 1.0   # Wall velocity

# Computed parameters.
 ν = Uw * h / Re    # From Re = Uw h / ν
Θw = Ri * Uw^2 / h  # From Ri = L Θw / Uw²
 κ = ν / Pr         # From Pr = ν / κ

# Impose boundary conditions.
Tbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Value,  Θw),
                                bottom = BoundaryCondition(Value, -Θw))

ubcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Value,  Uw),
                                bottom = BoundaryCondition(Value, -Uw))

vbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Value, 0),
                                bottom = BoundaryCondition(Value, 0))

# Non-dimensional model setup
model = Model(N = (128, 128, 64),
              L = (4π*h, 2π*h, 2h),
           arch = GPU(),
        closure = AnisotropicMinimumDissipation(ν=ν, κ=κ),
            eos = LinearEquationOfState(βT=1.0, βS=0.0),
      constants = PlanetaryConstants(f=0.0, g=1.0),
            bcs = BoundaryConditions(u=ubcs, v=vbcs, T=Tbcs))

# Add a bit of surface-concentrated noise to the initial condition
ε(z) = randn() * z/model.grid.Lz * (1 + z/model.grid.Lz)

T₀(x, y, z) = 2Θw * (1/2 + z/model.grid.Lz) * (1 + 5e-1 * ε(z))
u₀(x, y, z) = 2Uw * (1/2 + z/model.grid.Lz) * (1 + 5e-1 * ε(z)) * (1 + 0.5*sin(4π/model.grid.Lx * x))
v₀(x, y, z) = 5e-1 * ε(z)
w₀(x, y, z) = 5e-1 * ε(z)
S₀(x, y, z) = 5e-1 * ε(z)

set_ic!(model, u=u₀, v=v₀, w=w₀, T=T₀, S=S₀)

@printf(
    """
    stratified Couette flow

            N : %d, %d, %d
            L : %.3g, %.3g, %.3g
           Re : %.3f
           Ri : %.3f
           Pr : %.3f
            ν : %.3g
            κ : %.3g
           Uw : %.3f
           Θw : %.3f

    """, model.grid.Nx, model.grid.Ny, model.grid.Nz,
         model.grid.Lx, model.grid.Ly, model.grid.Lz,
         Re, Ri, Pr, ν, κ, Uw, Θw)

base_dir = "stratified_couette_flow_data"
prefix = @sprintf("stratified_couette_Re%d_Ri%.3f_Nz%d", Re, Ri, model.grid.Nz)

# Saving fields.
function init_save_parameters_and_bcs(file, model)
    file["parameters/reynolds_number"] = Re
    file["parameters/richardson_number"] = Ri
    file["parameters/prandtl_number"] = Pr
    file["parameters/viscosity"] = ν
    file["parameters/diffusivity"] = κ
    file["parameters/wall_velocity"] = Uw
    file["parameters/wall_temperature"] = Θw
end

fields = Dict(
    :u => model -> Array(model.velocities.u.data.parent),
    :v => model -> Array(model.velocities.v.data.parent),
    :w => model -> Array(model.velocities.w.data.parent),
    :T => model -> Array(model.tracers.T.data.parent),
    :kappaT => model -> Array(model.diffusivities.κₑ.T.data.parent),
    :nu => model -> Array(model.diffusivities.νₑ.data.parent))


field_writer = JLD2OutputWriter(model, fields; dir=base_dir, prefix=prefix * "_fields",
                                init=init_save_parameters_and_bcs,
                                max_filesize=25GiB, interval=10, force=true, verbose=true)
push!(model.output_writers, field_writer)

# Set up diagnostics.
push!(model.diagnostics, NaNChecker(model))

Δtₚ = 1 # Time interval for computing and saving profiles.

Up = HorizontalAverage(model, model.velocities.u; interval=Δtₚ)
Vp = HorizontalAverage(model, model.velocities.v; interval=Δtₚ)
Wp = HorizontalAverage(model, model.velocities.w; interval=Δtₚ)
Tp = HorizontalAverage(model, model.tracers.T;    interval=Δtₚ)
νp = HorizontalAverage(model, model.diffusivities.νₑ; interval=Δtₚ)
κp = HorizontalAverage(model, model.diffusivities.κₑ.T; interval=Δtₚ)

append!(model.diagnostics, [Up, Vp, Wp, Tp, νp, κp])

profiles = Dict(
     :u => model -> Array(Up.profile),
     :v => model -> Array(Vp.profile),
     :w => model -> Array(Wp.profile),
     :T => model -> Array(Tp.profile),
    :nu => model -> Array(νp.profile),
:kappaT => model -> Array(κp.profile))

profile_writer = JLD2OutputWriter(model, profiles; dir=base_dir, prefix=prefix * "_profiles",
                                  init=init_save_parameters_and_bcs, interval=Δtₚ, force=true, verbose=true)

push!(model.output_writers, profile_writer)

scalars = Dict(
    :u_tau2 => model -> uτ²(model),
        :qw => model -> qw(model),
 :theta_tau => model -> θτ(model),
   :delta_v => model -> δᵥ(model),
    :L_plus => model -> L⁺(model),
    :Re_tau => model -> Reτ(model),
    :Nu_tau => model -> Nu(model))

scalar_writer = JLD2OutputWriter(model, scalars; dir=base_dir, prefix=prefix * "_scalars",
                                 init=init_save_parameters_and_bcs, interval=Δtₚ/2, force=true, verbose=true)
push!(model.output_writers, scalar_writer)

wizard = TimeStepWizard(cfl=0.02, Δt=0.0001, max_change=1.1, max_Δt=0.02)

function cell_diffusion_timescale(model)
    Δ = min(model.grid.Δx, model.grid.Δy, model.grid.Δz)
    max_ν = maximum(model.diffusivities.νₑ.data.parent)
    max_κ = max(Tuple(maximum(κₑ.data.parent) for κₑ in model.diffusivities.κₑ)...)
    return min(Δ^2 / max_ν, Δ^2 / max_κ)
end

# Take Ni "intermediate" time steps at a time before printing a progress
# statement and updating the time step.
Ni = 10

cfl(t) = min(0.01*t, 0.1)

end_time = 1000
while model.clock.time < end_time
    wizard.cfl = cfl(model.clock.time)

    walltime = @elapsed time_step!(model; Nt=Ni, Δt=wizard.Δt)
    progress = 100 * (model.clock.time / end_time)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = wizard.Δt / cell_advection_timescale(model)

    νmax = maximum(model.diffusivities.νₑ.data.parent)
    κmax = maximum(model.diffusivities.κₑ.T.data.parent)
    
    Δ = min(model.grid.Δx, model.grid.Δy, model.grid.Δz)
    νCFL = wizard.Δt / (Δ^2 / νmax)
    κCFL = wizard.Δt / (Δ^2 / κmax)

    update_Δt!(wizard, model)

    @printf("[%06.2f%%] i: %d, t: %4.2f, umax: (%6.3g, %6.3g, %6.3g) m/s, CFL: %6.4g, νκmax: (%6.3g, %6.3g), νκCFL: (%6.4g, %6.4g), next Δt: %8.5g, ⟨wall time⟩: %s\n",
            progress, model.clock.iteration, model.clock.time,
            umax, vmax, wmax, CFL, νmax, κmax, νCFL, κCFL,
            wizard.Δt, prettytime(walltime / Ni))
end
