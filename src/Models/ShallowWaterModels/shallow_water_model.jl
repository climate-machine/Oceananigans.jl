using Oceananigans: AbstractModel, AbstractOutputWriter, AbstractDiagnostic

using Oceananigans.Architectures: AbstractArchitecture, CPU
using Oceananigans.Advection: CenteredSecondOrder
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Fields: Field, tracernames, TracerFields, XFaceField, YFaceField, CenterField
using Oceananigans.Forcings: model_forcing
using Oceananigans.Grids: with_halo, topology, inflate_halo_size, halo_size, Flat
using Oceananigans.TimeSteppers: Clock, TimeStepper, update_state!
using Oceananigans.TurbulenceClosures: with_tracers, DiffusivityFields
using Oceananigans.Utils: tupleit

function ShallowWaterTendencyFields(arch, grid, tracer_names)

    uh = XFaceField(arch, grid)
    vh = YFaceField(arch, grid)
    h = CenterField(arch, grid)

    tracers = TracerFields(tracer_names, arch, grid)

    return merge((uh=uh, vh=vh, h=h), tracers)
end

function ShallowWaterSolutionFields(arch, grid, bcs)
    uh = XFaceField(arch, grid, bcs.uh)
    vh = YFaceField(arch, grid, bcs.vh)
    h = CenterField(arch, grid, bcs.h)

    return (uh=uh, vh=vh, h=h)
end

mutable struct ShallowWaterModel{G, A<:AbstractArchitecture, T, V, R, F, E, B, Q, C, K, TS} <: AbstractModel{TS}

                          grid :: G         # Grid of physical points on which `Model` is solved
                  architecture :: A         # Computer `Architecture` on which `Model` is run
                         clock :: Clock{T}  # Tracks iteration number and simulation time of `Model`
    gravitational_acceleration :: T         # Gravitational acceleration, full, or reduced
                     advection :: V         # Advection scheme for velocities _and_ tracers
                      coriolis :: R         # Set of parameters for the background rotation rate of `Model`
                       forcing :: F         # Container for forcing functions defined by the user
                       closure :: E         # Diffusive 'turbulence closure' for all model fields
                    bathymetry :: B         # Bathymetry/Topography for the model
                      solution :: Q         # Container for transports `uh`, `vh`, and height `h`
                       tracers :: C         # Container for tracer fields
            diffusivity_fields :: K         # Container for turbulent diffusivities
                   timestepper :: TS        # Object containing timestepper fields and parameters

end

"""
    ShallowWaterModel(;
                               grid,
                               gravitational_acceleration,
      architecture::AbstractArchitecture = CPU(),
                                   clock = Clock{eltype(grid)}(0, 0, 1),
                               advection = UpwindBiasedFifthOrder(),
                                coriolis = nothing,
                     forcing::NamedTuple = NamedTuple(),
                                 closure = nothing,
                              bathymetry = nothing,
                                 tracers = (),
                      diffusivity_fields = nothing,
         boundary_conditions::NamedTuple = NamedTuple(),
                     timestepper::Symbol = :RungeKutta3)

Construct a shallow water `Oceananigans.jl` model on `grid` with `gravitational_acceleration` constant.

Keyword arguments
=================

    - `grid`: (required) The resolution and discrete geometry on which `model` is solved.
    - `gravitational_acceleration`: (required) The gravitational accelaration constant.
    - `architecture`: `CPU()` or `GPU()`. The computer architecture used to time-step `model`.
    - `clock`: The `clock` for the model
    - `advection`: The scheme that advects velocities and tracers. See `Oceananigans.Advection`.
    - `coriolis`: Parameters for the background rotation rate of the model.
    - `forcing`: `NamedTuple` of user-defined forcing functions that contribute to solution tendencies.
    - `bathymetry`: The bottom bathymetry.
    - `tracers`: A tuple of symbols defining the names of the modeled tracers, or a `NamedTuple` of
                 preallocated `CenterField`s.
    - `diffusivity_fields`: Stores diffusivity fields when the closures require a diffusivity to be
                            calculated at each timestep.
    - `boundary_conditions`: `NamedTuple` containing field boundary conditions.
    - `timestepper`: A symbol that specifies the time-stepping method. Either `:QuasiAdamsBashforth2`,
                     `:RungeKutta3`.
"""
function ShallowWaterModel(;
                           grid,
                           gravitational_acceleration,
  architecture::AbstractArchitecture = CPU(),
                               clock = Clock{eltype(grid)}(0, 0, 1),
                           advection = UpwindBiasedFifthOrder(),
                            coriolis = nothing,
                 forcing::NamedTuple = NamedTuple(),
                             closure = nothing,
                          bathymetry = nothing,
                             tracers = (),
                  diffusivity_fields = nothing,
     boundary_conditions::NamedTuple = NamedTuple(),
                 timestepper::Symbol = :RungeKutta3)

    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    @assert topology(grid, 3) === Flat "ShallowWaterModel requires `topology(grid, 3) === Flat`. Use `topology = ($(topology(grid, 1)), $(topology(grid, 2)), Flat)` when constructing `grid`."

    Hx, Hy, Hz = inflate_halo_size(grid.Hx, grid.Hy, 0, topology(grid), advection, closure)
    grid = with_halo((Hx, Hy, 0), grid)

    prognostic_field_names = (:uh, :vh, :h, tracers...)
    default_boundary_conditions = NamedTuple{prognostic_field_names}(Tuple(FieldBoundaryConditions() for name in prognostic_field_names))
    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)

    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, prognostic_field_names)

    solution           = ShallowWaterSolutionFields(architecture, grid, boundary_conditions)
    tracers            = TracerFields(tracers, architecture, grid, boundary_conditions)
    diffusivity_fields = DiffusivityFields(diffusivity_fields, architecture, grid,
                                      tracernames(tracers), boundary_conditions, closure)

    # Instantiate timestepper if not already instantiated
    timestepper = TimeStepper(timestepper, architecture, grid, tracernames(tracers);
                              Gⁿ = ShallowWaterTendencyFields(architecture, grid, tracernames(tracers)),
                              G⁻ = ShallowWaterTendencyFields(architecture, grid, tracernames(tracers)))

    # Regularize forcing and closure for model tracer and velocity fields.
    model_fields = merge(solution, tracers)
    forcing = model_forcing(model_fields; forcing...)
    closure = with_tracers(tracernames(tracers), closure)

    model = ShallowWaterModel(grid,
                              architecture,
                              clock,
                              eltype(grid)(gravitational_acceleration),
                              advection,
                              coriolis,
                              forcing,
                              closure,
                              bathymetry,
                              solution,
                              tracers,
                              diffusivity_fields,
                              timestepper)

    update_state!(model)

    return model
end
