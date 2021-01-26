using Oceananigans
using Oceananigans.Advection: CenteredSecondOrder
using Oceananigans.Models: AbstractModel, Clock, tracernames
using Oceananigans.Forcings: zeroforcing, ContinuousForcing

#####
##### Definition of a compressible model
#####

mutable struct CompressibleModel{A, FT, Ω, ∇, D, M, V, T, L, K, Θ, G, X, C, F, S} <: AbstractModel
              architecture :: A
                      grid :: Ω
                     clock :: Clock{FT}
                 advection :: ∇
             total_density :: D
                   momenta :: M
                velocities :: V
                   tracers :: T
              lazy_tracers :: L
             diffusivities :: K
    thermodynamic_variable :: Θ
                     gases :: G
                   gravity :: FT
                  coriolis :: X
                   closure :: C
                   forcing :: F
              time_stepper :: S
end

#####
##### Constructor for compressible models
#####

const pₛ_Earth = 100000
const  g_Earth = 9.80665

function CompressibleModel(;
                      grid,
              architecture = CPU(),
                float_type = Float64,
                     clock = Clock{float_type}(0, 0),
                 advection = CenteredSecondOrder(),
                     gases = DryEarth(float_type),
    thermodynamic_variable = Energy(),
               tracernames = collect_tracers(thermodynamic_variable, gases),
                  coriolis = nothing,
                   closure = IsotropicDiffusivity(float_type, ν=0.5, κ=0.5),
       boundary_conditions = NamedTuple(),
                   momenta = MomentumFields(architecture, grid, boundary_conditions),
             diffusivities = DiffusivityFields(architecture, grid, tracernames, boundary_conditions, closure),
                   forcing = NamedTuple(),
                   gravity = g_Earth,
              time_stepper = WickerSkamarockTimeStepper(architecture, grid, tracernames))

    total_density = CellField(architecture, grid)

    gravity = float_type(gravity)
    tracers = TracerFields(tracernames, architecture, grid, boundary_conditions)
    closure = with_tracers(tracernames, closure)
    forcing = model_forcing(tracernames; forcing...)

    velocities = LazyVelocityFields(architecture, grid, total_density, momenta)
    lazy_tracers = LazyTracerFields(architecture, grid, total_density, tracers)

    return CompressibleModel(
        architecture, grid, clock, advection, total_density, momenta, velocities, tracers,
        lazy_tracers, diffusivities, thermodynamic_variable, gases, gravity,
        coriolis, closure, forcing, time_stepper)
end

using Oceananigans.Grids: short_show

Base.show(io::IO, model::CompressibleModel{A, FT}) where {A, FT} =
    print(io, "CompressibleModel{$A, $FT} with $(length(model.gases)) gas(es) ",
        "(time = $(prettytime(model.clock.time)), iteration = $(model.clock.iteration)) \n",
        "├── grid: $(short_show(model.grid))\n",
        "├── tracers: $(tracernames(model.tracers))\n",
        "├── closure: $(typeof(model.closure))\n",
        "└── coriolis: $(typeof(model.coriolis))\n")

#####
##### Utilities for constructing compressible models
#####

generate_key(::Entropy) = :ρs
generate_key(::Energy) = :ρe

function collect_tracers(thermodynamic_variable, args...)
    list = [generate_key(thermodynamic_variable)]
    for arg in args
        if arg !== nothing
            for name in keys(arg)
                push!(list, name)
            end
        end
    end
    return Tuple(list)
end

function DryEarth(FT = Float64)
    return (ρ = EarthN₂O₂(FT),)
end

function DryEarth3(FT = Float64)
    return (ρ₁ = EarthN₂O₂(FT), ρ₂ = EarthN₂O₂(FT), ρ₃ = EarthN₂O₂(FT))
end

function MomentumFields(arch, grid, bcs::NamedTuple)
    ρu_bcs = :ρu ∈ keys(bcs) ? bcs[:ρu] : UVelocityBoundaryConditions(grid)
    ρv_bcs = :ρv ∈ keys(bcs) ? bcs[:ρv] : VVelocityBoundaryConditions(grid)
    ρw_bcs = :ρw ∈ keys(bcs) ? bcs[:ρw] : WVelocityBoundaryConditions(grid)

    ρu = XFaceField(arch, grid, ρu_bcs)
    ρv = YFaceField(arch, grid, ρv_bcs)
    ρw = ZFaceField(arch, grid, ρw_bcs)

    return (ρu=ρu, ρv=ρv, ρw=ρw)
end

function TracerFields(names, arch, grid, bcs)
    tracer_names = tracernames(names) # filter `names` if it contains velocity fields
    tracer_fields =
        Tuple(c ∈ keys(bcs) ?
              CellField(arch, grid, bcs[c]) :
              CellField(arch, grid, TracerBoundaryConditions(grid))
              for c in tracer_names)
    return NamedTuple{tracer_names}(tracer_fields)
end

#####
##### Utilities for constructing model forcing
##### Should merge with incompressible version
#####

assumed_field_location(name) = name === :ρu ? (Face, Cell, Cell) :
                               name === :ρv ? (Cell, Face, Cell) :
                               name === :ρw ? (Cell, Cell, Face) :
                                              (Cell, Cell, Cell)

regularize_forcing(forcing, field_name, model_field_names) = forcing # fallback

function regularize_forcing(forcing::Function, field_name, model_field_names)
    X, Y, Z = assumed_field_location(field_name)
    return ContinuousForcing{X, Y, Z}(forcing)
end

regularize_forcing(::Nothing, field_name, model_field_names) = zeroforcing

function model_forcing(tracer_names; ρu=nothing, ρv=nothing, ρw=nothing, tracer_forcings...)

    model_field_names = tuple(:ρu, :ρv, :ρw, tracer_names...)

    ρu = regularize_forcing(ρu, :ρu, model_field_names)
    ρv = regularize_forcing(ρv, :ρv, model_field_names)
    ρw = regularize_forcing(ρw, :ρw, model_field_names)

    # Build tuple of user-specified tracer forcings
    specified_tracer_forcings_tuple = Tuple(regularize_forcing(f.second, f.first, model_field_names) for f in tracer_forcings)
    specified_tracer_names = Tuple(f.first for f in tracer_forcings)

    specified_forcings = NamedTuple{specified_tracer_names}(specified_tracer_forcings_tuple)

    # Re-build with defaults for unspecified tracer forcing
    tracer_forcings = with_tracers(tracer_names, specified_forcings, (name, initial_tuple) -> zeroforcing)

    return merge((ρu=ρu, ρv=ρv, ρw=ρw), tracer_forcings)
end
