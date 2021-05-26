using Adapt
using KernelAbstractions

using Oceananigans.Fields: AbstractDataField, FieldStatus, validate_field_data, conditional_compute!
using Oceananigans.Fields: architecture, tracernames
using Oceananigans.Architectures: device
using Oceananigans.Utils: work_layout
using Oceananigans.Grids: new_data

import Oceananigans.Fields: compute!, compute_at!

import Oceananigans: short_show

struct BuoyancyField{B, S, A, D, G, T, C} <: AbstractDataField{Center, Center, Center, A, G, T, 3, D}
            data :: D
    architecture :: A
            grid :: G
        buoyancy :: B
         tracers :: C
          status :: S

    """
        BuoyancyField(data, grid, buoyancy, tracers)

    Returns a `BuoyancyField` with `data` on `grid` corresponding to
    `buoyancy` computed from `tracers`.
    """
    function BuoyancyField(data::D, arch::A, grid::G, buoyancy::B, tracers::C,
                           recompute_safely::Bool) where {D, A, G, B, C}

        validate_field_data(Center, Center, Center, data, grid)

        status = recompute_safely ? nothing : FieldStatus(zero(eltype(grid)))

        S = typeof(status)
        T = eltype(grid)

        return new{B, S, A, D, G, T, C}(data, arch, grid, buoyancy, tracers, status)
    end

    function BuoyancyField(data::D, arch::A, grid::G, buoyancy::B, tracers::C, status::S) where {D, A, G, B, C, S}
        validate_field_data(Center, Center, Center, data, grid)
        T = eltype(grid)
        return new{B, S, A, D, G, T, C}(data, grid, buoyancy, tracers, status)
    end
end

"""
    BuoyancyField(model; data=nothing, recompute_safely=true)

Returns a `BuoyancyField` corresponding to `model.buoyancy`.
Calling `compute!(b::BuoyancyField)` computes the current buoyancy field
associated with `model` and stores the result in `b.data`.
"""
BuoyancyField(model; data=nothing, recompute_safely=true) =
    _buoyancy_field(model.buoyancy, model.tracers, model.architecture, model.grid, data, recompute_safely)

# Convenience for buoyancy=nothing
_buoyancy_field(::Nothing, args...; kwargs...) = nothing

#####
##### BuoyancyTracer
#####

_buoyancy_field(buoyancy::BuoyancyTracerModel, tracers, arch, grid, args...) =
    BuoyancyField(tracers.b.data, arch, grid, buoyancy, tracers, true)

compute!(::BuoyancyField{<:BuoyancyTracerModel}, time=nothing) = nothing

#####
##### Other buoyancy types
#####

function _buoyancy_field(buoyancy::Buoyancy, tracers, arch, grid,
                         data, recompute_safely)

    if isnothing(data)
        data = new_data(arch, grid, (Center, Center, Center))
        recompute_safely = false
    end

    return BuoyancyField(data, arch, grid, buoyancy, tracers, recompute_safely)
end

"""
    compute!(buoyancy_field::BuoyancyField)

Compute the current `buoyancy_field` associated with `buoyancy_field.tracers` and store
the result in `buoyancy_field.data`.
"""
function compute!(buoyancy_field::BuoyancyField, time=nothing)

    data = buoyancy_field.data
    grid = buoyancy_field.grid
    tracers = buoyancy_field.tracers
    buoyancy = buoyancy_field.buoyancy
    arch = architecture(buoyancy_field)

    workgroup, worksize = work_layout(grid, :xyz)

    compute_kernel! = compute_buoyancy!(device(arch), workgroup, worksize)

    event = compute_kernel!(data, grid, buoyancy, tracers; dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end

compute_at!(b::BuoyancyField{B, <:FieldStatus}, time) where B =
    conditional_compute!(b, time)

"""Compute an `operation` and store in `data`."""
@kernel function compute_buoyancy!(data, grid, buoyancy, tracers)
    i, j, k = @index(Global, NTuple)
    @inbounds data[i, j, k] = buoyancy_perturbation(i, j, k, grid, buoyancy.model, tracers)
end

#####
##### Adapt
#####

Adapt.adapt_structure(to, buoyancy_field::BuoyancyField) = Adapt.adapt(to, buoyancy_field.data)

#####
##### Show
#####

short_show(field::BuoyancyField) = string("BuoyancyField for ", typeof(field.buoyancy))

show(io::IO, field::BuoyancyField) =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field.data))\n",
          "├── grid: $(short_show(field.grid))", '\n',
          "├── buoyancy: $(typeof(field.buoyancy))", '\n',
          "├── tracers: $(tracernames(field.tracers))", '\n',
          "└── status: ", show_status(field.status), '\n')
