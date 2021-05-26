using Base: @propagate_inbounds
using Statistics

using CUDA
using Adapt
using OffsetArrays

using Oceananigans.Architectures
using Oceananigans.Utils

using DimensionalData: AbstractDimArray
using Oceananigans.Grids: interior_indices, interior_parent_indices

import Base: minimum, maximum, extrema
import Statistics: mean
import Oceananigans: location, instantiated_location
import Oceananigans.Architectures: architecture
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Grids: interior_x_indices, interior_y_indices, interior_z_indices
import Oceananigans.Grids: total_size, topology, nodes, xnodes, ynodes, znodes, xnode, ynode, znode
import Oceananigans.Utils: datatuple

const ArchOrNothing = Union{AbstractArchitecture, Nothing}
const GridOrNothing = Union{AbstractGrid, Nothing}

"""
    AbstractField{X, Y, Z, A, G, T, N, D}

Abstract supertype for fields located at `(X, Y, Z)` on architecture `A`
and defined on a grid `G` with data `D` of eltype `T` and `N` dimensions.
"""
abstract type AbstractField{X, Y, Z, A <: ArchOrNothing, G <: GridOrNothing, T, N, D} <: AbstractDimArray{T, N, Tuple{}, D} end

"""
    AbstractDataField{X, Y, Z, A, G, T, N, D}

Abstract supertype for fields with concrete data in settable underlying arrays,
located at `(X, Y, Z)` on architecture `A` and defined on a grid `G` with data `D`
of eltype `T` and `N` dimensions.
"""
abstract type AbstractDataField{X, Y, Z, A, G, T, N, D} <: AbstractField{X, Y, Z, A, G, T, N, D} end

Base.IndexStyle(::AbstractField) = IndexCartesian()
Base.axes(field::AbstractField) = map(Base.oneto, size(field))

# We want iterate to work like it does for AbstractArray, not like AbstractDimArray.
function Base.iterate(A::AbstractField, state=(eachindex(A),))
    y = iterate(state...)
    y === nothing && return nothing
    A[y[1]], (state[1], Base.tail(y)...)
end

function validate_field_data(X, Y, Z, data, grid)
    Tx, Ty, Tz = total_size((X, Y, Z), grid)

    if size(data) != (Tx, Ty, Tz)
        e = "Cannot construct field at ($X, $Y, $Z) with size(data)=$(size(data)). " *
            "`data` must have size ($Tx, $Ty, $Tz)."
        throw(ArgumentError(e))
    end

    return nothing
end

# Endpoint for recursive `datatuple` function:
@inline datatuple(obj::AbstractField) = data(obj)

#####
##### Computing AbstractField
#####

# Note: overload compute! for custom fields to produce non-default behavior

"""
    compute!(field)

Computes `field.data`.
"""
compute!(field) = nothing

"""
    compute_at!(field, time)

Computes `field.data` at `time`. Falls back to compute!(field).
"""
compute_at!(field, time) = compute!(field)

mutable struct FieldStatus{T}
    time :: T
end

Adapt.adapt_structure(to, status::FieldStatus) = (time = status.time,)

"""
    conditional_compute!(field, time)

Computes `field.data` if `time != field.status.time`.
"""
function conditional_compute!(field, time)

    if time == zero(time) || time != field.status.time
        compute!(field, time)
        field.status.time = time
    end

    return nothing
end

# This edge case occurs if `fetch_output` is called with `model::Nothing`.
# We do the safe thing here and always compute.
conditional_compute!(field, ::Nothing) = compute!(field, nothing)

"""
    @compute(exprs...)

Call compute! on fields after defining them.
"""
macro compute(def)
    expr = Expr(:block)
    field = def.args[1]
    push!(expr.args, :($(esc(def))))
    push!(expr.args, :(compute!($(esc(field)))))
    return expr
end

#####
##### AbstractField functionality
#####

@inline location(a) = (Nothing, Nothing, Nothing)

"Returns the location `(X, Y, Z)` of an `AbstractField{X, Y, Z}`."
@inline location(::AbstractField{X, Y, Z}) where {X, Y, Z} = (X, Y, Z) # note no instantiation
@inline location(f, i) = location(f)[i]

@inline instantiated_location(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = (LX(), LY(), LZ())

"Returns the architecture of on which `f` is defined."
architecture(f::AbstractField) = f.architecture

"Returns the topology of a fields' `grid`."
@inline topology(f, args...) = topology(f.grid, args...)

"""
    size(f::AbstractField)

Returns the size of an `AbstractField{X, Y, Z}` located at `X, Y, Z`.
This is a 3-tuple of integers corresponding to the number of interior nodes
of `f` along `x, y, z`.
"""
Base.size(f::AbstractField) = size(location(f), f.grid)

"Returns the length of a field's `data`."
@inline Base.length(f::AbstractField) = prod(size(f))

"""
    total_size(field::AbstractField)

Returns a 3-tuple that gives the "total" size of a field including
both interior points and halo points.
"""
total_size(f::AbstractField) = total_size(location(f), f.grid)

Base.fill!(f::AbstractDataField, val) = fill!(parent(f), val)

#####
##### Accessing wrapped arrays
#####

"Returns `f.data` for `f::Field` or `f` for `f::AbstractArray."
@inline data(a) = a # fallback
@inline data(f::AbstractField) = f.data

@inline cpudata(a) = data(a)

@inline cpudata(f::AbstractField{X, Y, Z, <:AbstractGPUArchitecture}) where {X, Y, Z} =
    offset_data(Array(parent(f)), f.grid, location(f))

"Returns `f.data.parent` for `f::Field`."
@inline Base.parent(f::AbstractField) = parent(data(f))

"Returns a view of `f` that excludes halo points."
@inline interior(f::AbstractField{X, Y, Z}) where {X, Y, Z} =
    view(parent(f), interior_parent_indices(X, topology(f, 1), f.grid.Nx, f.grid.Hx),
                    interior_parent_indices(Y, topology(f, 2), f.grid.Ny, f.grid.Hy),
                    interior_parent_indices(Z, topology(f, 3), f.grid.Nz, f.grid.Hz))


@inline interior_copy(f::AbstractField{X, Y, Z}) where {X, Y, Z} =
    parent(f)[interior_parent_indices(X, topology(f, 1), f.grid.Nx, f.grid.Hx),
              interior_parent_indices(Y, topology(f, 2), f.grid.Ny, f.grid.Hy),
              interior_parent_indices(Z, topology(f, 3), f.grid.Nz, f.grid.Hz)]

#####
##### getindex
#####

# Don't use axes(f) to checkbounds; use axes(f.data)
Base.checkbounds(f::AbstractField, I...) = Base.checkbounds(f.data, I...)

@propagate_inbounds Base.getindex(f::AbstractDataField, inds::Integer...) = getindex(f.data, inds...)

# Linear indexing
@propagate_inbounds Base.getindex(f::AbstractDataField, i::Integer)  = parent(f)[i]

#####
##### setindex
#####

@propagate_inbounds function Base.setindex!(f::AbstractDataField, val, i::Integer, j::Integer, k::Integer)
    f.data[i, j, k] = val
    return f
end

# Linear indexing
@propagate_inbounds function Base.setindex!(f::AbstractDataField, val, i::Integer)
    parent(f)[i] = val
    return f
end

#####
##### Coordinates of fields
#####

@propagate_inbounds xnode(i, ψ::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = xnode(LX(), i, ψ.grid)
@propagate_inbounds ynode(j, ψ::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = ynode(LY(), j, ψ.grid)
@propagate_inbounds znode(k, ψ::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = znode(LZ(), k, ψ.grid)

@propagate_inbounds Base.lastindex(f::AbstractDataField) = lastindex(f.data)
@propagate_inbounds Base.lastindex(f::AbstractDataField, dim) = lastindex(f.data, dim)

xnodes(ψ::AbstractField) = xnodes(location(ψ, 1), ψ.grid)
ynodes(ψ::AbstractField) = ynodes(location(ψ, 2), ψ.grid)
znodes(ψ::AbstractField) = znodes(location(ψ, 3), ψ.grid)

nodes(ψ::AbstractField; kwargs...) = nodes(location(ψ), ψ.grid; kwargs...)

#####
##### fill_halo_regions!
#####

fill_halo_regions!(field::AbstractField, arch, args...) = fill_halo_regions!(field.data, field.boundary_conditions, arch, field.grid, args...)

#####
##### Field reductions
#####

const AbstractGPUDataField = AbstractDataField{X, Y, Z, GPU} where {X, Y, Z}

"""
    minimum(field::AbstractDataField; dims=:)
Compute the minimum value of an Oceananigans `field` over the given dimensions (not including halo points).
By default all dimensions are included.
"""
minimum(field::AbstractGPUDataField; dims=:) = minimum(interior_copy(field); dims=dims)

"""
    minimum(f, field::AbstractDataField; dims=:)
Returns the smallest result of calling the function `f` on each element of an Oceananigans `field`
(not including halo points) over the given dimensions. By default all dimensions are included.
"""
minimum(f, field::AbstractGPUDataField; dims=:) = minimum(f, interior_copy(field); dims=dims)

"""
    maximum(field::AbstractDataField; dims=:)
Compute the maximum value of an Oceananigans `field` over the given dimensions (not including halo points).
By default all dimensions are included.
"""
maximum(field::AbstractGPUDataField; dims=:) = maximum(interior_copy(field); dims=dims)

"""
    maximum(f, field::AbstractDataField; dims=:)
Returns the largest result of calling the function `f` on each element of an Oceananigans `field`
(not including halo points) over the given dimensions. By default all dimensions are included.
"""
maximum(f, field::AbstractGPUDataField; dims=:) = maximum(f, interior_copy(field); dims=dims)

"""
    mean(field::AbstractDataField; dims=:)
Compute the mean of an Oceananigans `field` over the given dimensions (not including halo points).
By default all dimensions are included.
"""
mean(field::AbstractGPUDataField; dims=:) = mean(interior_copy(field); dims=dims)

"""
    mean(f::Function, field::AbstractDataField; dims=:)
Apply the function `f` to each element of an Oceananigans `field` and take the mean over dimensions `dims`
(not including halo points). By default all dimensions are included.
"""
mean(f::Function, field::AbstractGPUDataField; dims=:) = mean(f, interior_copy(field); dims=dims)

# Risky to use these without tests. Docs would also be nice.
Statistics.norm(a::AbstractField) = sqrt(mapreduce(x -> x * x, +, interior(a)))
Statistics.dot(a::AbstractField, b::AbstractField) = mapreduce((x, y) -> x * y, +, interior(a), interior(b))
