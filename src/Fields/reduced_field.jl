using Oceananigans.Architectures: architecture

using Adapt

import Oceananigans.BoundaryConditions: fill_halo_regions!

#####
##### AbstractReducedField stuff
#####

abstract type AbstractReducedField{X, Y, Z, A, G, N} <: AbstractField{X, Y, Z, A, G} end

const ARF = AbstractReducedField

@inline Base.getindex( r::ARF{Nothing, Y, Z},    i, j, k) where {Y, Z} = @inbounds r.data[1, j, k]
@inline Base.setindex!(r::ARF{Nothing, Y, Z}, d, i, j, k) where {Y, Z} = @inbounds r.data[1, j, k] = d

@inline Base.getindex( r::ARF{X, Nothing, Z},    i, j, k) where {X, Z} = @inbounds r.data[i, 1, k]
@inline Base.setindex!(r::ARF{X, Nothing, Z}, d, i, j, k) where {X, Z} = @inbounds r.data[i, 1, k] = d

@inline Base.getindex( r::ARF{X, Y, Nothing},    i, j, k) where {X, Y} = @inbounds r.data[i, j, 1]
@inline Base.setindex!(r::ARF{X, Y, Nothing}, d, i, j, k) where {X, Y} = @inbounds r.data[i, j, 1] = d

@inline Base.getindex( r::ARF{X, Nothing, Nothing},    i, j, k) where X = @inbounds r.data[i, 1, 1]
@inline Base.setindex!(r::ARF{X, Nothing, Nothing}, d, i, j, k) where X = @inbounds r.data[i, 1, 1] = d

@inline Base.getindex( r::ARF{Nothing, Y, Nothing},    i, j, k) where Y = @inbounds r.data[1, j, 1]
@inline Base.setindex!(r::ARF{Nothing, Y, Nothing}, d, i, j, k) where Y = @inbounds r.data[1, j, 1] = d

@inline Base.getindex( r::ARF{Nothing, Nothing, Z},    i, j, k) where Z = @inbounds r.data[1, 1, k]
@inline Base.setindex!(r::ARF{Nothing, Nothing, Z}, d, i, j, k) where Z = @inbounds r.data[1, 1, k] = d

@inline Base.getindex( r::ARF{Nothing, Nothing, Nothing},    i, j, k) = @inbounds r.data[1, 1, 1]
@inline Base.setindex!(r::ARF{Nothing, Nothing, Nothing}, d, i, j, k) = @inbounds r.data[1, 1, 1] = d

fill_halo_regions!(field::AbstractReducedField, args...) =
    fill_halo_regions!(field.data, field.boundary_conditions, field.grid, args...; reduced_dimensions=field.dims)

const DimsType = NTuple{N, Int} where N

function validate_reduced_dims(dims)
    dims = Tuple(dims)

    # Check type
    dims isa DimsType || error("Reduced dims must be an integer or tuple of integers.")
    
    # Check length
    length(dims) > 3  && error("Models are 3-dimensional. Cannot reduce over 4+ dimensions.")

    # Check values
    all(1 <= d <= 3 for d in dims) || error("Dimensions must be one of 1, 2, 3.")

    return dims
end

function validate_reduced_locations(X, Y, Z, dims)
    loc = (X, Y, Z)

    # Check reduced locations
    for i in dims
        isnothing(loc[i]) || ArgumentError("The location of reduced dimensions must be Nothing")
    end

    return nothing
end

#####
##### Concrete ReducedField
#####

"""
    struct ReducedField{X, Y, Z, A, G, N} <: AbstractField{X, Y, Z, A, G}

Representation of a field at the location `(X, Y, Z)` with data of type `A`
on a grid of type `G` that is 'reduced' over `N` dimensions.
"""
struct ReducedField{X, Y, Z, A, G, N, B} <: AbstractReducedField{X, Y, Z, A, G, N}
                   data :: A
                   grid :: G
                   dims :: NTuple{N, Int}
    boundary_conditions :: B

    """
        ReducedField{X, Y, Z}(data, grid, dims)

    Returns a `ReducedField` at location `(X, Y, Z)` with `data` on `grid`
    that is reduced over the dimensions in `dims`.
    """
    function ReducedField{X, Y, Z}(data::A, grid::G, dims, bcs::B) where {X, Y, Z, A, G, B}

        dims = validate_reduced_dims(dims)
        validate_reduced_locations(X, Y, Z, dims)
        validate_field_data(X, Y, Z, data, grid)

        N = length(dims)

        return new{X, Y, Z, A, G, N, B}(data, grid, dims, bcs)
    end
    function ReducedField{X, Y, Z}(data::A, dims) where {X, Y, Z, A}
        N = length(dims)
        return new{X, Y, Z, A, Nothing, N, Nothing}(data, nothing, dims, nothing)
    end
end

"""
    ReducedField(X, Y, Z, arch, grid; dims, data=nothing, boundary_conditions=nothing)

Returns a `ReducedField` reduced over `dims` on `grid` and `arch`itecture with `boundary_conditions`.

The location `(X, Y, Z)` may be the parent, three-dimension location or the reduced location.

If `data` is specified, it should be an `OffsetArray` with singleton reduced dimensions;
otherwise `data` is allocated.

If `boundary_conditions` are not provided, default boundary conditions are constructed
using the reduced location.
"""
function ReducedField(Xr, Yr, Zr, arch, grid; dims, data=nothing,
                      boundary_conditions=nothing)

    dims = validate_reduced_dims(dims)

    # Reduce non-reduced dimensions
    X, Y, Z = reduced_location((Xr, Yr, Zr); dims=dims)

    if isnothing(data)
        data = new_data(arch, grid, (X, Y, Z))
    end

    if isnothing(boundary_conditions)
        boundary_conditions = FieldBoundaryConditions(grid, (X, Y, Z))
    end

    return ReducedField{X, Y, Z}(data, grid, dims, boundary_conditions)
end

ReducedField(loc::Tuple, args...; kwargs...) = ReducedField(loc..., args...; kwargs...)

#####
##### ReducedField utils
#####

reduced_location(loc; dims) = Tuple(i ∈ dims ? Nothing : loc[i] for i in 1:3)

Adapt.adapt_structure(to, reduced_field::ReducedField{X, Y, Z}) where {X, Y, Z} =
    ReducedField{X, Y, Z}(adapt(to, reduced_field.data), adapt(to, reduced_field.grid), reduced_field.dims, nothing)
