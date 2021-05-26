using Adapt
using Statistics
using Oceananigans.Grids
using Oceananigans.Grids: interior_parent_indices

struct AveragedField{X, Y, Z, S, A, D, G, T, N, O} <: AbstractReducedField{X, Y, Z, A, G, T, N, D}
            data :: D
    architecture :: A
            grid :: G
            dims :: NTuple{N, Int}
         operand :: O
          status :: S

    function AveragedField{X, Y, Z}(data::D, arch::A, grid::G, dims, operand::O;
                                    recompute_safely=true) where {X, Y, Z, D, A, G, O}

        dims = validate_reduced_dims(dims)
        validate_reduced_locations(X, Y, Z, dims)
        validate_field_data(X, Y, Z, data, grid)

        status = recompute_safely ? nothing : FieldStatus(0.0)

        S = typeof(status)
        N = length(dims)
        T = eltype(grid)

        return new{X, Y, Z, S, A, D, G, T, N, O}(data, arch, grid, dims, operand, status)
    end

    function AveragedField{X, Y, Z}(data::D, arch::A, grid::G, dims, operand::O, status::S) where {X, Y, Z, D, A, G, O, S}
        return new{X, Y, Z, S, A, D, G, eltype(grid), length(dims), O}(data, arch, grid, dims, operand, status)
    end
end

"""
    AveragedField(operand::AbstractField; dims, data=nothing, recompute_safely=false)

Returns an AveragedField.
"""
function AveragedField(operand::AbstractField; dims, data=nothing, recompute_safely=true)

    arch = architecture(operand)
    loc = reduced_location(location(operand), dims=dims)
    grid = operand.grid

    if isnothing(data)
        data = new_data(arch, grid, loc)
        recompute_safely = false
    end

    return AveragedField{loc[1], loc[2], loc[3]}(data, arch, grid, dims, operand,
                                                 recompute_safely=recompute_safely)
end

"""
    compute!(avg::AveragedField, time=nothing)

Compute the average of `avg.operand` and store the result in `avg.data`.
"""
function compute!(avg::AveragedField, time=nothing)
    compute_at!(avg.operand, time)
    mean!(avg, avg.operand)
    return nothing
end

compute_at!(avg::AveragedField{X, Y, Z, <:FieldStatus}, time) where {X, Y, Z} =
    conditional_compute!(avg, time)

#####
##### Adapt
#####

Adapt.adapt_structure(to, averaged_field::AveragedField{X, Y, Z}) where {X, Y, Z} =
    AveragedField{X, Y, Z}(Adapt.adapt(to, averaged_field.data), nothing,
                           nothing, averaged_field.dims, nothing, nothing)
