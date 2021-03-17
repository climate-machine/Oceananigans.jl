using Adapt
using Statistics
using KernelAbstractions: @kernel, @index, Event
using Oceananigans.Grids
using Oceananigans.BoundaryConditions: fill_halo_regions!

"""
    struct ComputedField{X, Y, Z, A, G, O} <: AbstractField{X, Y, Z, A, G}

Type representing a field computed from an operand.
"""
struct ComputedField{X, Y, Z, S, A, G, O, C} <: AbstractField{X, Y, Z, A, G}
                   data :: A
                   grid :: G
                operand :: O
    boundary_conditions :: C
                 status :: S

    function ComputedField{X, Y, Z}(data::A,
                                    grid::G,
                                    operand::O,
                                    boundary_conditions::C;
                                    recompute_safely=true) where {X, Y, Z, A, G, O, C}

        validate_field_data(X, Y, Z, data, grid)

        # Use FieldStatus if we want to avoid always recomputing
        status = recompute_safely ? nothing : FieldStatus(0.0)

        return new{X, Y, Z, typeof(status), A, G, O, C}(data, grid, operand, boundary_conditions, status)
    end
end

# We define special default boundary conditions for ComputedField, because currently
# `DefaultBoundaryCondition` uses ImpenetrableBoundaryCondition() in bounded directions
# and for fields on Faces, which is not what we want in general for ComputedFields
DefaultComputedFieldBoundaryCondition(::Type{Grids.Periodic}, loc) = PeriodicBoundaryCondition()
DefaultComputedFieldBoundaryCondition(::Type{Flat}, loc) = nothing
DefaultComputedFieldBoundaryCondition(::Type{Bounded}, ::Type{Center}) = NoFluxBoundaryCondition()
DefaultComputedFieldBoundaryCondition(::Type{Bounded}, ::Type{Face}) = nothing

function ComputedFieldBoundaryConditions(grid, loc;
                                           east = DefaultComputedFieldBoundaryCondition(topology(grid, 1), loc[1]),
                                           west = DefaultComputedFieldBoundaryCondition(topology(grid, 1), loc[1]),
                                          south = DefaultComputedFieldBoundaryCondition(topology(grid, 2), loc[2]),
                                          north = DefaultComputedFieldBoundaryCondition(topology(grid, 2), loc[2]),
                                         bottom = DefaultComputedFieldBoundaryCondition(topology(grid, 3), loc[3]),
                                            top = DefaultComputedFieldBoundaryCondition(topology(grid, 3), loc[3]))

    return FieldBoundaryConditions(grid, loc; east=east, west=west, south=south, north=north, bottom=bottom, top=top)
end

"""
    ComputedField(operand; data = nothing, recompute_safely = true,
                  boundary_conditions = ComputedFieldBoundaryConditions(operand.grid, location(operand))

Returns a field whose data is `computed` from `operand`.

If the keyword argument `data` is not provided, memory is allocated to store
the result. The `arch`itecture of `data` is inferred from `operand`.

If `data` is provided and `recompute_safely=false`, then "recomputation" of the `ComputedField`
is avoided if possible.

`boundary_conditions` are set to 
"""
function ComputedField(operand; data = nothing, recompute_safely = true,
                       boundary_conditions = ComputedFieldBoundaryConditions(operand.grid, location(operand)))
    
    loc = location(operand)
    arch = architecture(operand)
    grid = operand.grid

    if isnothing(data)
        data = new_data(arch, grid, loc)
        recompute_safely = false
    end

    return ComputedField{loc[1], loc[2], loc[3]}(data, grid, operand, boundary_conditions; recompute_safely=recompute_safely)
end

"""
    compute!(comp::ComputedField)

Compute `comp.operand` and store the result in `comp.data`.
"""
function compute!(comp::ComputedField{X, Y, Z}, time=nothing) where {X, Y, Z}
    compute_at!(comp.operand, time) # ensures any 'dependencies' of the computation are computed first

    arch = architecture(comp.data)

    workgroup, worksize = work_layout(comp.grid,
                                      :xyz,
                                      include_right_boundaries=true,
                                      location=(X, Y, Z))

    compute_kernel! = _compute!(device(arch), workgroup, worksize) 

    event = compute_kernel!(comp.data, comp.operand; dependencies=Event(device(arch)))

    wait(device(arch), event)

    fill_halo_regions!(comp)

    return nothing
end

compute_at!(field::ComputedField{X, Y, Z, <:FieldStatus}, time) where {X, Y, Z} =
    conditional_compute!(field, time)

"""Compute an `operand` and store in `data`."""
@kernel function _compute!(data, operand)
    i, j, k = @index(Global, NTuple)
    @inbounds data[i, j, k] = operand[i, j, k]
end

#####
##### Adapt
#####

Adapt.adapt_structure(to, computed_field::ComputedField) = Adapt.adapt(to, computed_field.data)
