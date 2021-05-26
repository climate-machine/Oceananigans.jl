using DimensionalData: AbstractDimArray, dims

import DimensionalData

DimensionalData.dims(field::AbstractField{X, Y, Z}) where {X, Y, Z} = dims(field.grid, (X, Y, Z))

# We cannot infer these things just yet...
DimensionalData.name(field::AbstractField) = nothing
DimensionalData.refdims(field::AbstractField) = tuple()
DimensionalData.metadata(field::AbstractField) = nothing

function DimensionalData.rebuild(field::Field{X, Y, Z}, data, dims, refdims, name, metadata) where {X, Y, Z}
    # TODO: Need to rebuild data and dims too!
    return Field{X, Y, Z}(data, field.architecture, field.grid, field.boundary_conditions)
end
