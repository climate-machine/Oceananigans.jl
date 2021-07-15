module Fields

export Face, Center
export AbstractField, AbstractDataField, Field
export CenterField, XFaceField, YFaceField, ZFaceField
export ReducedField, AveragedField, ComputedField, KernelComputedField, BackgroundField
export interior, data
export xnode, ynode, znode, location
export set!, compute!, @compute
export VelocityFields, TracerFields, tracernames, PressureFields, TendencyFields
export interpolate, FieldSlicer

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.BoundaryConditions

include("abstract_field.jl")
include("validate_field_boundary_conditions.jl")
include("reduced_getindex_setindex.jl")
include("field.jl")
include("zero_field.jl")
include("reduced_field.jl")
include("averaged_field.jl")
include("computed_field.jl")
include("kernel_computed_field.jl")
include("pressure_field.jl")
include("function_field.jl")
include("set!.jl")
include("tracer_names.jl")
include("validate_field_tuple_grid.jl")
include("field_tuples.jl")
include("background_fields.jl")
include("interpolate.jl")
include("field_slicer.jl")
include("show_fields.jl")
include("broadcasting_abstract_fields.jl")
include("mapreduce_abstract_fields.jl")

# Fallback: cannot infer boundary conditions.
boundary_conditions(field) = nothing
boundary_conditions(f::Union{Field, ReducedField, ComputedField, KernelComputedField}) = f.boundary_conditions

end
