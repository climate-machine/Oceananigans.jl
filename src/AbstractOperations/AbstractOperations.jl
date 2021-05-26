module AbstractOperations

export ∂x, ∂y, ∂z, @at, @unary, @binary, @multiary

using Base: @propagate_inbounds

import Adapt
using CUDA

using Oceananigans
using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BoundaryConditions
using Oceananigans.Fields

using Oceananigans.Operators: interpolation_operator
using Oceananigans.Architectures: device
using Oceananigans: AbstractModel

import Oceananigans.Architectures: architecture
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Fields: data, compute_at!

#####
##### Basic functionality
#####

abstract type AbstractOperation{X, Y, Z, A, G, T} <: AbstractField{X, Y, Z, A, G, T, 3, Nothing} end

const AF = AbstractField

# We (informally) require that all field-like objects define `parent`:
Base.parent(op::AbstractOperation) = op

# We have no halos to fill
fill_halo_regions!(::AbstractOperation, args...; kwargs...) = nothing

# AbstractOperation macros add their associated functions to this list
const operators = Set()

"""
    at(loc, abstract_operation)

Returns `abstract_operation` relocated to `loc`ation.
"""
at(loc, f) = f # fallback

include("grid_validation.jl")
include("grid_metrics.jl")
include("unary_operations.jl")
include("binary_operations.jl")
include("multiary_operations.jl")
include("derivatives.jl")
include("kernel_function_operation.jl")
include("at.jl")
include("broadcasting_abstract_operations.jl")
include("show_abstract_operations.jl")

# Make some operators!

# Some unaries:
import Base: sqrt, sin, cos, exp, tanh, -, +, /, ^, *

@unary sqrt sin cos exp tanh
@unary -

@binary +
@binary -
@binary /
@binary ^

@multiary +

# For unknown reasons, the operator definition macros @binary and @multiary fail to work
# properly for :*. We thus manually define :* for fields.
import Base: *

eval(define_binary_operator(:*))
push!(operators, :*)
push!(binary_operators, :*)

eval(define_multiary_operator(:*))
push!(operators, :*)
push!(multiary_operators, :*)

end # module

