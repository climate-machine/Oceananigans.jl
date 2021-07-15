import Oceananigans.BoundaryConditions

validate_boundary_condition(::Periodic, bc, loc, side) = throw(ArgumentError("$side boundary condition $bc is non-periodic. " *
                                                                             "Boundary conditions cannot be specified in Periodic directions!"))

validate_boundary_condition(::Flat, bc, loc, side) = throw(ArgumentError("$side boundary condition $bc is not nothing. " *
                                                                         "Boundary conditions cannot be specified in Flat directions!"))

validate_boundary_condition(::Bounded, bc, ::Center, side) = throw(ArgumentError("$side boundary condition $bc is invalid. " *
                                                                                 "Field at cell Center in Bounded directions must have either \n" *
                                                                                 "FluxBoundaryCondition, ValueBoundaryCondition, or GradientBoundaryCondition"))

validate_boundary_condition(::Bounded, bc, ::Face, side) = throw(ArgumentError("$side boundary condition $bc is invalid. " *
                                                                               "Field at cell Face in Bounded directions must have either \n" *
                                                                               "NormalFlowBoundaryCondition or nothing."))

# Whitelist...
CenterBoundedBCs = Union{BoundaryCondition{<:Value},
                         BoundaryCondition{<:Gradient},
                         BoundaryCondition{<:Flux}}

validate_boundary_condition(::Flat, ::Nothing, loc, side) = nothing
validate_boundary_condition(::Bounded, ::CenterBoundedBCs, ::Center, side) = nothing
validate_boundary_condition(::Bounded, ::Union{Nothing, BoundaryCondition{<:NormalFlow}}, ::Face, side) = nothing
validate_boundary_condition(::Periodic, ::BoundaryCondition{<:BoundaryConditions.Periodic}, loc, side) = nothing

# Validate boundary conditions
validate_field_boundary_conditions(::Nothing, grid, LX, LY, LZ) = nothing

function validate_field_boundary_conditions(bcs, grid, LX, LY, LZ)
    west_bc   = bcs.west
    east_bc   = bcs.east
    south_bc  = bcs.south
    north_bc  = bcs.north
    bottom_bc = bcs.bottom
    top_bc    = bcs.top

    TX, TY, TZ = topology(grid)

    validate_boundary_condition(TX, west_bc, LX, :west)
    validate_boundary_condition(TX, east_bc, LX, :east)

    validate_boundary_condition(TY, south_bc, LY, :south)
    validate_boundary_condition(TY, north_bc, LY, :north)

    validate_boundary_condition(TZ, top_bc, LZ, :top)
    validate_boundary_condition(TZ, bottom_bc, LZ, :bottom)

    return nothing
end

