using Oceananigans.Architectures: architecture

#####
##### General halo filling functions
#####

fill_halo_regions!(::Nothing, args...) = []

"""
    fill_halo_regions!(fields)

Fill halo regions for each field in the tuple `fields` according to their boundary
conditions, possibly recursing into `fields` if it is a nested tuple-of-tuples.
"""
function fill_halo_regions!(fields::Union{Tuple, NamedTuple}, args...)

    for field in fields
        fill_halo_regions!(field, args...)
    end

    return nothing
end

fill_halo_regions!(field, args...) =
    fill_halo_regions!(field.data, field.boundary_conditions, field.grid, args...)

"Fill halo regions in x, y, and z for a given field."
function fill_halo_regions!(c::AbstractArray, fieldbcs, grid, args...; kwargs...)

    arch = architecture(c)
    barrier = Event(device(arch))

      west_event =   fill_west_halo!(c, fieldbcs.west,   arch, barrier, grid, args...; kwargs...)
      east_event =   fill_east_halo!(c, fieldbcs.east,   arch, barrier, grid, args...; kwargs...)
     south_event =  fill_south_halo!(c, fieldbcs.south,  arch, barrier, grid, args...; kwargs...)
     north_event =  fill_north_halo!(c, fieldbcs.north,  arch, barrier, grid, args...; kwargs...)
    bottom_event = fill_bottom_halo!(c, fieldbcs.bottom, arch, barrier, grid, args...; kwargs...)
       top_event =    fill_top_halo!(c, fieldbcs.top,    arch, barrier, grid, args...; kwargs...)

    # Wait at the end
    events = [west_event, east_event, south_event, north_event, bottom_event, top_event]
    events = filter(e -> typeof(e) <: Event, events)
    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Halo-filling for nothing boundary conditions
#####

  fill_west_halo!(c, ::Nothing, args...; kwargs...) = nothing
  fill_east_halo!(c, ::Nothing, args...; kwargs...) = nothing
 fill_south_halo!(c, ::Nothing, args...; kwargs...) = nothing 
 fill_north_halo!(c, ::Nothing, args...; kwargs...) = nothing
   fill_top_halo!(c, ::Nothing, args...; kwargs...) = nothing
fill_bottom_halo!(c, ::Nothing, args...; kwargs...) = nothing
