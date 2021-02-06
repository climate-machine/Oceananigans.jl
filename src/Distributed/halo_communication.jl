using KernelAbstractions: @kernel, @index, Event, MultiEvent

import Oceananigans.BoundaryConditions:
    fill_halo_regions!,
    fill_west_halo!, fill_east_halo!, fill_south_halo!,
    fill_north_halo!, fill_bottom_halo!, fill_top_halo!

#####
##### MPI tags for halo communication BCs
#####

sides  = (:west, :east, :south, :north, :top, :bottom)

side_id = Dict(
    :east => 1, :west => 2,
    :north => 3, :south => 4,
    :top => 5, :bottom => 6
)

opposite_side = Dict(
    :east => :west, :west => :east,
    :north => :south, :south => :north,
    :top => :bottom, :bottom => :top
)

# Unfortunately can't call MPI.Comm_size(MPI.COMM_WORLD) before MPI.Init().
const MAX_RANKS = 10^3
RANK_DIGITS = 3

# Define functions that return unique send and recv MPI tags for each side.
# It's an integer where
#   digit 1: the side
#   digits 2-4: the from rank
#   digits 5-7: the to rank

for side in sides
    side_str = string(side)
    send_tag_fn_name = Symbol(side, :_send_tag)
    recv_tag_fn_name = Symbol(side, :_recv_tag)
    @eval begin
        function $send_tag_fn_name(my_rank, rank_to_send_to)
            from_digits = string(my_rank, pad=RANK_DIGITS)
            to_digits = string(rank_to_send_to, pad=RANK_DIGITS)
            side_digit = string(side_id[Symbol($side_str)])
            return parse(Int, from_digits * to_digits * side_digit)
        end

        function $recv_tag_fn_name(my_rank, rank_to_recv_from)
            from_digits = string(rank_to_recv_from, pad=RANK_DIGITS)
            to_digits = string(my_rank, pad=RANK_DIGITS)
            side_digit = string(side_id[opposite_side[Symbol($side_str)]])
            return parse(Int, from_digits * to_digits * side_digit)
        end
    end
end

#####
##### Filling halos for halo communication boundary conditions
#####

fill_halo_regions!(field::AbstractField{LX, LY, LZ}, arch::AbstractMultiArchitecture, args...) where {LX, LY, LZ} =
    fill_halo_regions!(field.data, field.boundary_conditions, arch, field.grid, (LX, LY, LZ), args...)

function fill_halo_regions!(c::AbstractArray, bcs, arch::AbstractMultiArchitecture, grid, location, args...)

    barrier = Event(device(child_architecture(arch)))

    east_event, west_event = fill_east_and_west_halos!(c, bcs.east, bcs.west, arch, barrier, grid, location, args...)
    # north_event, south_event = fill_north_and_south_halos!(c, bcs.north, bcs.south, arch, barrier, grid, args...)
    # top_event, bottom_event = fill_top_and_bottom_halos!(c, bcs.east, bcs.west, arch, barrier, grid, args...)

    events = [east_event, west_event] # , north_event, south_event, top_event, bottom_event]
    events = filter(e -> e isa Event, events)
    wait(device(child_architecture(arch)), MultiEvent(Tuple(events)))

    return nothing
end

function fill_east_and_west_halos!(c, east_bc, west_bc, arch, barrier, grid, location, args...)
    east_event = fill_east_halo!(c, east_bc, child_architecture(arch), barrier, grid, args...)
    west_event = fill_west_halo!(c, west_bc, child_architecture(arch), barrier, grid, args...)
    return east_event, west_event
end

function send_east_halo(c, grid, c_location, my_rank, rank_to_send_to)
    send_buffer = underlying_east_halo(c, grid, c_location)
    send_tag = east_send_tag(my_rank, rank_to_send_to)

    @debug "Sending east halo: my_rank=$my_rank, rank_to_send_to=$rank_to_send_to, send_tag=$send_tag"
    status = MPI.Isend(send_buffer, rank_to_send_to, send_tag, MPI.COMM_WORLD)

    return status
end

function send_west_halo(c, grid, c_location, my_rank, rank_to_send_to)
    send_buffer = underlying_west_halo(c, grid, c_location)
    send_tag = west_send_tag(my_rank, rank_to_send_to)

    @debug "Sending west halo: my_rank=$my_rank, rank_to_send_to=$rank_to_send_to, send_tag=$send_tag"
    status = MPI.Isend(send_buffer, rank_to_send_to, send_tag, MPI.COMM_WORLD)

    return status
end

function recv_and_fill_east_halo!(c, grid, c_location, my_rank, rank_to_recv_from)
    recv_buffer = underlying_east_halo(c, grid, c_location)
    recv_tag = east_recv_tag(my_rank, rank_to_recv_from)

    @debug "Receiving east halo: my_rank=$my_rank, rank_to_recv_from=$rank_to_recv_from, recv_tag=$recv_tag"
    MPI.Recv!(recv_buffer, rank_to_recv_from, recv_tag, MPI.COMM_WORLD)

    return nothing
end

function recv_and_fill_west_halo!(c, grid, c_location, my_rank, rank_to_recv_from)
    recv_buffer = underlying_west_halo(c, grid, c_location)
    recv_tag = west_recv_tag(my_rank, rank_to_recv_from)

    @debug "Receiving west halo: my_rank=$my_rank, rank_to_recv_from=$rank_to_recv_from, recv_tag=$recv_tag"
    MPI.Recv!(recv_buffer, rank_to_recv_from, recv_tag, MPI.COMM_WORLD)

    return nothing
end

function fill_east_and_west_halos!(c, east_bc::HaloCommunicationBC, west_bc::HaloCommunicationBC, arch, barrier, grid, c_location, args...)
    my_rank = east_bc.condition.from
    send_east_halo(c, grid, c_location, my_rank, east_bc.condition.to)
    send_west_halo(c, grid, c_location, my_rank, west_bc.condition.to)

    recv_and_fill_east_halo!(c, grid, c_location, my_rank, east_bc.condition.to)
    recv_and_fill_west_halo!(c, grid, c_location, my_rank, west_bc.condition.to)

    return nothing, nothing
end