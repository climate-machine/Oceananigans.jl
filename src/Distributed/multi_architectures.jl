using Oceananigans.Architectures
import Oceananigans.Architectures: device_event

using Oceananigans.Grids: topology, validate_tupled_argument

struct MultiCPU{G, R, I, ρ, C, γ} <: AbstractCPUArchitecture
    distributed_grid :: G
          local_rank :: R
         local_index :: I
               ranks :: ρ
        connectivity :: C
        communicator :: γ
end

struct MultiGPU{G, R, I, ρ, C, γ} <: AbstractGPUArchitecture
    distributed_grid :: G
          local_rank :: R
         local_index :: I
               ranks :: ρ
        connectivity :: C
        communicator :: γ
end

const AbstractMultiArchitecture = Union{MultiCPU, MultiGPU}

child_architecture(::MultiCPU) = CPU()
child_architecture(::CPU) = CPU()

child_architecture(::MultiGPU) = GPU()
child_architecture(::GPU) = GPU()

device_event(arch::AbstractMultiArchitecture) = device_event(child_architecture(arch))

#####
##### Converting between index and MPI rank taking k as the fast index
#####

index2rank(i, j, k, Rx, Ry, Rz) = (i-1)*Ry*Rz + (j-1)*Rz + (k-1)

function rank2index(r, Rx, Ry, Rz)
    i = div(r, Ry*Rz)
    r -= i*Ry*Rz
    j = div(r, Rz)
    k = mod(r, Rz)
    return i+1, j+1, k+1  # 1-based Julia
end

#####
##### Rank connectivity graph
#####

struct RankConnectivity{E, W, N, S, T, B}
      east :: E
      west :: W
     north :: N
     south :: S
       top :: T
    bottom :: B
end

RankConnectivity(; east, west, north, south, top, bottom) =
    RankConnectivity(east, west, north, south, top, bottom)

function increment_index(i, R, topo)
    R == 1 && return nothing
    if i+1 > R
        if topo == Periodic
            return 1
        else
            return nothing
        end
    else
        return i+1
    end
end

function decrement_index(i, R, topo)
    R == 1 && return nothing
    if i-1 < 1
        if topo == Periodic
            return R
        else
            return nothing
        end
    else
        return i-1
    end
end

function RankConnectivity(model_index, ranks, topology)
    i, j, k = model_index
    Rx, Ry, Rz = ranks
    TX, TY, TZ = topology

    i_east  = increment_index(i, Rx, TX)
    i_west  = decrement_index(i, Rx, TX)
    j_north = increment_index(j, Ry, TY)
    j_south = decrement_index(j, Ry, TY)
    k_top   = increment_index(k, Rz, TZ)
    k_bot   = decrement_index(k, Rz, TZ)

    r_east  = isnothing(i_east)  ? nothing : index2rank(i_east, j, k, Rx, Ry, Rz)
    r_west  = isnothing(i_west)  ? nothing : index2rank(i_west, j, k, Rx, Ry, Rz)
    r_north = isnothing(j_north) ? nothing : index2rank(i, j_north, k, Rx, Ry, Rz)
    r_south = isnothing(j_south) ? nothing : index2rank(i, j_south, k, Rx, Ry, Rz)
    r_top   = isnothing(k_top)   ? nothing : index2rank(i, j, k_top, Rx, Ry, Rz)
    r_bot   = isnothing(k_bot)   ? nothing : index2rank(i, j, k_bot, Rx, Ry, Rz)

    return RankConnectivity(east=r_east, west=r_west, north=r_north,
                            south=r_south, top=r_top, bottom=r_bot)
end

#####
##### Constructors
#####

function MultiArchitecture(MultiArch, grid, ranks, communicator)
    MPI.Initialized() || error("Must call MPI.Init() before constructing a MultiCPU.")

    validate_tupled_argument(ranks, Int, "ranks")

    Rx, Ry, Rz = ranks
    total_ranks = Rx*Ry*Rz

    mpi_ranks = MPI.Comm_size(communicator)
    local_rank   = MPI.Comm_rank(communicator)

    i, j, k = local_index = rank2index(local_rank, Rx, Ry, Rz)

    if total_ranks != mpi_ranks
        throw(ArgumentError("ranks=($Rx, $Ry, $Rz) [$total_ranks total] inconsistent " *
                            "with number of MPI ranks: $mpi_ranks."))
    end

    local_connectivity = RankConnectivity(local_index, ranks, topology(grid))

    return MultiArch(grid, local_rank, local_index, ranks, local_connectivity, communicator)
end

MultiCPU(; grid, ranks, communicator=MPI.COMM_WORLD) = MultiArchitecture(MultiCPU, grid, ranks, communicator)
MultiGPU(; grid, ranks, communicator=MPI.COMM_WORLD) = MultiArchitecture(MultiGPU, grid, ranks, communicator)

#####
##### Pretty printing
#####

function Base.show(io::IO, arch::MultiCPU)
    c = arch.connectivity
    print(io, "MultiCPU architecture (rank $(arch.local_rank)/$(prod(arch.ranks)-1)) [index $(arch.local_index) / $(arch.ranks)]\n",
              "└── connectivity:",
              isnothing(c.east) ? "" : " east=$(c.east)",
              isnothing(c.west) ? "" : " west=$(c.west)",
              isnothing(c.north) ? "" : " north=$(c.north)",
              isnothing(c.south) ? "" : " south=$(c.south)",
              isnothing(c.top) ? "" : " top=$(c.top)",
              isnothing(c.bottom) ? "" : " bottom=$(c.bottom)")
end
