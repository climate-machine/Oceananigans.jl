using Base: @propagate_inbounds

using OffsetArrays
using JLD2

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields

using Oceananigans.Grids: topology, total_size, interior_parent_indices
using Oceananigans.Fields: show_location

import Oceananigans: short_show
import Oceananigans.Fields: interior

struct FieldTimeSeries{X, Y, Z, K, A, T, N, D, G, B, χ} <: AbstractDataField{X, Y, Z, A, G, T, N}
                   data :: D
           architecture :: A
                   grid :: G
    boundary_conditions :: B
                  times :: χ
                   name :: String
               filepath :: String

    function FieldTimeSeries{X, Y, Z}(backend::K, data::D, arch::A, grid::G, bcs::B, times::χ, name, filepath, N) where {X, Y, Z, K, D, A, G, B, χ}
        T = eltype(grid)
        return new{X, Y, Z, K, A, T, N, D, G, B, χ}(data, arch, grid, bcs, times, name, filepath)
    end
end

#####
##### Constructors
#####

"""
    FieldTimeSeries(filepath, name; architecture=CPU(), backend=InMemory())

Returns a `FieldTimeSeries` for the field `name` describing a field's time history from a JLD2 file
located at `filepath`. Note that model output must have been saved with halos. The `InMemory` backend
will store the data fully in memory as a 4D multi-dimensional array while the `OnDisk` backend will
lazily load field time snapshots when the `FieldTimeSeries` is indexed linearly.
"""
FieldTimeSeries(filepath, name; architecture=CPU(), ArrayType=array_type(architecture), backend=InMemory()) =
    FieldTimeSeries(filepath, name, architecture, ArrayType, backend)

function FieldTimeSeries(filepath, name, architecture, ArrayType, backend::InMemory)
    file = jldopen(filepath)

    grid = file["serialized/grid"]
    Hx, Hy, Hz = halo_size(grid)

    iterations = parse.(Int, keys(file["timeseries/t"]))
    times = [file["timeseries/t/$i"] for i in iterations]

    LX, LY, LZ = location = file["timeseries/$name/serialized/location"]

    Nt = length(times)
    data_size = size(file["timeseries/$name/0"])

    raw_data = zeros(data_size..., Nt) |> ArrayType
    data = offset_data(raw_data, grid, location)

    for (n, iter) in enumerate(iterations)
        data.parent[:, :, :, n] .= file["timeseries/$name/$iter"] |> ArrayType
    end

    bcs = file["timeseries/$name/serialized/boundary_conditions"]

    close(file)

    return FieldTimeSeries{LX, LY, LZ}(backend, data, architecture, grid, bcs, times, name, abspath(filepath), ndims(data))
end

function FieldTimeSeries(filepath, name, architecture, ArrayType, backend::OnDisk)
    file = jldopen(filepath)

    grid = file["serialized/grid"]
    iterations = parse.(Int, keys(file["timeseries/t"]))
    times = [file["timeseries/t/$i"] for i in iterations]

    data = nothing
    LX, LY, LZ = file["timeseries/$name/serialized/location"]
    bcs = file["timeseries/$name/serialized/boundary_conditions"]

    close(file)

    return FieldTimeSeries{LX, LY, LZ}(backend, data, architecture, grid, bcs, times, name, abspath(filepath), 4)
end

# For creating an empty `FieldTimeSeries`.
function FieldTimeSeries(grid, location, times; architecture=CPU(), ArrayType=array_type(architecture), name="", filepath="", bcs=nothing)
    LX, LY, LZ = location

    Nt = length(times)
    data_size = total_size(location, grid)

    raw_data = zeros(data_size..., Nt) |> ArrayType
    data = offset_data(raw_data, grid, location)

    return FieldTimeSeries{LX, LY, LZ}(InMemory(), data, architecture, grid, bcs, times, name, filepath, 4)
end

#####
##### Methods
#####

# Include the time dimension.
@inline Base.size(fts::FieldTimeSeries) = (size(location(fts), fts.grid)..., length(fts.times))

@propagate_inbounds Base.getindex(f::FieldTimeSeries{X, Y, Z, InMemory}, i, j, k, n) where {X, Y, Z} = f.data[i, j, k, n]

Base.getindex(fts::FieldTimeSeries{X, Y, Z, InMemory}, n::Int) where {X, Y, Z} =
    Field((X, Y, Z), fts.architecture, fts.grid, fts.boundary_conditions, fts.data[:, :, :, n])

function Base.getindex(fts::FieldTimeSeries{X, Y, Z, OnDisk}, n::Int) where {X, Y, Z}
    file = jldopen(fts.filepath)
    iter = keys(file["timeseries/t"])[n]
    raw_data = file["timeseries/$(fts.name)/$iter"] |> array_type(fts.architecture)
    close(file)

    loc = (X, Y, Z)
    field_data = offset_data(raw_data, fts.grid, loc)
    return Field(loc, fts.architecture, fts.grid, fts.boundary_conditions, field_data)
end

Base.setindex!(fts::FieldTimeSeries, val, inds...) = Base.setindex!(fts.data, val, inds...)

interior(f::FieldTimeSeries{X, Y, Z}) where {X, Y, Z} =
    view(parent(f), interior_parent_indices(X, topology(f, 1), f.grid.Nx, f.grid.Hx),
                    interior_parent_indices(Y, topology(f, 2), f.grid.Ny, f.grid.Hy),
                    interior_parent_indices(Z, topology(f, 3), f.grid.Nz, f.grid.Hz),
                    :)

#####
##### Show methods
#####

backend_str(::InMemory) = "InMemory"
backend_str(::OnDisk) = "OnDisk"

short_show(fts::FieldTimeSeries{X, Y, Z, K}) where {X, Y, Z, K} =
    string("$(join(size(fts), "×")) FieldTimeSeries{$(backend_str(K()))} located at $(show_location(fts))")

Base.show(io::IO, fts::FieldTimeSeries{X, Y, Z, K, A}) where {X, Y, Z, K, A} =
    print(io, "$(short_show(fts))\n",
          "├── filepath: $(fts.filepath)\n",
          "├── architecture: $A\n",
          "└── grid: $(short_show(fts.grid))")
