module OutputWriters

export JLD2OutputWriter, NetCDFOutputWriter
export Checkpointer
export WindowedTimeAverage
export TimeInterval, IterationInterval, WallTimeInterval, AveragedTimeInterval

using CUDA

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Models

using Oceananigans: AbstractOutputWriter
using Oceananigans.Fields: OffsetArray, FieldSlicer
using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval

import Oceananigans: write_output!

Base.open(ow::AbstractOutputWriter) = nothing
Base.close(ow::AbstractOutputWriter) = nothing

include("output_writer_utils.jl")
include("fetch_output.jl")
include("windowed_time_average.jl")
include("time_average_outputs.jl")
include("jld2_output_writer.jl")
include("netcdf_output_writer.jl")
include("checkpointer.jl")

end
