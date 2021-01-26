module JULES

using Oceananigans
using Oceananigans: AbstractGrid

using Oceananigans.TurbulenceClosures:
    IsotropicDiffusivity, DiffusivityFields, with_tracers

export
    Entropy, Energy,
    DryEarth, DryEarth3,
    CompressibleModel,
    update_total_density!,
    time_step!,
    cfl, acoustic_cfl

include("Operators/Operators.jl")

include("lazy_fields.jl")
include("thermodynamics.jl")
include("pressure_gradients.jl")
include("time_steppers.jl")
include("compressible_model.jl")

include("source_terms.jl")
include("time_stepping_kernels.jl")
include("time_stepping.jl")
include("utils.jl")

end # module
