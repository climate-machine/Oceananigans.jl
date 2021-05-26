using Oceananigans: AbstractModel
using Oceananigans.Grids
using Oceananigans.Utils: tupleit

struct KernelComputedField{X, Y, Z, A, S, D, G, T, K, B, F, P} <: AbstractDataField{X, Y, Z, A, G, T, 3, D}
                     data :: D
             architecture :: A
                     grid :: G
                   kernel :: K
      boundary_conditions :: B
    computed_dependencies :: F
               parameters :: P
                   status :: S

    function KernelComputedField{X, Y, Z}(kernel::K, arch::A, grid::G;
                                          boundary_conditions::B = ComputedFieldBoundaryConditions(grid, (X, Y, Z)),
                                           computed_dependencies = (),
                                                   parameters::P = nothing,
                                                         data::D = new_data(arch, grid, (X, Y, Z)),
                                                recompute_safely = true) where {X, Y, Z, A, D, B, G, K, P}

        computed_dependencies = tupleit(computed_dependencies)

        # Use FieldStatus if we want to avoid always recomputing
        status = recompute_safely ? nothing : FieldStatus(0.0)

        S = typeof(status)
        F = typeof(computed_dependencies)
        T = eltype(grid)

        return new{X, Y, Z, A, S, D, G, T, K, B, F, P}(
            data, arch, grid, kernel, boundary_conditions, computed_dependencies, parameters, status)
    end
end

"""
    KernelComputedField(X, Y, Z, kernel, model;
                        boundary_conditions = ComputedFieldBoundaryConditions(grid, (X, Y, Z)),
                        computed_dependencies = (),
                        parameters = nothing,
                        data = nothing,
                        recompute_safely = true)

Builds a `KernelComputedField` at `X, Y, Z` computed with `kernel` and `model.architecture` and `model.grid`, with `boundary_conditions`.

`computed_dependencies` are an iterable of `AbstractField`s or other objects on which `compute!` is called prior to launching `kernel`.

`data` is a three-dimensional `OffsetArray` of scratch space where the kernel computation is stored.

If `data=nothing` (the default) then additional memory will be allocated to store the `data` of `KernelComputedField`.

If `isnothing(parameters)`, `kernel` is launched with the function signature

`kernel(data, grid, computed_dependencies...)`

Otherwise, `kernel` is launched with the function signature

`kernel(data, grid, computed_dependencies..., parameters)`

`recompute_safely` (default: `true`) determines whether the `KernelComputedField` is "recomputed" if embedded in the expression
tree of another operation.
    - If `recompute_safely=true`, the `KernelComputedField` is always recomputed.
    - If `recompute_safely=false`, the `KernelComputedField` will not be recomputed if its status is up-to-date.

Example
=======

```julia
using KernelAbstractions: @index, @kernel
using Oceananigans.Fields: AveragedField, KernelComputedField, compute!
using Oceananigans.Grids: Center, Face

@inline ψ′²(i, j, k, grid, ψ, Ψ) = @inbounds (ψ[i, j, k] - Ψ[i, j, k])^2
@inline ψ′²(i, j, k, grid, ψ, Ψ::Number) = @inbounds (ψ[i, j, k] - Ψ)^2

@kernel function compute_variance!(var, grid, ϕ, Φ)
    i, j, k = @index(Global, NTuple)

    @inbounds var[i, j, k] = ψ′²(i, j, k, grid, ϕ, Φ)
end

u, v, w = model.velocities

U = AveragedField(u, dims=(1, 2))
V = AveragedField(v, dims=(1, 2))

u′² = KernelComputedField(Face, Center, Center, compute_variance!, model; computed_dependencies=(u, U))
v′² = KernelComputedField(Center, Face, Center, compute_variance!, model; computed_dependencies=(v, V))
w′² = KernelComputedField(Center, Center, Face, compute_variance!, model; computed_dependencies=(w, 0))

compute!(u′²)
compute!(v′²)
compute!(w′²)
```
"""
KernelComputedField(X, Y, Z, kernel, model::AbstractModel; kwargs...) =
    KernelComputedField{X, Y, Z}(kernel, model.architecture, model.grid; kwargs...)

KernelComputedField(X, Y, Z, kernel, arch::AbstractArchitecture, grid::AbstractGrid; kwargs...) =
    KernelComputedField{X, Y, Z}(kernel, arch, grid; kwargs...)

function compute!(kcf::KernelComputedField{X, Y, Z}) where {X, Y, Z}

    for dependency in kcf.computed_dependencies
        compute!(dependency)
    end

    arch = architecture(kcf)

    args = isnothing(kcf.parameters) ?
        tuple(kcf.data, kcf.grid, kcf.computed_dependencies...) :
        tuple(kcf.data, kcf.grid, kcf.computed_dependencies..., kcf.parameters)

    event = launch!(arch, kcf.grid, :xyz, kcf.kernel, args...;
                    location=(X, Y, Z), include_right_boundaries=true)

    wait(device(arch), event)

    fill_halo_regions!(kcf, arch)

    return nothing
end

compute!(field::KernelComputedField{X, Y, Z, <:FieldStatus}, time) where {X, Y, Z} =
    conditional_compute!(field, time)

Adapt.adapt_structure(to, kcf::KernelComputedField) = Adapt.adapt(to, kcf.data)
