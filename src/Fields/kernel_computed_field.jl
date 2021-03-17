using Oceananigans: AbstractModel
using Oceananigans.Grids
using Oceananigans.Utils: tupleit

struct KernelComputedField{X, Y, Z, S, A, G, K, C, F, P} <: AbstractField{X, Y, Z, A, G}
                     data :: A
                     grid :: G
                   kernel :: K
      boundary_conditions :: C
    computed_dependencies :: F
               parameters :: P
                   status :: S

    function KernelComputedField{X, Y, Z}(kernel::K, arch, grid;
                                          boundary_conditions = ComputedFieldBoundaryConditions(grid, (X, Y, Z)),
                                          computed_dependencies = (),
                                          parameters::P = nothing,
                                          data = nothing,
                                          recompute_safely = true) where {X, Y, Z, K, P}

        computed_dependencies = tupleit(computed_dependencies)

        if isnothing(data)
            data = new_data(arch, grid, (X, Y, Z))
        end

        # Use FieldStatus if we want to avoid always recomputing
        status = recompute_safely ? nothing : FieldStatus(0.0)

        G = typeof(grid)
        A = typeof(data)
        S = typeof(status)
        F = typeof(computed_dependencies)
        C = typeof(boundary_conditions)

        return new{X, Y, Z, S,
                   A, G, K, C, F, P}(data, grid, kernel, boundary_conditions, computed_dependencies, parameters)
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
    - If `data=nothing`, then `recompute_safely` is switched to `false`.

Example
=======

```julia
using KernelAbstractions: @index, @kernel
using Oceananigans.Fields: AveragedField, KernelComputedField, compute!
using Oceananigans.Grids: Center, Face

@inline ψ²(i, j, k, grid, ψ, Ψ) = @inbounds (ψ[i, j, k] - Ψ[i, j, k])^2
@kernel function compute_variance!(var, grid, ϕ, Φ)
    i, j, k = @index(Global, NTuple)

    @inbounds var[i, j, k] = ψ′²(i, j, k, grid, ϕ, Φ)
end

u, v, w = model.velocities

U = AveragedField(u, dims=(1, 2))
V = AveragedField(v, dims=(1, 2))

u′² = KernelComputedField(Face, Center, Center, compute_variance!, model; computed_dependencies=(u, U,))
v′² = KernelComputedField(Center, Face, Center, compute_variance!, model; computed_dependencies=(v, V,))
w′² = KernelComputedField(Center, Center, Face, compute_variance!, model; computed_dependencies=(w, 0,))

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

    arch = architecture(kcf.data)

    workgroup, worksize = work_layout(kcf.grid,
                                      :xyz,
                                      location=(X, Y, Z),
                                      include_right_boundaries=true)

    compute_kernel! = kcf.kernel(device(arch), workgroup, worksize)

    event = isnothing(kcf.parameters) ?
        compute_kernel!(kcf.data, kcf.grid, kcf.computed_dependencies...) :
        compute_kernel!(kcf.data, kcf.grid, kcf.computed_dependencies..., kcf.parameters)

    wait(device(arch), event)

    fill_halo_regions!(kcf)

    return nothing
end

compute!(field::KernelComputedField{X, Y, Z, <:FieldStatus}, time) where {X, Y, Z} =
    conditional_compute!(field, time)

Adapt.adapt_structure(to, kcf::KernelComputedField) = Adapt.adapt(to, kcf.data)
