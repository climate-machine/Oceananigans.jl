#####
##### 'Tupled closure' implementation.
#####
##### For now, we have hacked an implementation for tuples of two closures.
##### We cannot do the case of arbitrary closure tuples until we figure out how to
##### obtain the length of a closure tuple at compile-time on the GPU.
#####

with_tracers(tracers, closure_tuple::Tuple) =
    Tuple(with_tracers(tracers, closure) for closure in closure_tuple)

# Stress divergences

for stress_div in (:∂ⱼ_2ν_Σ₁ⱼ, :∂ⱼ_2ν_Σ₂ⱼ, :∂ⱼ_2ν_Σ₃ⱼ)
    @eval begin
        @inline $stress_div(i, j, k, grid::AbstractGrid, clock, closures::Tuple{C1, C2}, U, Ks) where {C1, C2} = (
                $stress_div(i, j, k, grid, clock, closures[1], U, Ks[1])
              + $stress_div(i, j, k, grid, clock, closures[2], U, Ks[2]))
    end
end

# Tracer flux divergences

@inline ∇_κ_∇c(i, j, k, grid::AbstractGrid, clock, closures::Tuple{C1, C2}, c, iᶜ, Ks, C, buoyancy) where {C1, C2} = (
        ∇_κ_∇c(i, j, k, grid, clock, closures[1], c, iᶜ, Ks[1], C, buoyancy)
      + ∇_κ_∇c(i, j, k, grid, clock, closures[2], c, iᶜ, Ks[2], C, buoyancy))

# Utilities for closures

function calculate_diffusivities!(Ks, arch, grid, closures::Tuple, args...)
    for (α, closure) in enumerate(closures)
        @inbounds K = Ks[α]
        calculate_diffusivities!(K, arch, grid, closure, args...)
    end
    return nothing
end

#####
##### Preliminary implementation for arbitrary length tupled closures
#####

for stress_div in (:∂ⱼ_2ν_Σ₁ⱼ, :∂ⱼ_2ν_Σ₂ⱼ, :∂ⱼ_2ν_Σ₃ⱼ)
    @eval begin
        @inline function $stress_div(i, j, k, grid::AbstractGrid{FT}, clock, closure_tuple::Tuple, U,
                                     K_tuple, args...) where FT

            stress_div_ijk = zero(FT)

            ntuple(Val(length(closure_tuple))) do α
                @inbounds closure = closure_tuple[α]
                @inbounds K = K_tuple[α]
                stress_div_ijk += $stress_div(i, j, k, grid, clock, closure, U, K, args...)
            end

            return stress_div_ijk
        end
    end
end

@inline function ∇_κ_∇c(i, j, k, grid::AbstractGrid{FT}, clock, closure_tuple::Tuple,
                        c, tracer_index, K_tuple, args...) where FT

    flux_div_ijk = zero(FT)

    ntuple(Val(length(closure_tuple))) do α
        @inbounds closure = closure_tuple[α]
        @inbounds K = K_tuple[α]
        flux_div_ijk +=  ∇_κ_∇c(i, j, k, grid, clock, closure, c, tracer_index, K, args...)
    end

    return flux_div_ijk
end

function calculate_diffusivities!(K_tuple::Tuple, arch, grid, closure_tuple::Tuple, args...)
    ntuple(Val(length(closure_tuple))) do α
        @inbounds closure = closure_tuple[α]
        @inbounds K = K_tuple[α]
        calculate_diffusivities!(K, arch, grid, closure, args...)
    end

    return nothing
end
