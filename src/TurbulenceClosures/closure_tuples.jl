#####
##### 'Tupled closure' implementation: 1-tuple, 2-tuple, and then n-tuple by induction
#####

#####
##### Stress divergences
#####

for stress_div in (:∂ⱼ_τ₁ⱼ, :∂ⱼ_τ₂ⱼ, :∂ⱼ_τ₃ⱼ)
    @eval begin
        @inline $stress_div(i, j, k, grid::AbstractGrid, closures::Tuple{C1}, clock, U, Ks, args...) where {C1} =
                    $stress_div(i, j, k, grid, closures[1], clock, U, Ks[1], args...)

        @inline $stress_div(i, j, k, grid::AbstractGrid, closures::Tuple{C1, C2}, clock, U, Ks, args...) where {C1, C2} = (
                    $stress_div(i, j, k, grid, closures[1], clock, U, Ks[1], args...)
                  + $stress_div(i, j, k, grid, closures[2], clock, U, Ks[2], args...))

        @inline $stress_div(i, j, k, grid::AbstractGrid, closures::Tuple, clock, U, Ks, args...) = (
                    $stress_div(i, j, k, grid, closures[1:2], clock, U, Ks[1:2], args...)
                  + $stress_div(i, j, k, grid, closures[3:end], clock, U, K[3:end], args...))
    end
end

#####
##### Tracer flux divergences
#####

@inline ∇_dot_qᶜ(i, j, k, grid::AbstractGrid, closures::Tuple{C1}, c, iᶜ, clock, Ks, args...) where {C1} =
        ∇_dot_qᶜ(i, j, k, grid, closures[1], c, iᶜ, clock, Ks[1], args...)

@inline ∇_dot_qᶜ(i, j, k, grid::AbstractGrid, closures::Tuple{C1, C2}, c, iᶜ, clock, Ks, args...) where {C1, C2} = (
        ∇_dot_qᶜ(i, j, k, grid, closures[1], c, iᶜ, clock, Ks[1], args...)
      + ∇_dot_qᶜ(i, j, k, grid, closures[2], c, iᶜ, clock, Ks[2], args...))

@inline ∇_dot_qᶜ(i, j, k, grid::AbstractGrid, closures::Tuple, c, iᶜ, clock, Ks, args...) = (
        ∇_dot_qᶜ(i, j, k, grid, closures[1:2], c, iᶜ, clock, Ks[1:2], args...)
      + ∇_dot_qᶜ(i, j, k, grid, closures[3:end], c, iᶜ, clock, Ks[3:end], args...))

#####
##### Utilities
#####

with_tracers(tracers, closure_tuple::Tuple) =
    Tuple(with_tracers(tracers, closure) for closure in closure_tuple)

function calculate_diffusivities!(Ks, arch, grid, closures::Tuple, args...)
    for (α, closure) in enumerate(closures)
        @inbounds K = Ks[α]
        calculate_diffusivities!(K, arch, grid, closure, args...)
    end
    return nothing
end

#####
##### Support for VerticallyImplicitTimeDiscretization
#####

const EC = AbstractTurbulenceClosure{<:ExplicitTimeDiscretization}
const VIC = AbstractTurbulenceClosure{<:VerticallyImplicitTimeDiscretization}

# Filter explicitly-discretized closures.
@inline z_diffusivity(clo::Tuple{<:EC},        iᶜ, Ks, args...) = tuple(0)
@inline z_diffusivity(clo::Tuple{<:VIC},       iᶜ, Ks, args...) = tuple(z_diffusivity(clo[1], iᶜ, Ks[1], args...))
@inline z_diffusivity(clo::Tuple{<:VIC, <:EC}, iᶜ, Ks, args...) = tuple(z_diffusivity(clo[1], iᶜ, Ks[1], args...))
@inline z_diffusivity(clo::Tuple{<:EC, <:VIC}, iᶜ, Ks, args...) = tuple(z_diffusivity(clo[2], iᶜ, Ks[2], args...))

@inline z_diffusivity(clo::Tuple{<:VIC, <:VIC}, iᶜ, Ks, args...) = tuple(z_diffusivity(clo[1], iᶜ, Ks[1], args...),
                                                                         z_diffusivity(clo[2], iᶜ, Ks[2], args...))

@inline z_diffusivity(clo::Tuple, iᶜ, Ks, args...) = tuple(z_diffusivity(clo[1:2],   iᶜ, Ks[1:2],   args...)...,
                                                           z_diffusivity(clo[3:end], iᶜ, Ks[3:end], args...)...)

@inline z_viscosity(clo::Tuple{<:EC},         Ks, args...) = tuple(0)
@inline z_viscosity(clo::Tuple{<:VIC},        Ks, args...) = tuple(z_viscosity(clo[1], Ks[1], args...))
@inline z_viscosity(clo::Tuple{<:VIC, <:EC},  Ks, args...) = tuple(z_viscosity(clo[1], Ks[1], args...))
@inline z_viscosity(clo::Tuple{<:EC, <:VIC},  Ks, args...) = tuple(z_viscosity(clo[2], Ks[2], args...))

@inline z_viscosity(clo::Tuple{<:VIC, <:VIC}, Ks, args...) = tuple(z_viscosity(clo[1], Ks[1], args...),
                                                                   z_viscosity(clo[2], Ks[2], args...))

@inline z_viscosity(clo::Tuple, Ks, args...) = tuple(z_viscosity(clo[1:2],   Ks[1:2], args...)...,
                                                     z_viscosity(clo[3:end], Ks[3:end], args...)...)

for coeff in (:νᶜᶜᶜ, :νᶠᶠᶜ, :νᶠᶜᶠ, :νᶜᶠᶠ, :κᶜᶜᶠ, :κᶜᶠᶜ, :κᶠᶜᶜ)
    @eval begin
        @inline $coeff(i, j, k, grid, clock, ν::Tuple{C1})     where C1       = $coeff(i, j, k, grid, clock, ν[1])
        @inline $coeff(i, j, k, grid, clock, ν::Tuple{C1, C2}) where {C1, C2} = $coeff(i, j, k, grid, clock, ν[1])   + $coeff(i, j, k, grid, clock, ν[2])
        @inline $coeff(i, j, k, grid, clock, ν::Tuple)                        = $coeff(i, j, k, grid, clock, ν[1:2]) + $coeff(i, j, k, grid, clock, ν[3:end])
    end
end

const ImplicitClosure = AbstractTurbulenceClosure{TD} where TD <: VerticallyImplicitTimeDiscretization
const ExplicitOrNothing = Union{ExplicitTimeDiscretization, Nothing}

@inline combine_time_discretizations(disc) = disc

@inline combine_time_discretizations(::ExplicitTimeDiscretization, ::VerticallyImplicitTimeDiscretization)           = VerticallyImplicitTimeDiscretization()
@inline combine_time_discretizations(::VerticallyImplicitTimeDiscretization, ::ExplicitTimeDiscretization)           = VerticallyImplicitTimeDiscretization()
@inline combine_time_discretizations(::VerticallyImplicitTimeDiscretization, ::VerticallyImplicitTimeDiscretization) = VerticallyImplicitTimeDiscretization()
@inline combine_time_discretizations(::ExplicitTimeDiscretization, ::ExplicitTimeDiscretization)                     = ExplicitTimeDiscretization()

@inline combine_time_discretizations(disc1, disc2, other_discs...) =
    combine_time_discretizations(combine_time_discretizations(disc1, disc2), other_discs...)

@inline time_discretization(closures::Tuple) = combine_time_discretizations(time_discretization.(closures)...)
