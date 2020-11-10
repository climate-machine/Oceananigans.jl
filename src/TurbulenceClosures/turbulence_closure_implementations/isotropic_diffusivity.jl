"""
    IsotropicDiffusivity{N, K}

Holds viscosity and diffusivities for models with prescribed isotropic diffusivities.
"""
struct IsotropicDiffusivity{N, K} <: AbstractIsotropicDiffusivity
    ν :: N
    κ :: K
end

"""
    IsotropicDiffusivity(; ν=ν₀, κ=κ₀)

Returns parameters for an isotropic diffusivity model with viscosity `ν`
and thermal diffusivities `κ` for each tracer field in `tracers`
`ν` and the fields of `κ` may be constants or functions of `(x, y, z, t)`, and
may represent molecular diffusivities in cases that all flow
features are explicitly resovled, or turbulent eddy diffusivities that model the effect of
unresolved, subgrid-scale turbulence.
`κ` may be a `NamedTuple` with fields corresponding
to each tracer, or a single number to be a applied to all tracers.

By default, a molecular viscosity of `ν₀ = 1.05×10⁻⁶` m² s⁻¹ and a molecular thermal
diffusivity of `κ₀ = 1.46×10⁻⁷` m² s⁻¹ is used for each tracer. These molecular values are
the approximate viscosity and thermal diffusivity for seawater at 20°C and 35 psu,
according to Sharqawy et al., "Thermophysical properties of seawater: A review of existing
correlations and data" (2010).
"""
function IsotropicDiffusivity(FT=Float64; ν=ν₀, κ=κ₀)
    if ν isa Number && κ isa Number
        κ = convert_diffusivity(FT, κ)
        return IsotropicDiffusivity(FT(ν), κ)
    else
        return IsotropicDiffusivity(ν, κ)
    end
end

function with_tracers(tracers, closure::IsotropicDiffusivity)
    κ = tracer_diffusivities(tracers, closure.κ)
    return IsotropicDiffusivity(closure.ν, κ)
end

# Support for ComputedField diffusivities
function calculate_diffusivities!(K, arch, grid, closure::IsotropicDiffusivity, args...)

    compute!(closure.ν)

    for κ in closure.κ
        compute!(κ)
    end

    return nothing
end

@inline function ∇_κ_∇c(i, j, k, grid, clock, closure::IsotropicDiffusivity,
                        c, ::Val{tracer_index}, args...) where tracer_index

    @inbounds κ = closure.κ[tracer_index]

    return ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, clock, κ, c)
end

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, clock, closure::IsotropicDiffusivity, U, args...) =
    ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, clock, closure.ν, U.u)

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, clock, closure::IsotropicDiffusivity, U, args...) =
    ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, clock, closure.ν, U.v)

@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, clock, closure::IsotropicDiffusivity, U, args...) =
    ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, clock, closure.ν, U.w)

Base.show(io::IO, closure::IsotropicDiffusivity) =
    print(io, "IsotropicDiffusivity: ν=$(closure.ν), κ=$(closure.κ)")
