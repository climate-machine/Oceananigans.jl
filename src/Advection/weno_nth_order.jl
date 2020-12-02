#####
##### Weighted Essentially Non-Oscillatory (WENO) schemes
#####

struct WENO{K, FT, C, B, Γ} <: AbstractAdvectionScheme
    C :: C
    B :: B
    γ :: Γ
    ε :: FT
    η :: FT
end

function WENO(FT, n)
    (n < 3 || iseven(n)) &&
        error("WENO schemes are only defined for order n = 3, 5, 7, 9, 11, ...")

    @warn "Generated WENO schemes are experimental! They may be very slow (allocate tons of memory) and may not work on GPUs."

    n >= 7 &&
        @info "Computing WENO-$n smoothness indicator coefficients. This could take a while for n >= 7..."

    k = Int((n + 1) / 2)
    C = eno_coefficients_matrix(FT, k)
    B = β_coefficients(FT, k)
    γ = optimal_weno_weights(FT, k)

    ε = 1e-6  # To avoid dividing by zero
    η = 2     # WENO exponent

    return WENO{k, FT, typeof(C), typeof(B), typeof(γ)}(C, B, γ, ε, η)
end

WENO(n) = WENO(Float64, n)

#####
##### ENO interpolants
#####

@inline px(i, j, k, weno::WENO{K}, r, ϕ) where K = @inbounds sum(weno.C[ℓ+1, r+1] * ϕ[i-r-1+ℓ, j, k] for ℓ in 0:K-1)
@inline py(i, j, k, weno::WENO{K}, r, ϕ) where K = @inbounds sum(weno.C[ℓ+1, r+1] * ϕ[i, j-r-1+ℓ, k] for ℓ in 0:K-1)
@inline pz(i, j, k, weno::WENO{K}, r, ϕ) where K = @inbounds sum(weno.C[ℓ+1, r+1] * ϕ[i, j, k-r-1+ℓ] for ℓ in 0:K-1)

#####
##### Jiang & Shu (1996) WENO smoothness indicators
#####

@inline βx(i, j, k, weno::WENO{K}, r, ϕ) where K = @inbounds sum(weno.B[m+1, n+1, r+1] * ϕ[i-r-1+m, j, k] * ϕ[i-r-1+n, j, k] for m in 0:K-1 for n in 0:m)
@inline βy(i, j, k, weno::WENO{K}, r, ϕ) where K = @inbounds sum(weno.B[m+1, n+1, r+1] * ϕ[i, j-r-1+m, k] * ϕ[i, j-r-1+n, k] for m in 0:K-1 for n in 0:m)
@inline βz(i, j, k, weno::WENO{K}, r, ϕ) where K = @inbounds sum(weno.B[m+1, n+1, r+1] * ϕ[i, j, k-r-1+m] * ϕ[i, j, k-r-1+n] for m in 0:K-1 for n in 0:m)

#####
##### WENO raw weights
#####

@inline αx(i, j, k, weno, r, ϕ) = @inbounds weno.γ[r+1] / (βx(i, j, k, weno, r, ϕ) + weno.ε)^weno.η
@inline αy(i, j, k, weno, r, ϕ) = @inbounds weno.γ[r+1] / (βy(i, j, k, weno, r, ϕ) + weno.ε)^weno.η
@inline αz(i, j, k, weno, r, ϕ) = @inbounds weno.γ[r+1] / (βz(i, j, k, weno, r, ϕ) + weno.ε)^weno.η

#####
##### WENO normalized weights
#####

@inline function weno_weights_x(i, j, k, weno::WENO{K}, ϕ) where K
    α = [αx(i, j, k, weno, r, ϕ) for r in 0:K-1]
    return α ./ sum(α)
end

@inline function weno_weights_y(i, j, k, weno::WENO{K}, ϕ) where K
    α = [αy(i, j, k, weno, r, ϕ) for r in 0:K-1]
    return α ./ sum(α)
end

@inline function weno_weights_z(i, j, k, weno::WENO{K}, ϕ) where K
    α = [αz(i, j, k, weno, r, ϕ) for r in 0:K-1]
    return α ./ sum(α)
end

#####
##### WENO flux reconstruction
#####

@inline function weno_flux_x(i, j, k, weno::WENO{K}, ϕ) where K
    w = weno_weights_x(i, j, k, weno, ϕ)
    return sum(w[r+1] * px(i, j, k, weno, r, ϕ) for r in 0:K-1)
end

@inline function weno_flux_y(i, j, k, weno::WENO{K}, ϕ) where K
    w = weno_weights_y(i, j, k, weno, ϕ)
    return sum(w[r+1] * py(i, j, k, weno, r, ϕ) for r in 0:K-1)
end

@inline function weno_flux_z(i, j, k, weno::WENO{K}, ϕ) where K
    w = weno_weights_z(i, j, k, weno, ϕ)
    return sum(w[r+1] * pz(i, j, k, weno, r, ϕ) for r in 0:K-1)
end

#####
##### Momentum advection fluxes
#####

@inline momentum_flux_uu(i, j, k, grid, weno::WENO, u)    = ℑxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno_flux_x(i, j, k, weno, u)
@inline momentum_flux_uv(i, j, k, grid, weno::WENO, u, v) = ℑxᶠᵃᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno_flux_y(i, j, k, weno, u)
@inline momentum_flux_uw(i, j, k, grid, weno::WENO, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno_flux_z(i, j, k, weno, u)

@inline momentum_flux_vu(i, j, k, grid, weno::WENO, u, v) = ℑyᵃᶠᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno_flux_x(i, j, k, weno, v)
@inline momentum_flux_vv(i, j, k, grid, weno::WENO, v)    = ℑyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno_flux_y(i, j, k, weno, v)
@inline momentum_flux_vw(i, j, k, grid, weno::WENO, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno_flux_z(i, j, k, weno, v)

@inline momentum_flux_wu(i, j, k, grid, weno::WENO, u, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno_flux_x(i, j, k, weno, w)
@inline momentum_flux_wv(i, j, k, grid, weno::WENO, v, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno_flux_y(i, j, k, weno, w)
@inline momentum_flux_ww(i, j, k, grid, weno::WENO, w)    = ℑzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno_flux_z(i, j, k, weno, w)

#####
##### Advective tracer fluxes
#####

@inline advective_tracer_flux_x(i, j, k, grid, weno::WENO, u, c) = Ax_ψᵃᵃᶠ(i, j, k, grid, u) * weno_flux_x(i, j, k, weno, c)
@inline advective_tracer_flux_y(i, j, k, grid, weno::WENO, v, c) = Ay_ψᵃᵃᶠ(i, j, k, grid, v) * weno_flux_y(i, j, k, weno, c)
@inline advective_tracer_flux_z(i, j, k, grid, weno::WENO, w, c) = Az_ψᵃᵃᵃ(i, j, k, grid, w) * weno_flux_z(i, j, k, weno, c)

#####
##### Need to advect momentum like tracers
#####

@inline div_ũu(i, j, k, grid, advection::WENO, U) = div_uc(i, j, k, grid, advection, U, U.u)
@inline div_ũv(i, j, k, grid, advection::WENO, U) = div_uc(i, j, k, grid, advection, U, U.v)
@inline div_ũw(i, j, k, grid, advection::WENO, U) = div_uc(i, j, k, grid, advection, U, U.w)
