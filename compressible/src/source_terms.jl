using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.Coriolis

####
#### Element-wise forcing and right-hand-side calculations
####

@inline ρu_slow_source_term(i, j, k, grid, coriolis, closure, ρ, ρũ, K̃) =
    (- x_f_cross_U(i, j, k, grid, coriolis, ρũ)
     + ∂ⱼτ₁ⱼ(i, j, k, grid, closure, ρ, ρũ, K̃))


@inline ρv_slow_source_term(i, j, k, grid, coriolis, closure, ρ, ρũ, K̃) =
    (- y_f_cross_U(i, j, k, grid, coriolis, ρũ)
     + ∂ⱼτ₂ⱼ(i, j, k, grid, closure, ρ, ρũ, K̃))

@inline ρw_slow_source_term(i, j, k, grid, coriolis, closure, ρ, ρũ, K̃) =
    (- z_f_cross_U(i, j, k, grid, coriolis, ρũ)
     + ∂ⱼτ₃ⱼ(i, j, k, grid, closure, ρ, ρũ, K̃))

@inline ρc_slow_source_term(i, j, k, grid, closure, tracer_index, ρ, ρc, K̃) =
    ∂ⱼDᶜⱼ(i, j, k, grid, closure, tracer_index, ρ, ρc, K̃)

@inline ρt_slow_source_term(i, j, k, grid, closure, tvar::Energy, gases, gravity, ρ, ρũ, ρc̃, K̃) =
    ∂ⱼDᵖⱼ(i, j, k, grid, closure, 1, diagnose_p_over_ρ, tvar, gases, gravity, ρ, ρũ, ρc̃, K̃)

@inline function ρt_slow_source_term(i, j, k, grid, closure, tvar::Entropy, gases, gravity, ρ, ρũ, ρc̃, K̃)
    @inbounds begin
        Ṡ = 0.0
        for gas_index = 1:length(gases)
            tracer_index = gas_index + 1
            ρc = ρc̃[tracer_index]
            Ṡ += ∂ⱼtᶜDᶜⱼ(i, j, k, grid, closure, diagnose_ρs, tracer_index, tvar, gases, gravity, ρ, ρũ, ρc̃, ρc)
        end
        T = diagnose_temperature(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃)
        Ṡ += Q_dissipation(i, j, k, grid, closure, ρ, ρũ) / T
        return Ṡ
    end
end

@inline function ρu_fast_source_term(i, j, k, grid, tvar, gases, gravity, advection_scheme, ρ, ρũ, ρc̃, FU)
    @inbounds begin
        return (- div_ρuũ(i, j, k, grid, advection_scheme, ρ, ρũ)
                - ∂p∂x(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃)
                + FU[i, j, k])
    end
end

@inline function ρv_fast_source_term(i, j, k, grid, tvar, gases, gravity, advection_scheme, ρ, ρũ, ρc̃, FV)
    @inbounds begin
        return (- div_ρvũ(i, j, k, grid, advection_scheme, ρ, ρũ)
                - ∂p∂y(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃)
                + FV[i, j, k])
    end
end

@inline function ρw_fast_source_term(i, j, k, grid, tvar, gases, gravity, advection_scheme, ρ, ρũ, ρc̃, FW)
    @inbounds begin
        return (- div_ρwũ(i, j, k, grid, advection_scheme, ρ, ρũ)
                - ∂p∂z(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃)
                - gravity * ℑzᵃᵃᶠ(i, j, k, grid, ρ)
                + FW[i, j, k])
    end
end

@inline function ρc_fast_source_term(i, j, k, grid, advection_scheme, ρ, ρũ, ρc, FC)
    @inbounds begin
        return -div_ρUc(i, j, k, grid, advection_scheme, ρ, ρũ, ρc) + FC[i, j, k]
    end
end

@inline ρt_fast_source_term(i, j, k, grid::AbstractGrid{T}, tvar::Entropy, gases, gravity, ρ, ρũ, ρc̃) where T = zero(T)

@inline ρt_fast_source_term(i, j, k, grid, tvar::Energy, gases, gravity, ρ, ρũ, ρc̃) =
    -∂ⱼpuⱼ(i, j, k, grid, diagnose_pressure, tvar, gases, gravity, ρ, ρũ, ρc̃)
