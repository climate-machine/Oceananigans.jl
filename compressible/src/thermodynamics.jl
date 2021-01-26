using JULES.Operators
using Oceananigans.Buoyancy: AbstractEquationOfState

#####
##### Universal gas constant and default reference states
#####

const atm = 101325.0         # 1 atmosphere in Pa
const R⁰ = 8.31446261815324  # Universal gas constant [J/mol/K]
const T₀ = 273.16            # Reference temperature [K]
const p₀ = 1atm              # Reference pressure [Pa]
const s₀ = 0.0               # Reference entropy [J/kg/K]
const u₀ = 0.0               # Reference internal energy [J/kg]

#####
##### Molar masses and degrees of freedom for common gases
#####

const M_N₂ = 28e-3           # Molar mass of diatomic nitrogen (kg/mol)
const dof_N₂ = 5.0           # DOF of diatomic nitrogren (nondim)
const M_O₂ = 32e-3           # Molar mass of diatomic nitrogen (kg/mol)
const dof_O₂ = 5.0           # DOF of diatomic nitrogren (nondim)

#####
##### Composition information for common atmospheres
#####

const η_N₂_earth = 0.7809    # Molar mixing ratio of N₂ on earth (nondim)
const η_O₂_earth = 0.2095    # Molar mixing ratio of O₂ on earth (nondim)

#####
##### Non-condensible ideal gas
#####

struct IdealGas{FT} <: AbstractEquationOfState
     R :: FT
    cₚ :: FT
    cᵥ :: FT
    T₀ :: FT
    p₀ :: FT
    ρ₀ :: FT
    s₀ :: FT
    u₀ :: FT
end

function EarthN₂O₂(FT=Float64; T₀=T₀, p₀=p₀, s₀=s₀, u₀=u₀)
    # Calculate abundance-weighted molar masses and DOFs
    M = (η_N₂_earth * M_N₂ + η_O₂_earth * M_O₂)/(η_N₂_earth + η_O₂_earth)
    dof = (η_N₂_earth * dof_N₂ + η_O₂_earth * dof_O₂)/(η_N₂_earth + η_O₂_earth)
    return IdealGas(FT, M, dof; T₀=T₀, p₀=p₀, s₀=s₀, u₀=u₀)
end

function IdealGas(FT, M, dof; T₀=T₀, p₀=p₀, s₀=s₀, u₀=u₀)
    R = R⁰/M
    cᵥ = R*dof/2
    cₚ = cᵥ + R
    ρ₀ = p₀/(R*T₀)
    return IdealGas{FT}(R, cₚ, cᵥ, T₀, p₀, ρ₀, s₀, u₀)
end

#####
##### Thermodynamic variables
#####

abstract type AbstractThermodynamicVariable end

struct Entropy <: AbstractThermodynamicVariable end
struct Energy  <: AbstractThermodynamicVariable end

#####
##### Thermodynamic state diagnostics
#####

@inline function diagnose_ρs(i, j, k, grid::AbstractGrid{FT}, tracer_index, tvar, gases, gravity, ρ, ρũ, ρc̃) where FT
    @inbounds begin
        T = diagnose_temperature(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃)
        ρᵅ = ρc̃[tracer_index][i, j, k]
        gas = gases[tracer_index-1]
        return ρᵅ > zero(FT) ? ρᵅ*(gas.s₀ + gas.cᵥ*log(T/gas.T₀) - gas.R*log(ρᵅ/gas.ρ₀)) : zero(FT)
    end
end

@inline function diagnose_temperature(i, j, k, grid::AbstractGrid{FT}, tvar::Entropy, gases, gravity, ρ, ρũ, ρc̃) where FT
    @inbounds begin
        numerator = ρc̃.ρs[i, j, k]
        denominator = zero(FT)
        for (gas_index, gas) in enumerate(gases)
            ρᵅ = ρc̃[gas_index+1][i, j, k]
            numerator += ρᵅ > 0 ? (ρᵅ*gas.R*log(ρᵅ/gas.ρ₀) - ρᵅ*gas.s₀) : zero(FT)
            denominator += ρᵅ*gas.cᵥ
        end
        return T₀*exp(numerator/denominator)
    end
end

@inline function diagnose_temperature(i, j, k, grid::AbstractGrid{FT}, tvar::Energy, gases, gravity, ρ, ρũ, ρc̃) where FT
    @inbounds begin
        numerator = ρc̃.ρe[i,j,k]
        denominator = zero(FT)
        KE = kinetic_energy(i, j, k, grid, ρ, ρũ)
        Φ = gravity * grid.zC[clamp(k, 1, grid.Nz)]
        for (gas_index, gas) in enumerate(gases)
            ρᵅ = ρc̃[gas_index+1][i,j,k]
            numerator += -ρᵅ*(gas.u₀ + Φ + KE - gas.cᵥ*gas.T₀)
            denominator += ρᵅ*gas.cᵥ
        end
        return numerator/denominator
    end
end

@inline function diagnose_pressure(i, j, k, grid::AbstractGrid{FT}, tvar, gases, gravity, ρ, ρũ, ρc̃) where FT
    @inbounds begin
        T = diagnose_temperature(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃)
        p = zero(FT)
        for gas_index in 1:length(gases)
            R = gases[gas_index].R
            ρᵅ = ρc̃[gas_index+1][i, j, k]
            p += ρᵅ*R*T
        end
        return p
    end
end

@inline function diagnose_density(i, j, k, grid::AbstractGrid{FT}, gases, ρc̃) where FT
    @inbounds begin
        ρ = zero(FT)
        for gas_index in 1:length(gases)
            ρ += ρc̃[gas_index+1][i, j, k]
        end
        return ρ
    end
end

@inline function diagnose_p_over_ρ(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃)
    @inbounds begin
        p = diagnose_pressure(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃)
        return p / ρ[i, j, k]
    end
end
