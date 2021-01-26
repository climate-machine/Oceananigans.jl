using JULES.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶠ

####
#### Pressure gradient ∇p terms for entropy S = ρs
####

@inline ∂p∂x(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃) = ∂xᶠᵃᵃ(i, j, k, grid, diagnose_pressure, tvar, gases, gravity, ρ, ρũ, ρc̃)
@inline ∂p∂y(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃) = ∂yᵃᶠᵃ(i, j, k, grid, diagnose_pressure, tvar, gases, gravity, ρ, ρũ, ρc̃)
@inline ∂p∂z(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃) = ∂zᵃᵃᶠ(i, j, k, grid, diagnose_pressure, tvar, gases, gravity, ρ, ρũ, ρc̃)
