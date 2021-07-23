#####
##### Rectilinear derivative operators
#####

@inline ∂xᶜᵃᵃ(i, j, k, grid::ARG, u) = δxᶜᵃᵃ(i, j, k, grid, u) / Δx(i, j, k, grid)
@inline ∂xᶠᵃᵃ(i, j, k, grid::ARG, c) = δxᶠᵃᵃ(i, j, k, grid, c) / Δx(i, j, k, grid)

@inline ∂yᵃᶜᵃ(i, j, k, grid::ARG, v) = δyᵃᶜᵃ(i, j, k, grid, v) / Δy(i, j, k, grid)
@inline ∂yᵃᶠᵃ(i, j, k, grid::ARG, c) = δyᵃᶠᵃ(i, j, k, grid, c) / Δy(i, j, k, grid)

@inline ∂zᵃᵃᶜ(i, j, k, grid, w) = δzᵃᵃᶜ(i, j, k, grid, w) / Δzᵃᵃᶜ(i, j, k, grid)
@inline ∂zᵃᵃᶠ(i, j, k, grid, c) = δzᵃᵃᶠ(i, j, k, grid, c) / Δzᵃᵃᶠ(i, j, k, grid)


@inline ∂xᶜᵃᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = δxᶜᵃᵃ(i, j, k, grid, f, args...) / Δx(i, j, k, grid)
@inline ∂xᶠᵃᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = δxᶠᵃᵃ(i, j, k, grid, f, args...) / Δx(i, j, k, grid)

@inline ∂yᵃᶜᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = δyᵃᶜᵃ(i, j, k, grid, f, args...) / Δy(i, j, k, grid)
@inline ∂yᵃᶠᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = δyᵃᶠᵃ(i, j, k, grid, f, args...) / Δy(i, j, k, grid)

@inline ∂zᵃᵃᶜ(i, j, k, grid, f::F, args...) where F<:Function = δzᵃᵃᶜ(i, j, k, grid, f, args...) / Δzᵃᵃᶜ(i, j, k, grid)
@inline ∂zᵃᵃᶠ(i, j, k, grid, f::F, args...) where F<:Function = δzᵃᵃᶠ(i, j, k, grid, f, args...) / Δzᵃᵃᶠ(i, j, k, grid)

#####
##### Operators of the form A*δ(q) where A is an area and q is some quantity.
#####

@inline Ax_∂xᶠᶜᶜ(i, j, k, grid, c) = Axᶠᶜᶜ(i, j, k, grid) * ∂xᶠᶜᵃ(i, j, k, grid, c)
@inline Ax_∂xᶜᶜᶜ(i, j, k, grid, u) = Axᶜᶜᶜ(i, j, k, grid) * ∂xᶜᶜᵃ(i, j, k, grid, u)
@inline Ax_∂xᶠᶠᶜ(i, j, k, grid, v) = Axᶠᶠᶜ(i, j, k, grid) * ∂xᶠᶠᵃ(i, j, k, grid, v)

@inline Ay_∂yᶜᶠᶜ(i, j, k, grid, c) = Ayᶜᶠᶜ(i, j, k, grid) * ∂yᶜᶠᵃ(i, j, k, grid, c)
@inline Ay_∂yᶠᶠᶜ(i, j, k, grid, u) = Ayᶠᶠᶜ(i, j, k, grid) * ∂yᶠᶠᵃ(i, j, k, grid, u)
@inline Ay_∂yᶜᶜᶜ(i, j, k, grid, v) = Ayᶜᶜᶜ(i, j, k, grid) * ∂yᶜᶜᵃ(i, j, k, grid, v)

@inline Az_∂zᶜᶜᶠ(i, j, k, grid, c) = Azᶜᶜᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, c)

#####
##### Second derivatives
#####

@inline ∂²xᶜᵃᵃ(i, j, k, grid::ARG, c) = ∂xᶜᵃᵃ(i, j, k, grid, ∂xᶠᵃᵃ, c)
@inline ∂²xᶠᵃᵃ(i, j, k, grid::ARG, u) = ∂xᶠᵃᵃ(i, j, k, grid, ∂xᶜᵃᵃ, u)

@inline ∂²yᵃᶜᵃ(i, j, k, grid::ARG, c) = ∂yᵃᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, c)
@inline ∂²yᵃᶠᵃ(i, j, k, grid::ARG, v) = ∂yᵃᶠᵃ(i, j, k, grid, ∂yᵃᶜᵃ, v)

@inline ∂²zᵃᵃᶜ(i, j, k, grid::Union{ARG, AHCG}, c) = ∂zᵃᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, c)
@inline ∂²zᵃᵃᶠ(i, j, k, grid::Union{ARG, AHCG}, w) = ∂zᵃᵃᶠ(i, j, k, grid, ∂zᵃᵃᶜ, w)


@inline ∂²xᶜᵃᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = ∂xᶜᵃᵃ(i, j, k, grid, ∂xᶠᵃᵃ, f, args...)
@inline ∂²xᶠᵃᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = ∂xᶠᵃᵃ(i, j, k, grid, ∂xᶜᵃᵃ, f, args...)

@inline ∂²yᵃᶜᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = ∂yᵃᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, f, args...)
@inline ∂²yᵃᶠᵃ(i, j, k, grid::ARG, f::F, args...) where F<:Function = ∂yᵃᶠᵃ(i, j, k, grid, ∂yᵃᶜᵃ, f, args...)

@inline ∂²zᵃᵃᶜ(i, j, k, grid, f::F, args...) where F<:Function = ∂zᵃᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, f, args...)
@inline ∂²zᵃᵃᶠ(i, j, k, grid, f::F, args...) where F<:Function = ∂zᵃᵃᶠ(i, j, k, grid, ∂zᵃᵃᶜ, f, args...)

#####
##### Third derivatives
#####

@inline ∂³zᵃᵃᶜ(i, j, k, grid, w) = ∂zᵃᵃᶜ(i, j, k, grid, ∂²zᵃᵃᶠ, w)
@inline ∂³zᵃᵃᶠ(i, j, k, grid, c) = ∂zᵃᵃᶠ(i, j, k, grid, ∂²zᵃᵃᶜ, c)

#####
##### Fourth derivatives
#####

@inline ∂⁴xᶜᵃᵃ(i, j, k, grid::ARG, c, args...) = ∂²xᶜᵃᵃ(i, j, k, grid, ∂²xᶜᵃᵃ, c, args...)
@inline ∂⁴xᶠᵃᵃ(i, j, k, grid::ARG, u, args...) = ∂²xᶠᵃᵃ(i, j, k, grid, ∂²xᶠᵃᵃ, u, args...)

@inline ∂⁴yᵃᶜᵃ(i, j, k, grid::ARG, c, args...) = ∂²yᵃᶜᵃ(i, j, k, grid, ∂²yᵃᶜᵃ, c, args...)
@inline ∂⁴yᵃᶠᵃ(i, j, k, grid::ARG, v, args...) = ∂²yᵃᶠᵃ(i, j, k, grid, ∂²yᵃᶠᵃ, v, args...)

@inline ∂⁴zᵃᵃᶜ(i, j, k, grid, c, args...) = ∂²zᵃᵃᶜ(i, j, k, grid, ∂²zᵃᵃᶜ, c, args...)
@inline ∂⁴zᵃᵃᶠ(i, j, k, grid, w, args...) = ∂²zᵃᵃᶠ(i, j, k, grid, ∂²zᵃᵃᶠ, w, args...)

#####
##### Horizontally curvilinear derivative operators
#####

@inline ∂xᶜᶜᵃ(i, j, k, grid, u, args...) = δxᶜᵃᵃ(i, j, k, grid, u, args...) / Δxᶜᶜᵃ(i, j, k, grid)
@inline ∂xᶜᶠᵃ(i, j, k, grid, ζ, args...) = δxᶜᵃᵃ(i, j, k, grid, ζ, args...) / Δxᶜᶠᵃ(i, j, k, grid)

@inline ∂xᶠᶠᵃ(i, j, k, grid, v, args...) = δxᶠᵃᵃ(i, j, k, grid, v, args...) / Δxᶠᶠᵃ(i, j, k, grid)
@inline ∂xᶠᶜᵃ(i, j, k, grid, c, args...) = δxᶠᵃᵃ(i, j, k, grid, c, args...) / Δxᶠᶜᵃ(i, j, k, grid)

@inline ∂yᶜᶜᵃ(i, j, k, grid, v, args...) = δyᵃᶜᵃ(i, j, k, grid, v, args...) / Δyᶜᶜᵃ(i, j, k, grid)
@inline ∂yᶠᶜᵃ(i, j, k, grid, ζ, args...) = δyᵃᶜᵃ(i, j, k, grid, ζ, args...) / Δyᶠᶜᵃ(i, j, k, grid)

@inline ∂yᶠᶠᵃ(i, j, k, grid, u, args...) = δyᵃᶠᵃ(i, j, k, grid, u, args...) / Δyᶠᶠᵃ(i, j, k, grid)
@inline ∂yᶜᶠᵃ(i, j, k, grid, c, args...) = δyᵃᶠᵃ(i, j, k, grid, c, args...) / Δyᶜᶠᵃ(i, j, k, grid)

