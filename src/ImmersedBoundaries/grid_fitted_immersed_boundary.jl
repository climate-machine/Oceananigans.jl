using Oceananigans.Advection: AbstractAdvectionScheme
using Oceananigans.BoundaryConditions: BoundaryCondition
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑxᶜᵃᵃ, ℑyᵃᶠᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ 
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, AbstractTimeDiscretization

const ATC = AbstractTurbulenceClosure
const ATD = AbstractTimeDiscretization

struct RasterDepthMask end

struct GridFittedBoundary{M, S} <: AbstractImmersedBoundary
    mask :: S
    mask_type :: M
end

GridFittedBoundary(mask; mask_type=nothing) = GridFittedBoundary(mask, mask_type)

@inline is_immersed(i, j, k, underlying_grid, ib::GridFittedBoundary) = ib.mask(node(c, c, c, i, j, k, underlying_grid)...)
@inline is_immersed(i, j, k, underlying_grid, ib::GridFittedBoundary{<:RasterDepthMask}) = ib.mask(i, j)

const IBG = ImmersedBoundaryGrid

const c = Center()
const f = Face()

#####
##### GridFittedImmersedBoundaryGrid
#####

const GFIBG = ImmersedBoundaryGrid{FT, TX, TY, TZ, G, <:GridFittedBoundary} where {FT, TX, TY, TZ, G}

@inline solid_cell(i, j, k, ibg) = is_immersed(i, j, k, ibg.grid, ibg.immersed_boundary)

@inline solid_node(LX, LY, LZ, i, j, k, ibg) = solid_cell(i, j, k, ibg) # fallback (for Center or Nothing LX, LY, LZ)

@inline solid_node(::Face, LX, LY, i, j, k, ibg) = solid_cell(i, j, k, ibg) || solid_cell(i-1, j, k, ibg)
@inline solid_node(LX, ::Face, LZ, i, j, k, ibg) = solid_cell(i, j, k, ibg) || solid_cell(i, j-1, k, ibg)
@inline solid_node(LX, LY, ::Face, i, j, k, ibg) = solid_cell(i, j, k, ibg) || solid_cell(i, j, k-1, ibg)

@inline solid_node(::Face, ::Face, LZ, i, j, k, ibg) = solid_node(c, f, c, i, j, k, ibg) || solid_node(c, f, c, i-1, j, k, ibg)
@inline solid_node(::Face, LY, ::Face, i, j, k, ibg) = solid_node(c, c, f, i, j, k, ibg) || solid_node(c, c, f, i-1, j, k, ibg)
@inline solid_node(LX, ::Face, ::Face, i, j, k, ibg) = solid_node(c, f, c, i, j, k, ibg) || solid_node(c, f, c, i, j, k-1, ibg)

@inline solid_node(::Face, ::Face, ::Face, i, j, k, ibg) = solid_node(c, f, f, i, j, k, ibg) || solid_node(c, f, f, i-1, j, k, ibg)

@inline conditional_flux_ccc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_node(c, c, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_flux_ffc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_node(f, f, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_flux_fcf(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_node(f, c, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_flux_cff(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_node(c, f, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))

@inline conditional_flux_fcc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_node(f, c, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_flux_cfc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_node(c, f, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_flux_ccf(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_node(c, c, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))

#####
##### Advective fluxes
#####

const VelocitiesNamedTuple = NamedTuple{(:u, :v, :w)}

# ccc, ffc, fcf
 @inline _viscous_flux_ux(i, j, k, ibg::GFIBG, disc, closure, immersed_bcs::VelocitiesNamedTuple, args...) = conditional_flux_ccc(i, j, k, ibg, viscous_flux_ux, disc, closure, args...)
 @inline _viscous_flux_uy(i, j, k, ibg::GFIBG, disc, closure, immersed_bcs::VelocitiesNamedTuple, args...) = conditional_flux_ffc(i, j, k, ibg, viscous_flux_uy, disc, closure, args...)
 @inline _viscous_flux_uz(i, j, k, ibg::GFIBG, disc, closure, immersed_bcs::VelocitiesNamedTuple, args...) = conditional_flux_fcf(i, j, k, ibg, viscous_flux_uz, disc, closure, args...)
 
 # ffc, ccc, cff
 @inline _viscous_flux_vx(i, j, k, ibg::GFIBG, disc, closure, immersed_bcs::VelocitiesNamedTuple, args...) = conditional_flux_ffc(i, j, k, ibg, viscous_flux_vx, disc, closure, args...)
 @inline _viscous_flux_vy(i, j, k, ibg::GFIBG, disc, closure, immersed_bcs::VelocitiesNamedTuple, args...) = conditional_flux_ccc(i, j, k, ibg, viscous_flux_vy, disc, closure, args...)
 @inline _viscous_flux_vz(i, j, k, ibg::GFIBG, disc, closure, immersed_bcs::VelocitiesNamedTuple, args...) = conditional_flux_cff(i, j, k, ibg, viscous_flux_vz, disc, closure, args...)
 
 # fcf, cff, ccc
 @inline _viscous_flux_wx(i, j, k, ibg::GFIBG, disc, closure, immersed_bcs::VelocitiesNamedTuple, args...) = conditional_flux_fcf(i, j, k, ibg, viscous_flux_wx, disc, closure, args...)
 @inline _viscous_flux_wy(i, j, k, ibg::GFIBG, disc, closure, immersed_bcs::VelocitiesNamedTuple, args...) = conditional_flux_cff(i, j, k, ibg, viscous_flux_wy, disc, closure, args...)
 @inline _viscous_flux_wz(i, j, k, ibg::GFIBG, disc, closure, immersed_bcs::VelocitiesNamedTuple, args...) = conditional_flux_ccc(i, j, k, ibg, viscous_flux_wz, disc, closure, args...)

# fcc, cfc, ccf
const BC = BoundaryCondition
@inline _diffusive_flux_x(i, j, k, ibg::GFIBG, disc::ATD, closure, immersed_bc::BC, c::AbstractArray, args...) = conditional_flux_fcc(i, j, k, ibg, diffusive_flux_x, disc, closure, c, args...)
@inline _diffusive_flux_y(i, j, k, ibg::GFIBG, disc::ATD, closure, immersed_bc::BC, c::AbstractArray, args...) = conditional_flux_cfc(i, j, k, ibg, diffusive_flux_y, disc, closure, c, args...)
@inline _diffusive_flux_z(i, j, k, ibg::GFIBG, disc::ATD, closure, immersed_bc::BC, c::AbstractArray, args...) = conditional_flux_ccf(i, j, k, ibg, diffusive_flux_z, disc, closure, c, args...)

#####
##### Advective fluxes
#####

# dx(uu), dy(vu), dz(wu)
# ccc,    ffc,    fcf
@inline _advective_momentum_flux_Uu(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, advective_momentum_flux_Uu, args...)
@inline _advective_momentum_flux_Vu(i, j, k, ibg::GFIBG, args...) = conditional_flux_ffc(i, j, k, ibg, advective_momentum_flux_Vu, args...)
@inline _advective_momentum_flux_Wu(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcf(i, j, k, ibg, advective_momentum_flux_Wu, args...)

# dx(uv), dy(vv), dz(wv)
# ffc,    ccc,    cff
@inline _advective_momentum_flux_Uv(i, j, k, ibg::GFIBG, args...) = conditional_flux_ffc(i, j, k, ibg, advective_momentum_flux_Uv, args...)
@inline _advective_momentum_flux_Vv(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, advective_momentum_flux_Vv, args...)
@inline _advective_momentum_flux_Wv(i, j, k, ibg::GFIBG, args...) = conditional_flux_cff(i, j, k, ibg, advective_momentum_flux_Wv, args...)

# dx(uw), dy(vw), dz(ww)
# fcf,    cff,    ccc
@inline _advective_momentum_flux_Uw(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcf(i, j, k, ibg, advective_momentum_flux_Uw, args...)
@inline _advective_momentum_flux_Vw(i, j, k, ibg::GFIBG, args...) = conditional_flux_cff(i, j, k, ibg, advective_momentum_flux_Vw, args...)
@inline _advective_momentum_flux_Ww(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, advective_momentum_flux_Ww, args...)

   @inline _advective_tracer_flux_x(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcc(i, j, k, ibg, advective_tracer_flux_x, args...)
   @inline _advective_tracer_flux_y(i, j, k, ibg::GFIBG, args...) = conditional_flux_cfc(i, j, k, ibg, advective_tracer_flux_y, args...)
   @inline _advective_tracer_flux_z(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccf(i, j, k, ibg, advective_tracer_flux_z, args...)

#####
##### "Boundary-aware" interpolation
#####
##### Don't interpolate dead cells.
#####

@inline near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
@inline near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
@inline near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false

@inline near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{1}) = solid_cell(i - 1, j, k, ibg) || solid_cell(i, j, k, ibg) || solid_cell(i + 1, j, k, ibg)
@inline near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{1}) = solid_cell(i, j - 1, k, ibg) || solid_cell(i, j, k, ibg) || solid_cell(i, j + 1, k, ibg)
@inline near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{1}) = solid_cell(i, j, k - 1, ibg) || solid_cell(i, j, k, ibg) || solid_cell(i, j, k + 1, ibg)

@inline near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{2}) = solid_cell(i - 2, j, k, ibg) || solid_cell(i - 1, j, k, ibg) || solid_cell(i, j, k, ibg) || solid_cell(i + 1, j, k, ibg) || solid_cell(i + 2, j, k, ibg)
@inline near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{2}) = solid_cell(i, j - 2, k, ibg) || solid_cell(i, j - 1, k, ibg) || solid_cell(i, j, k, ibg) || solid_cell(i, j + 1, k, ibg) || solid_cell(i, j + 2, k, ibg)
@inline near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{2}) = solid_cell(i, j, k - 2, ibg) || solid_cell(i, j, k - 1, ibg) || solid_cell(i, j, k, ibg) || solid_cell(i, j, k + 1, ibg) || solid_cell(i, j, k + 2, ibg)

# Takes forever to compile, but works.
# @inline near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{buffer}) where buffer = any(ntuple(δ -> solid_cell(i - buffer - 1 + δ, j, k, ibg), Val(2buffer + 1)))
# @inline near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{buffer}) where buffer = any(ntuple(δ -> solid_cell(i, j - buffer - 1 + δ, k, ibg), Val(2buffer + 1)))
# @inline near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{buffer}) where buffer = any(ntuple(δ -> solid_cell(i, j, k - buffer - 1 + δ, ibg), Val(2buffer + 1)))

for bias in (:symmetric, :left_biased, :right_biased)
    for (d, ξ) in enumerate((:x, :y, :z))

        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ)
            code[d] = loc
            second_order_interp = Symbol(:ℑ, ξ, code...)
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt_interp = Symbol(:_, interp)

            near_boundary = Symbol(:near_, ξ, :_boundary)

            # Conditional high-order interpolation in Bounded directions
            @eval begin
                import Oceananigans.Advection: $alt_interp
                using Oceananigans.Advection: $interp

                @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme, ψ) =
                    ifelse($near_boundary(i, j, k, ibg, scheme),
                           $second_order_interp(i, j, k, ibg.grid, ψ),
                           $interp(i, j, k, ibg.grid, scheme, ψ))
            end
        end
    end
end

#####
##### Masking for GridFittedBoundary
#####

@inline function scalar_mask(i, j, k, grid, ::GridFittedBoundary, LX, LY, LZ, value, field)
    return @inbounds ifelse(solid_node(LX, LY, LZ, i, j, k, grid),
                            value,
                            field[i, j, k])
end

mask_immersed_velocities!(U, arch, grid::GFIBG) = Tuple(mask_immersed_field!(q) for q in U)
