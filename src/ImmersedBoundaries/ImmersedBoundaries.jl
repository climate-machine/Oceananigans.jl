module ImmersedBoundaries

using Adapt

using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Fields
using Oceananigans.Utils
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, time_discretization

using Oceananigans.TurbulenceClosures:
    viscous_flux_ux,
    viscous_flux_uy,
    viscous_flux_uz,
    viscous_flux_vx,
    viscous_flux_vy,
    viscous_flux_vz,
    viscous_flux_wx,
    viscous_flux_wy,
    viscous_flux_wz,
    diffusive_flux_x,
    diffusive_flux_y,
    diffusive_flux_z

using Oceananigans.Advection:
    advective_momentum_flux_Uu,
    advective_momentum_flux_Uv,
    advective_momentum_flux_Uw,
    advective_momentum_flux_Vu,
    advective_momentum_flux_Vv,
    advective_momentum_flux_Vw,
    advective_momentum_flux_Wu,
    advective_momentum_flux_Wv,
    advective_momentum_flux_Ww,
    advective_tracer_flux_x,
    advective_tracer_flux_y,
    advective_tracer_flux_z

import Oceananigans.Utils: cell_advection_timescale
import Oceananigans.Grids: with_halo
import Oceananigans.Coriolis: φᶠᶠᵃ
import Oceananigans.Grids: with_halo, xnode, ynode, znode, all_x_nodes, all_y_nodes, all_z_nodes

import Oceananigans.Advection:
    _advective_momentum_flux_Uu,
    _advective_momentum_flux_Uv,
    _advective_momentum_flux_Uw,
    _advective_momentum_flux_Vu,
    _advective_momentum_flux_Vv,
    _advective_momentum_flux_Vw,
    _advective_momentum_flux_Wu,
    _advective_momentum_flux_Wv,
    _advective_momentum_flux_Ww,
    _advective_tracer_flux_x,
    _advective_tracer_flux_y,
    _advective_tracer_flux_z

import Oceananigans.TurbulenceClosures:
    _viscous_flux_ux,
    _viscous_flux_uy,
    _viscous_flux_uz,
    _viscous_flux_vx,
    _viscous_flux_vy,
    _viscous_flux_vz,
    _viscous_flux_wx,
    _viscous_flux_wy,
    _viscous_flux_wz,
    _diffusive_flux_x,
    _diffusive_flux_y,
    _diffusive_flux_z,
    κᶠᶜᶜ,
    κᶜᶠᶜ,
    κᶜᶜᶠ,
    νᶜᶜᶜ,
    νᶠᶠᶜ,
    νᶜᶠᶠ,
    νᶠᶜᶠ

export AbstractImmersedBoundary

abstract type AbstractImmersedBoundary end

struct ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I} <: AbstractGrid{FT, TX, TY, TZ}
    grid :: G
    immersed_boundary :: I

    function ImmersedBoundaryGrid{TX, TY, TZ}(grid::G, ib::I) where {TX, TY, TZ, G <: AbstractUnderlyingGrid, I}
        FT = eltype(grid)
        return new{FT, TX, TY, TZ, G, I}(grid, ib)
    end
end

function ImmersedBoundaryGrid(grid, ib)
    @warn "ImmersedBoundaryGrid is unvalidated and may produce incorrect results. " *
              "Don't hesitate to help validate ImmersedBoundaryGrid by reporting any bugs " *
              "or unexpected behavior to https://github.com/CliMA/Oceananigans.jl/issues"
    
    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib)
end

const IBG = ImmersedBoundaryGrid

@inline Base.getproperty(ibg::IBG, property::Symbol) = get_ibg_property(ibg, Val(property))
@inline get_ibg_property(ibg::IBG, ::Val{property}) where property = getfield(getfield(ibg, :grid), property)
@inline get_ibg_property(ibg::IBG, ::Val{:immersed_boundary}) = getfield(ibg, :immersed_boundary)
@inline get_ibg_property(ibg::IBG, ::Val{:grid}) = getfield(ibg, :grid)

Adapt.adapt_structure(to, ibg::IBG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    ImmersedBoundaryGrid{TX, TY, TZ}(adapt(to, ibg.grid), adapt(to, ibg.immersed_boundary))

with_halo(halo, ibg::ImmersedBoundaryGrid) = ImmersedBoundaryGrid(with_halo(halo, ibg.grid), ibg.immersed_boundary)


@inline cell_advection_timescale(u, v, w, ibg::ImmersedBoundaryGrid) = cell_advection_timescale(u, v, w, ibg.grid)
@inline φᶠᶠᵃ(i, j, k, ibg::ImmersedBoundaryGrid) = φᶠᶠᵃ(i, j, k, ibg.grid)

@inline xnode(LX, LY, LZ, i, j, k, ibg::ImmersedBoundaryGrid) = xnode(LX, LY, LZ, i, j, k, ibg.grid)
@inline ynode(LX, LY, LZ, i, j, k, ibg::ImmersedBoundaryGrid) = ynode(LX, LY, LZ, i, j, k, ibg.grid)
@inline znode(LX, LY, LZ, i, j, k, ibg::ImmersedBoundaryGrid) = znode(LX, LY, LZ, i, j, k, ibg.grid)

all_x_nodes(loc, ibg::ImmersedBoundaryGrid) = all_x_nodes(loc, ibg.grid)
all_y_nodes(loc, ibg::ImmersedBoundaryGrid) = all_y_nodes(loc, ibg.grid)
all_z_nodes(loc, ibg::ImmersedBoundaryGrid) = all_z_nodes(loc, ibg.grid)

include("immersed_grid_metrics.jl")
include("grid_fitted_immersed_boundary.jl")
include("mask_immersed_field.jl")

#####
##### Diffusivities (for VerticallyImplicitTimeDiscretization)
#####

for (locate_coeff, loc) in ((:κᶠᶜᶜ, (f, c, c)),
                            (:κᶜᶠᶜ, (c, f, c)),
                            (:κᶜᶜᶠ, (c, c, f)),
                            (:νᶜᶜᶜ, (c, c, c)),
                            (:νᶠᶠᶜ, (f, f, c)),
                            (:νᶠᶜᶠ, (f, c, f)),
                            (:νᶜᶠᶠ, (c, f, f)))

    @eval begin
        @inline $locate_coeff(i, j, k, ibg::IBG{FT}, coeff) where FT =
            ifelse(solid_cell(loc..., i, j, k, ibg), $locate_coeff(i, j, k, ibg.grid, coeff), zero(FT))
    end
end

end # module
