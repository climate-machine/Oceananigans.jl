using Oceananigans.Advection
using Oceananigans.BuoyancyModels
using Oceananigans.Coriolis
using Oceananigans.Operators
using Oceananigans.StokesDrift
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ, ∇_dot_qᶜ

"""
Return the tendency for the horizontal velocity in the x-direction, or the east-west
direction, ``u``, at grid point `i, j, k`.

The tendency for ``u`` is called ``G_u`` and defined via

    ``∂_t u = G_u - ∂_x p_n``

where ∂_x p_n is the non-hydrostatic kinematic pressure gradient in the x-direction.

`coriolis`, `stokes_drift`, and `closure` are types encoding information about Coriolis
forces, surface waves, and the prescribed turbulence closure.

`background_fields` is a `NamedTuple` containing background velocity and tracer
`FunctionFields`.

The arguments `velocities`, `tracers`, and `diffusivities` are `NamedTuple`s with the three
velocity components, tracer fields, and precalculated diffusivities where applicable.
`forcings` is a named tuple of forcing functions. `hydrostatic_pressure` is the hydrostatic
pressure anomaly.

`clock` keeps track of `clock.time` and `clock.iteration`.
"""
@inline function u_velocity_tendency(i, j, k, grid,
                                     advection,
                                     coriolis,
                                     stokes_drift,
                                     closure,
                                     buoyancy,
                                     background_fields,
                                     velocities,
                                     velocities_immersed_bcs,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     hydrostatic_pressure,
                                     clock)

    return ( - div_𝐯u(i, j, k, grid, advection, velocities, velocities.u)
             - div_𝐯u(i, j, k, grid, advection, background_fields.velocities, velocities.u)
             - div_𝐯u(i, j, k, grid, advection, velocities, background_fields.velocities.u)
             - x_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂xᶠᶜᵃ(i, j, k, grid, hydrostatic_pressure)
             - ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, velocities_immersed_bcs, clock, velocities, diffusivities, tracers, buoyancy)
             + x_curl_Uˢ_cross_U(i, j, k, grid, stokes_drift, velocities, clock.time)
             + ∂t_uˢ(i, j, k, grid, stokes_drift, clock.time)
             + x_dot_g_b(i, j, k, grid, buoyancy, tracers)
             + forcings.u(i, j, k, grid, clock, merge(velocities, tracers)))
end

"""
Return the tendency for the horizontal velocity in the y-direction, or the north-south
direction, ``v``, at grid point `i, j, k`.

The tendency for ``v`` is called ``G_v`` and defined via

    ``∂_t v = G_v - ∂_y p_n``

where ∂_y p_n is the non-hydrostatic kinematic pressure gradient in the y-direction.

`coriolis`, `stokes_drift`, and `closure` are types encoding information about Coriolis
forces, surface waves, and the prescribed turbulence closure.

`background_fields` is a `NamedTuple` containing background velocity and tracer
`FunctionFields`.

The arguments `velocities`, `tracers`, and `diffusivities` are `NamedTuple`s with the three
velocity components, tracer fields, and precalculated diffusivities where applicable.
`forcings` is a named tuple of forcing functions. `hydrostatic_pressure` is the hydrostatic
pressure anomaly.

`clock` keeps track of `clock.time` and `clock.iteration`.
"""
@inline function v_velocity_tendency(i, j, k, grid,
                                     advection,
                                     coriolis,
                                     stokes_drift,
                                     closure,
                                     buoyancy,
                                     background_fields,
                                     velocities,
                                     velocities_immersed_bcs,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     hydrostatic_pressure,
                                     clock)

    return ( - div_𝐯v(i, j, k, grid, advection, velocities, velocities.v)
             - div_𝐯v(i, j, k, grid, advection, background_fields.velocities, velocities.v)
             - div_𝐯v(i, j, k, grid, advection, velocities, background_fields.velocities.v)
             - y_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂yᶜᶠᵃ(i, j, k, grid, hydrostatic_pressure)
             - ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, velocities_immersed_bcs, clock, velocities, diffusivities, tracers, buoyancy)
             + y_curl_Uˢ_cross_U(i, j, k, grid, stokes_drift, velocities, clock.time)
             + ∂t_vˢ(i, j, k, grid, stokes_drift, clock.time)
             + y_dot_g_b(i, j, k, grid, buoyancy, tracers)
             + forcings.v(i, j, k, grid, clock, merge(velocities, tracers)))
end

"""
Return the tendency for the vertical velocity ``w`` at grid point `i, j, k`.
The tendency for ``w`` is called ``G_w`` and defined via

    ``∂_t w = G_w - ∂_z p_n``

where ∂_z p_n is the non-hydrostatic kinematic pressure gradient in the z-direction.

`coriolis`, `stokes_drift`, and `closure` are types encoding information about Coriolis
forces, surface waves, and the prescribed turbulence closure.

`background_fields` is a `NamedTuple` containing background velocity and tracer
`FunctionFields`.

The arguments `velocities`, `tracers`, and `diffusivities` are `NamedTuple`s with the three
velocity components, tracer fields, and precalculated diffusivities where applicable.
`forcings` is a named tuple of forcing functions.

`clock` keeps track of `clock.time` and `clock.iteration`.
"""
@inline function w_velocity_tendency(i, j, k, grid,
                                     advection,
                                     coriolis,
                                     stokes_drift,
                                     closure,
                                     buoyancy,
                                     background_fields,
                                     velocities,
                                     velocities_immersed_bcs,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     hydrostatic_pressure,
                                     clock)

    return ( - div_𝐯w(i, j, k, grid, advection, velocities, velocities.w)
             - div_𝐯w(i, j, k, grid, advection, background_fields.velocities, velocities.w)
             - div_𝐯w(i, j, k, grid, advection, velocities, background_fields.velocities.w)
             - z_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂ⱼ_τ₃ⱼ(i, j, k, grid, closure, velocities_immersed_bcs, clock, velocities, diffusivities, tracers, buoyancy)
             + z_curl_Uˢ_cross_U(i, j, k, grid, stokes_drift, velocities, clock.time)
             + ∂t_wˢ(i, j, k, grid, stokes_drift, clock.time)
             + forcings.w(i, j, k, grid, clock, merge(velocities, tracers)))
end

"""
Return the tendency for a tracer field with index `tracer_index`
at grid point `i, j, k`.

The tendency is called ``G_c`` and defined via

    ``∂_t c = G_c``

where `c = C[tracer_index]`.

`closure` and `buoyancy` are types encoding information about the prescribed
turbulence closure and buoyancy model.

`background_fields` is a `NamedTuple` containing background velocity and tracer
`FunctionFields`.

The arguments `velocities`, `tracers`, and `diffusivities` are `NamedTuple`s with the three
velocity components, tracer fields, and precalculated diffusivities where applicable.
`forcings` is a named tuple of forcing functions.

`clock` keeps track of `clock.time` and `clock.iteration`.
"""
@inline function tracer_tendency(i, j, k, grid,
                                 val_tracer_index::Val{tracer_index},
                                 advection,
                                 tracer_immersed_bc,
                                 closure,
                                 buoyancy,
                                 background_fields,
                                 velocities,
                                 tracers,
                                 diffusivities,
                                 forcing,
                                 clock) where tracer_index

    @inbounds c = tracers[tracer_index]
    @inbounds background_fields_c = background_fields.tracers[tracer_index]

    return ( - div_Uc(i, j, k, grid, advection, velocities, c)
             - div_Uc(i, j, k, grid, advection, background_fields.velocities, c)
             - div_Uc(i, j, k, grid, advection, velocities, background_fields_c)
             - ∇_dot_qᶜ(i, j, k, grid, closure, tracer_immersed_bc, c, val_tracer_index, clock, diffusivities, tracers, buoyancy, velocities)
             + forcing(i, j, k, grid, clock, merge(velocities, tracers)))
end
