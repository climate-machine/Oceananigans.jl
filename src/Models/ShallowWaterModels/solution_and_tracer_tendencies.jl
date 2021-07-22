using Oceananigans.Advection
using Oceananigans.Coriolis
using Oceananigans.Operators
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ

@inline squared(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

@inline half_g_h²(i, j, k, grid, h, g) = @inbounds 1/2 * g * h[i, j, k]^2

@inline x_pressure_gradient(i, j, k, grid, h, gravitational_acceleration) = ∂xᶠᶜᵃ(i, j, k, grid, half_g_h², h, gravitational_acceleration)
@inline y_pressure_gradient(i, j, k, grid, h, gravitational_acceleration) = ∂yᶜᶠᵃ(i, j, k, grid, half_g_h², h, gravitational_acceleration)

"""
Compute the tendency for the x-directional transport, uh
"""
@inline function uh_solution_tendency(i, j, k, grid,
                                      gravitational_acceleration,
                                      advection,
                                      coriolis,
                                      closure,
                                      bathymetry,
                                      solution,
                                      tracers,
                                      diffusivities,
                                      forcings,
                                      clock)

    g = gravitational_acceleration

    return ( - div_hUu(i, j, k, grid, advection, solution)
             - x_pressure_gradient(i, j, k, grid, solution.h, gravitational_acceleration)
             - x_f_cross_U(i, j, k, grid, coriolis, solution)
             + forcings.uh(i, j, k, grid, clock, merge(solution, tracers)))
end

"""
Compute the tendency for the y-directional transport, vh.
"""
@inline function vh_solution_tendency(i, j, k, grid,
                                      gravitational_acceleration,
                                      advection,
                                      coriolis,
                                      closure,
                                      bathymetry,
                                      solution,
                                      tracers,
                                      diffusivities,
                                      forcings,
                                      clock)

     g = gravitational_acceleration

    return ( - div_hUv(i, j, k, grid, advection, solution)
             - y_pressure_gradient(i, j, k, grid, solution.h, gravitational_acceleration)
             - y_f_cross_U(i, j, k, grid, coriolis, solution)
             + forcings.vh(i, j, k, grid, clock, merge(solution, tracers)))
end

"""
Compute the tendency for the height, h.
"""
@inline function h_solution_tendency(i, j, k, grid,
                                     gravitational_acceleration,
                                     coriolis,
                                     closure,
                                     bathymetry,
                                     solution,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     clock)

    return ( - div_Uh(i, j, k, grid, solution)
             + forcings.h(i, j, k, grid, clock, merge(solution, tracers)))
end

@inline function tracer_tendency(i, j, k, grid,
                                 val_tracer_index::Val{tracer_index},
                                 advection,
                                 closure,
                                 solution,
                                 tracers,
                                 diffusivities,
                                 forcing,
                                 clock) where tracer_index

    @inbounds c = tracers[tracer_index]

    return ( -  div_Uc(i, j, k, grid, advection, solution, c) 
             + c_div_U(i, j, k, grid, solution, c)         
             - ∇_dot_qᶜ(i, j, k, grid, closure, c, val_tracer_index, clock, diffusivities, tracers, nothing)
             + forcing(i, j, k, grid, clock, merge(solution, tracers)) 
            )
end
