using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.Architectures
using Oceananigans.Fields: ReducedField

import Oceananigans.Solvers: solve!

struct PCGImplicitFreeSurfaceSolver{V, S, R}
    vertically_integrated_lateral_areas :: V
    preconditioned_conjugate_gradient_solver :: S
    right_hand_side :: R
end

"""
    PCGImplicitFreeSurfaceSolver(arch::AbstractArchitecture, grid, settings)

Return a solver based on a preconditioned conjugate gradient method for the elliptic equation
    
```math
[∇ ⋅ H ∇ - Az / (g Δt²)] ηⁿ⁺¹ = (∇ʰ ⋅ Q★ - Az ηⁿ / Δt) / (g Δt) 
```

representing an implicit time discretization of the linear free surface evolution equation
for a fluid with variable depth `H`, horizontal areas `Az`, barotropic volume flux `Q★`, time
step `Δt`, gravitational acceleration `g`, and free surface at time-step `n` `ηⁿ`.
"""
function PCGImplicitFreeSurfaceSolver(arch::AbstractArchitecture, grid, settings)
    # Initialize vertically integrated lateral face areas
    ∫ᶻ_Axᶠᶜᶜ = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    ∫ᶻ_Ayᶜᶠᶜ = ReducedField(Center, Face, Nothing, arch, grid; dims=3)

    vertically_integrated_lateral_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    compute_vertically_integrated_lateral_areas!(vertically_integrated_lateral_areas, grid, arch)

    right_hand_side = ReducedField(Center, Center, Nothing, arch, grid; dims=3)

    # Set maximum iterations to Nx * Ny if not set
    settings = Dict{Symbol, Any}(settings)
    maximum_iterations = get(settings, :maximum_iterations, grid.Nx * grid.Ny)
    settings[:maximum_iterations] = maximum_iterations

    solver = PreconditionedConjugateGradientSolver(implicit_free_surface_linear_operation!;
                                                   template_field = right_hand_side,
                                                   settings...)

    return PCGImplicitFreeSurfaceSolver(vertically_integrated_lateral_areas, solver, right_hand_side)
end

build_implicit_step_solver(::Val{:PreconditionedConjugateGradient}, arch, grid, settings) =
    PCGImplicitFreeSurfaceSolver(arch, grid, settings)

#####
##### Solve...
#####

function solve!(η, implicit_free_surface_solver::PCGImplicitFreeSurfaceSolver, rhs, g, Δt)
    #=
    # Somehow take an explicit step first to produce an improved initial guess
    # for η for the iterative solver.
    event = explicit_ab2_step_free_surface!(free_surface, model, Δt, χ)
    wait(device(model.architecture), event)
    =#

    ∫ᶻA = implicit_free_surface_solver.vertically_integrated_lateral_areas
    solver = implicit_free_surface_solver.preconditioned_conjugate_gradient_solver

    # solve!(x, solver, b, args...) solves A*x = b for x.
    solve!(η, solver, rhs, ∫ᶻA.xᶠᶜᶜ, ∫ᶻA.yᶜᶠᶜ, g, Δt)

    return nothing
end

function compute_implicit_free_surface_right_hand_side!(rhs,
                                                        implicit_solver::PCGImplicitFreeSurfaceSolver,
                                                        g, Δt, ∫ᶻQ, η)

    solver = implicit_solver.preconditioned_conjugate_gradient_solver
    arch = solver.architecture
    grid = solver.grid

    event = launch!(arch, grid, :xy,
                    implicit_free_surface_right_hand_side!,
                    rhs, grid, g, Δt, ∫ᶻQ, η,
		            dependencies = device_event(arch))

    return event
end

""" Compute the divergence of fluxes Qu and Qv. """
@inline flux_div_xyᶜᶜᵃ(i, j, k, grid, Qu, Qv) = δxᶜᵃᵃ(i, j, k, grid, Qu) + δyᵃᶜᵃ(i, j, k, grid, Qv)

@kernel function implicit_free_surface_right_hand_side!(rhs, grid, g, Δt, ∫ᶻQ, η)
    i, j = @index(Global, NTuple)
    Az = Azᶜᶜᵃ(i, j, 1, grid)
    δ_Q = flux_div_xyᶜᶜᵃ(i, j, 1, grid, ∫ᶻQ.u, ∫ᶻQ.v)
    @inbounds rhs[i, j, 1] = (δ_Q - Az * η[i, j, 1] / Δt) / (g * Δt)
end

"""
    implicit_free_surface_linear_operation!(result, x, arch, grid, bcs; args...)

Returns `L(ηⁿ)`, where `ηⁿ` is the free surface displacement at time step `n`
and `L` is the linear operator that arises
in an implicit time step for the free surface displacement `η`.

(See the docs section on implicit time stepping.)
"""
function implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)
    grid = L_ηⁿ⁺¹.grid
    arch = architecture(L_ηⁿ⁺¹)

    fill_halo_regions!(ηⁿ⁺¹, arch)

    event = launch!(arch, grid, :xy, _implicit_free_surface_linear_operation!,
                    L_ηⁿ⁺¹, grid,  ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    fill_halo_regions!(L_ηⁿ⁺¹, arch)

    return nothing
end

# Kernels that act on vertically integrated / surface quantities
@inline ∫ᶻ_Ax_∂x_ηᶠᶜᶜ(i, j, k, grid, ∫ᶻ_Axᶠᶜᶜ, η) = @inbounds ∫ᶻ_Axᶠᶜᶜ[i, j, k] * ∂xᶠᶜᵃ(i, j, k, grid, η)
@inline ∫ᶻ_Ay_∂y_ηᶜᶠᶜ(i, j, k, grid, ∫ᶻ_Ayᶜᶠᶜ, η) = @inbounds ∫ᶻ_Ayᶜᶠᶜ[i, j, k] * ∂yᶜᶠᵃ(i, j, k, grid, η)

"""
Compute the horizontal divergence of vertically-uniform quantity using
vertically-integrated face areas `∫ᶻ_Axᶠᶜᶜ` and `∫ᶻ_Ayᶜᶠᶜ`.
"""
@inline Az_∇h²ᶜᶜᵃ(i, j, k, grid, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, η::ReducedField{X, Y, Nothing}) where {X, Y} =
    (δxᶜᵃᵃ(i, j, k, grid, ∫ᶻ_Ax_∂x_ηᶠᶜᶜ, ∫ᶻ_Axᶠᶜᶜ, η) +
     δyᵃᶜᵃ(i, j, k, grid, ∫ᶻ_Ay_∂y_ηᶜᶠᶜ, ∫ᶻ_Ayᶜᶠᶜ, η))

"""
    _implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, grid, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

Return the left side of the "implicit η equation"

```math
(∇ʰ⋅H∇ʰ - 1/gΔt²) ηⁿ⁺¹ = 1 / (gΔt) ∇ʰ ⋅ Q★ - 1 / (gΔt²) ηⁿ
----------------------
        ≡ L_ηⁿ⁺¹
```

which is derived from the discretely summed barotropic mass conservation equation,
and arranged in a symmetric form by multiplying by horizontal areas Az:

```
δⁱÂʷ∂ˣηⁿ⁺¹ + δʲÂˢ∂ʸηⁿ⁺¹ - 1/gΔt² Az ηⁿ⁺¹ = 1 / (gΔt) (δⁱÂʷu̅ˢᵗᵃʳ + δʲÂˢv̅ˢᵗᵃʳ) - 1 / gΔt² Az ηⁿ
```

where  ̂ indicates a vertical integral, and                   
       ̅ indicates a vertical average                         
"""
@kernel function _implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, grid, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)
    i, j = @index(Global, NTuple)
    Az = Azᶜᶜᵃ(i, j, 1, grid)
    @inbounds L_ηⁿ⁺¹[i, j, 1] = Az_∇h²ᶜᶜᵃ(i, j, 1, grid, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, ηⁿ⁺¹) - Az * ηⁿ⁺¹[i, j, 1] / (g * Δt^2)
end
