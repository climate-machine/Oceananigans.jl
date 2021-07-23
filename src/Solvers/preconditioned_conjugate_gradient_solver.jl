using Oceananigans.Architectures: architecture
using Oceananigans.Grids: interior_parent_indices
using Statistics: norm, dot
using LinearAlgebra

mutable struct PreconditionedConjugateGradientSolver{A, G, L, T, F, M, P}
               architecture :: A
                       grid :: G
          linear_operation! :: L
                  tolerance :: T
         maximum_iterations :: Int
                  iteration :: Int
                       ρⁱ⁻¹ :: T
    linear_operator_product :: F
           search_direction :: F
                   residual :: F
              precondition! :: M
     preconditioner_product :: P
end

no_precondition!(args...) = nothing

initialize_precondition_product(precondition, template_field) = similar(template_field)
initialize_precondition_product(::Nothing, template_field) = nothing

maybe_precondition(::Nothing, ::Nothing, r, args...) = r

function maybe_precondition(precondition!, z, r, args...)
    precondition!(z, r, args...)
    return z
end
    
"""
    PreconditionedConjugateGradientSolver(linear_operation;
                                          template_field,
                                          maximum_iterations = size(template_field.grid),
                                          tolerance = 1e-13,
                                          precondition = nothing)

Returns a PreconditionedConjugateGradientSolver that solves the linear equation
``A*x = b`` using a iterative conjugate gradient method with optional preconditioning.
The solver is used by calling

```
solve!(x, solver::PreconditionedConjugateGradientOperator, b, args...)
```

for `solver`, right-hand side `b`, solution `x`, and optional arguments `args...`.

Args
====

* `template_field`: Dummy field that is the same type and size as `x` and `b`, which
                    is used to infer the `architecture`, `grid`, and to create work arrays
                    that are used internally by the solver.

* `linear_operation`: Function with signature `linear_operation!(p, y, args...)` that calculates
                     `A*y` and stores the result in `p` for a "candidate solution `y`. `args...`
                     are optional positional arguments passed from `solve!(x, solver, b, args...)`.

* `maximum_iterations`: Maximum number of iterations the solver may perform before exiting.

* `tolerance`: Tolerance for convergence of the algorithm. The algorithm quits when
               `norm(A*x - b) < tolerance`.

* `precondition`: Function with signature `preconditioner!(z, y, args...)` that calculates
                    `P*y` and stores the result in `z` for linear operator `P`.
                    Note that some precondition algorithms describe the step
                    "solve `M*x = b`" for precondition `M`"; in this context,
                    `P = M⁻¹`.

See `solve!` for more information about the preconditioned conjugate-gradient algorithm.
"""
function PreconditionedConjugateGradientSolver(linear_operation;
                                               template_field::AbstractField,
                                               maximum_iterations = prod(size(template_field)),
                                               tolerance = 1e-13, #sqrt(eps(eltype(template_field.grid))),
                                               precondition = nothing)

    arch = architecture(template_field)
    grid = template_field.grid

    # Create work arrays for solver
    linear_operator_product = similar(template_field) # A*xᵢ = qᵢ
    search_direction = similar(template_field) # pᵢ
            residual = similar(template_field) # rᵢ

    # Either nothing (no precondition) or P*xᵢ = zᵢ
    precondition_product = initialize_precondition_product(precondition, template_field)

    return PreconditionedConjugateGradientSolver(arch,
                                                 grid,
                                                 linear_operation,
                                                 tolerance,
                                                 maximum_iterations,
                                                 0,
                                                 0.0,
                                                 linear_operator_product,
                                                 search_direction,
                                                 residual,
                                                 precondition,
                                                 precondition_product)
end

"""
    solve!(x, solver::PreconditionedConjugateGradientSolver, b, args...)

Solves A*x = b using an iterative conjugate-gradient method,
where A*x is determined by solver.linear_operation
    
See fig 2.5 in

> The Preconditioned Conjugate Gradient Method in
  "Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods"
  Barrett et. al, 2nd Edition.
    
Given:
    * Linear Preconditioner operator M!(solution, x, other_args...) that computes `M*x=solution`
    * Linear A matrix operator A as a function A();
    * A dot product function norm();
    * A right-hand side b;
    * An initial guess x; and
    * Local vectors: z, r, p, q

This function executes the algorithm
    
```
β  = 0
r = b - A(x)
iteration  = 0

Loop:
     if iteration > maximum_iterations
        break
     end

     ρ = r ⋅ z

     z = M(r)
     β = ρⁱ⁻¹ / ρ
     p = z + β * p
     q = A(p)

     α = ρ / (p ⋅ q)
     x = x + α * p
     r = r - α * q

     if |r| < tolerance
        break
     end

     iteration += 1
     ρⁱ⁻¹ = ρ
```
"""
function solve!(x, solver::PreconditionedConjugateGradientSolver, b, args...)

    # Initialize
    solver.iteration = 0

    # q = A*x
    q = solver.linear_operator_product
    solver.linear_operation!(q, x, args...)

    # r = b - A*x
    solver.residual .= b .- q

    @debug "PreconditionedConjugateGradientSolver, |b|: $(norm(b))"
    @debug "PreconditionedConjugateGradientSolver, |A(x)|: $(norm(q))"

    while iterating(solver)
        iterate!(x, solver, b, args...)
    end

    fill_halo_regions!(x, solver.architecture) # blocking

    return x
end

function iterate!(x, solver, b, args...)
    r = solver.residual
    p = solver.search_direction
    q = solver.linear_operator_product

    @debug "PreconditionedConjugateGradientSolver $(solver.iteration), |r|: $(norm(r))"

    # Preconditioned:   z = P * r
    # Unpreconditioned: z = r
    z = maybe_precondition(solver.precondition!, solver.preconditioner_product, r, args...) 
    ρ = dot(z, r)

    @debug "PreconditionedConjugateGradientSolver $(solver.iteration), ρ: $ρ"
    @debug "PreconditionedConjugateGradientSolver $(solver.iteration), |z|: $(norm(z))"

    if solver.iteration == 0
        p .= z
    else
        β = ρ / solver.ρⁱ⁻¹
        p .= z .+ β .* p

        @debug "PreconditionedConjugateGradientSolver $(solver.iteration), β: $β"
    end

    # q = A * p
    solver.linear_operation!(q, p, args...)
    α = ρ / dot(p, q)

    @debug "PreconditionedConjugateGradientSolver $(solver.iteration), |q|: $(norm(q))"
    @debug "PreconditionedConjugateGradientSolver $(solver.iteration), α: $α"
        
    x .+= α .* p
    r .-= α .* q

    solver.iteration += 1
    solver.ρⁱ⁻¹ = ρ

    return nothing
end

function iterating(solver)
    # End conditions
    solver.iteration >= solver.maximum_iterations && return false
    norm(solver.residual) <= solver.tolerance && return false
    return true
end

function Base.show(io::IO, solver::PreconditionedConjugateGradientSolver)
    print(io, "Oceananigans-compatible preconditioned conjugate gradient solver.\n")
    print(io, " Problem size = "  , size(solver.q), '\n')
    print(io, " Grid = "  , solver.grid)
    return nothing
end
