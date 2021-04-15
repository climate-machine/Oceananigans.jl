using Oceananigans.Grids: interior_parent_indices

struct PreconditionedConjugateGradientSolver{A, S}
    architecture :: A
        settings :: S
end

function PreconditionedConjugateGradientSolver(; arch = arch, grid = nothing, boundary_conditions = nothing, parameters = parameters)

    bcs = boundary_conditions
    if isnothing(boundary_conditions)
        bcs = parameters.Template_field.boundary_conditions
    end

    if isnothing(grid)
        grid = parameters.Template_field.grid
    end

    maxit = parameters.maxit
    if isnothing(parameters.maxit)
        maxit = grid.Nx * grid.Ny * grid.Nz
    end

    tol = parameters.tol
    if isnothing(parameters.tol)
        tol = 1e-13
    end

    tf = parameters.Template_field
    if isnothing(parameters.Template_field)
        tf = CenterField(arch, grid)
    end

    ## a_res = similar(tf) #.data)
    ## q     = similar(tf) #.data)
    ## p     = similar(tf) #.data)
    ## z     = similar(tf) #.data)
    ## r     = similar(tf) #.data)
    ## RHS   = similar(tf) #.data)
    ## x     = similar(tf) #.data)
    a_res = similar(interior(tf) ) #.data)
    q     = similar(interior(tf) ) #.data)
    p     = similar(interior(tf) ) #.data)
    z     = similar(interior(tf) ) #.data)
    r     = similar(interior(tf) ) #.data)
    RHS   = similar(interior(tf) ) #.data)
    x     = similar(interior(tf) ) #.data)


    parent(a_res) .= 0

    if isnothing(parameters.PCmatrix_function)
        # preconditioner not provided, use the Identity matrix
        PCmatrix_function(x; args...) = x
    else
        PCmatrix_function = parameters.PCmatrix_function
    end

    M(x; args...) = PCmatrix_function(x; args...)

    dr=range( 1,length=ndims( interior(tf) ) )
    ininds=map(i->range(1,length=size( interior(tf) ,i)),dr)

    ## ii = interior_parent_indices(location(tf, 1), topology(grid, 1), grid.Nx, grid.Hx)
    ## ji = interior_parent_indices(location(tf, 2), topology(grid, 2), grid.Ny, grid.Hy)
    ## ki = interior_parent_indices(location(tf, 3), topology(grid, 3), grid.Nz, grid.Hz)

    # view(interior(η),ininds...)

    dotproduct(x, y) = mapreduce((x, y) -> x * y, +, view(x, ininds...), view(y, ininds...))
    norm(x) = sqrt(mapreduce(x -> x * x, +, view(x, ininds...)))

    Amatrix_function = parameters.Amatrix_function
    A(x; args...) = (Amatrix_function(a_res, x, arch, grid, bcs; args...); return a_res)

    reference_pressure_solver = nothing
    if haskey(parameters, :reference_pressure_solver )
        reference_pressure_solver = parameters.reference_pressure_solver
    end

    settings = (
                                q = q,
                                p = p,
                                z = z,
                                r = r,
                                x = x,
                              RHS = RHS,
                              bcs = bcs,
                             grid = grid,
                                A = A,
                                M = M,
                            maxit = maxit,
                              tol = tol,
                          dotprod = dotproduct,
                             norm = norm,
                             arch = arch,
        reference_pressure_solver = reference_pressure_solver,
    )

   return PreconditionedConjugateGradientSolver(arch, settings)
end

@kernel function compute_residual!(r, RHS, A)
    i, j, k = @index(Global, NTuple)
    @inbounds r[i, j, k] = RHS[i, j, k] - A[i, j, k]
end

function quick_launch!(arch, grid, kernel!, args...)
    event = launch!(arch, grid, :xyz, kernel!, args...; dependencies=Event(device(arch)))
    wait(device(arch), event)
    return nothing
end

using Statistics

"""
    solve_poisson_equation!(solver::PreconditionedConjugateGradientSolver, RHS, x; args...)

Solves the Poisson equation using an iterative conjugate-gradient method.
    
See fig 2.5 in

> The Preconditioned Conjugate Gradient Method in
  "Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods"
  Barrett et. al, 2nd Edition.
    
Given:
    * Linear Preconditioner operator M as a function M();
    * Linear A matrix operator A as a function A();
    * A dot product function norm();
    * A right-hand side b;
    * An initial guess x; and
    * Local vectors: z, r, p, q

This function executes the algorithm
    
```
    β  = 0
    r .= b - A(x)
    i  = 0
    
    Loop:
         if i > MAXIT
            break
         end

         z  = M( r )
         ρ .= dotprod(r,z)
         p  = z + β*p
         q  = A(p)

         α = ρ / dotprod(p,q)
         x = x .+ αp
         r = r .- αq

         if norm2(r) < tol
            break
         end

         i = i + 1
         ρⁱᵐ1 .= ρ
         β    .= ρⁱᵐ¹/ρ
"""
function solve_poisson_equation!(solver::PreconditionedConjugateGradientSolver, RHS, x; args...)
    arch       = solver.architecture
    settings   = solver.settings
    grid       = settings.grid
    z, r, p, q = settings.z, settings.r, settings.p, settings.q
    A          = settings.A
    M          = settings.M
    maxit      = settings.maxit
    tol        = settings.tol
    dotprod    = settings.dotprod
    norm       = settings.norm

    β    = 0.0
    i    = 0
    ρ    = 0.0
    ρⁱᵐ¹ = 0.0

    parent(r) .= 0

    # quick_launch!(arch, grid, compute_residual!, r, RHS, A(x))
    parent(r) .= parent(RHS) .- parent(A(x; args...))

    ### println("PreconditionedConjugateGradientSolver ", i," RHS ", norm(RHS.parent) )
    ### println("PreconditionedConjugateGradientSolver ", i," A(x) ", norm(A(x; args...).parent) )

    while true
        @debug println("PreconditionedConjugateGradientSolver ", i," norm(parent(r)) ", norm(parent(r)))
        
        i >= maxit && break
        norm(parent(r)) <= tol && break

        parent(z) .= parent(M(r; args...))
        @debug println("PreconditionedConjugateGradientSolver ", i," norm(parent(z)) ", norm(parent(z)))

        ρ = dotprod(parent(z), parent(r))
        @debug println("PreconditionedConjugateGradientSolver ", i," ρ ", ρ )

        if i == 0
            parent(p) .= parent(z)
        else
            β = ρ / ρⁱᵐ¹
            parent(p) .= parent(z) .+ β .* parent(p)
        end

        parent(q) .= parent(A(p; args...))
        @debug println("PreconditionedConjugateGradientSolver ", i," norm(parent(q)) ", norm(parent(q)))
        
        α = ρ / dotprod(parent(p), parent(q))
        @debug println("PreconditionedConjugateGradientSolver ", i," α ", α)
        
        parent(x) .= parent(x) .+ α .* parent(p)
        parent(r) .= parent(r) .- α .* parent(q)

        i     = i+1
        ρⁱᵐ¹  = ρ
    end

    #=
    # No preconditioner verison
    i    = 0
    r   .= RHS .- A(x)
    p   .= r
    γ    = dotprod(r,r)

    while true
        if i > maxit
            break
        end
        q   .= A(p)
        α    = γ/dotprod(p,q)
        x   .= x .+ α .* p
        r   .= r .- α .* q
        println("Solver ", i," ", norm(r) )
        if norm(r) <= tol
            break
        end
        γ⁺   = dotprod(r,r)
        β    = γ⁺/γ
        p   .= r .+ β .* p
        γ    = γ⁺
        i    = i+1
    end

    println("PreconditionedConjugateGradientSolver ", i," ", norm(r.parent) )
    =#

    fill_halo_regions!(x, arch)

    return x, norm(parent(r))
end

function Base.show(io::IO, solver::PreconditionedConjugateGradientSolver)
    print(io, "Oceanigans compatible preconditioned conjugate gradient solver.\n")
    print(io, " Problem size = "  , size(solver.settings.q) )
    print(io, "\n Boundary conditions = "  , solver.settings.bcs  )
    print(io, "\n Grid = "  , solver.settings.grid  )
    return nothing
end
