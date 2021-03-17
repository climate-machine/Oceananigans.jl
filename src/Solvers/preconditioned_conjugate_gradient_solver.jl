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

    a_res = similar(tf.data)
    q     = similar(tf.data)
    p     = similar(tf.data)
    z     = similar(tf.data)
    r     = similar(tf.data)
    RHS   = similar(tf.data)
    x     = similar(tf.data)

    a_res.parent .= 0

    if isnothing(parameters.PCmatrix_function)
        # preconditioner not provided, use the Identity matrix
        PCmatrix_function(x; args...) = x
    else
        PCmatrix_function = parameters.PCmatrix_function
    end

    M(x; args...) = PCmatrix_function(x; args...)

    ii = interior_parent_indices(Center, topology(grid, 1), grid.Nx, grid.Hx)
    ji = interior_parent_indices(Center, topology(grid, 2), grid.Ny, grid.Hy)
    ki = interior_parent_indices(Center, topology(grid, 3), grid.Nz, grid.Hz)

    dotproduct(x,y)  = mapreduce((x,y)->x*y, + , view(x, ii, ji, ki), view(y, ii, ji, ki))
    norm(x)          = ( mapreduce((x)->x*x, + , view(x, ii, ji, ki)   ) )^0.5

    Amatrix_function = parameters.Amatrix_function
    A(x; args...) = ( Amatrix_function(a_res, x, arch, grid, bcs; args...); return  a_res )

    reference_pressure_solver = nothing
    if haskey(parameters, :reference_pressure_solver )
        reference_pressure_solver = parameters.reference_pressure_solver
    end

    settings = (
                                q=q,
                                p=p,
                                z=z,
                                r=r,
                                x=x,
                              RHS=RHS,
                              bcs=bcs,
                             grid=grid,
                                A=A,
                                M=M,
                            maxit=maxit,
                              tol=tol,
                          dotprod=dotproduct,
                             norm=norm,
                             arch=arch,
        reference_pressure_solver=reference_pressure_solver,
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

function solve_poisson_equation!(solver::PreconditionedConjugateGradientSolver, RHS, x; args...)
#
# Alg - see Fig 2.5 The Preconditioned Conjugate Gradient Method in
#                    "Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods"
#                    Barrett et. al, 2nd Edition.
#
#     given
#        linear Preconditioner operator M as a function M()
#        linear A matrix operator A as a function A()
#        a dot product function norm()
#        a right-hand side b
#        an initial guess x
#
#        local vectors: z, r, p, q
#
#     β  = 0
#     r .= b-A(x)
#     i  = 0
#
#     loop:
#      if i > MAXIT
#       break
#      end
#      z = M( r )
#      ρ    .= dotprod(r,z)
#      p = z+β*p
#      q = A(p)
#      α=ρ/dotprod(p,q)
#      x=x.+αp
#      r=r.-αq
#      if norm2(r) < tol
#       break
#      end
#      i=i+1
#      ρⁱᵐ1 .= ρ
#      β    .= ρⁱᵐ¹/ρ
#
    arch       = solver.architecture
    sset       = solver.settings
    grid       = sset.grid
    z, r, p, q = sset.z, sset.r, sset.p, sset.q
    A          = sset.A
    M          = sset.M
    maxit      = sset.maxit
    tol        = sset.tol
    dotprod    = sset.dotprod
    norm       = sset.norm

    β    = 0.0
    i    = 0
    ρ    = 0.0
    ρⁱᵐ¹ = 0.0

    r.parent .= 0
    # quick_launch!(arch, grid, compute_residual!, r, RHS, A(x))
    r.parent .= RHS.parent .- A(x; args...).parent
    ### println("PreconditionedConjugateGradientSolver ", i," RHS ", norm(RHS.parent) )
    ### println("PreconditionedConjugateGradientSolver ", i," A(x) ", norm(A(x; args...).parent) )

    while true
        ### println("PreconditionedConjugateGradientSolver ", i," norm(r.parent) ", norm(r.parent) )
        i >= maxit && break
        norm(r.parent) <= tol && break

        z.parent       .= M(r; args...).parent
        ### println("PreconditionedConjugateGradientSolver ", i," norm(z.parent) ", norm(z.parent) )
        ρ        = dotprod(z.parent, r.parent)
        ### println("PreconditionedConjugateGradientSolver ", i," ρ ", ρ )

        if i == 0
            p.parent   .= z.parent
        else
            β    = ρ/ρⁱᵐ¹
            p.parent   .= z.parent .+ β .* p.parent
        end

        q.parent       .= A(p; args...).parent
        ### println("PreconditionedConjugateGradientSolver ", i," norm(q.parent) ", norm(q.parent) )
        α        = ρ / dotprod(p.parent, q.parent)
        ### println("PreconditionedConjugateGradientSolver ", i," α ", α )
        x.parent       .= x.parent .+ α .* p.parent
        r.parent       .= r.parent .- α .* q.parent

        i     = i+1
        ρⁱᵐ¹  = ρ
    end

#==
#     No preconditioner verison
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
==#
      println("PreconditionedConjugateGradientSolver ", i," ", norm(r.parent) )

    fill_halo_regions!(x, sset.bcs, sset.grid)

    return x, norm(r.parent)
end

function Base.show(io::IO, solver::PreconditionedConjugateGradientSolver)
    print(io, "Oceanigans compatible preconditioned conjugate gradient solver.\n")
    print(io, " Problem size = "  , size(solver.settings.q) )
    print(io, "\n Boundary conditions = "  , solver.settings.bcs  )
    print(io, "\n Grid = "  , solver.settings.grid  )
    return nothing
end
