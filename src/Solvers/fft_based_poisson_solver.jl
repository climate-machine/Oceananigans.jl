using Oceananigans.Architectures: device_event
using Oceananigans: short_show

struct FFTBasedPoissonSolver{A, G, Λ, S, B, T}
    architecture :: A
            grid :: G
     eigenvalues :: Λ
         storage :: S
          buffer :: B
      transforms :: T
end

transform_str(transform) = string(typeof(transform).name.wrapper, ", ")

function transform_list_str(transform_list)
    transform_strs = (transform_str(t) for t in transform_list)
    list = string(transform_strs...)
    list = list[1:end-2]
    return list
end

Base.show(io::IO, solver::FFTBasedPoissonSolver{A, G}) where {A, G}=
    print(io, "FFTBasedPoissonSolver{$A}: \n",
          "├── grid: $(short_show(solver.grid))\n",
          "├── storage: $(typeof(solver.storage))\n",
          "├── buffer: $(typeof(solver.buffer))\n",
          "└── transforms:\n",
          "    ├── forward: ", transform_list_str(solver.transforms.forward), "\n",
          "    └── backward: ", transform_list_str(solver.transforms.backward))

function FFTBasedPoissonSolver(arch, grid, planner_flag=FFTW.PATIENT)
    topo = (TX, TY, TZ) =  topology(grid)

    λx = poisson_eigenvalues(grid.Nx, grid.Lx, 1, TX())
    λy = poisson_eigenvalues(grid.Ny, grid.Ly, 2, TY())
    λz = poisson_eigenvalues(grid.Nz, grid.Lz, 3, TZ())

    eigenvalues = (
        λx = arch_array(arch, λx),
        λy = arch_array(arch, λy),
        λz = arch_array(arch, λz)
    )

    storage = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))

    transforms = plan_transforms(arch, grid, storage, planner_flag)

    # Need buffer for index permutations and transposes.
    buffer_needed = arch isa GPU && Bounded in topo
    buffer = buffer_needed ? similar(storage) : nothing

    return FFTBasedPoissonSolver(arch, grid, eigenvalues, storage, buffer, transforms)
end

"""
    solve!(ϕ, poisson_solver::FFTBasedPoissonSolver, b, m=0)

Solves the "generalized" Poisson equation,

```math
(∇² + m) ϕ = b,
```

where ``m`` is a number, using a eigenfunction expansion of the discrete Poisson operator
on a staggered grid and for periodic or Neumann boundary conditions.

In-place transforms are applied to ``b``, which means ``b`` must have complex-valued
elements (typically the same type as `poisson_solver.storage`).

Note: ``(∇² + m) ϕ = b`` is sometimes called the "screened Poisson" equation
when ``m < 0``, or the Helmholtz equation when ``m > 0``.
"""
function solve!(ϕ, solver::FFTBasedPoissonSolver, b, m=0)
    arch = solver.architecture
    topo = TX, TY, TZ = topology(solver.grid)
    Nx, Ny, Nz = size(solver.grid)
    λx, λy, λz = solver.eigenvalues

    # Temporarily store the solution in ϕc
    ϕc = solver.storage

    # Transform b *in-place* to eigenfunction space
    [transform!(b, solver.buffer) for transform! in solver.transforms.forward]

    # Solve the discrete screened Poisson equation (∇² + m) ϕ = b.
    @. ϕc = - b / (λx + λy + λz - m)

    # If m === 0, the "zeroth mode" at `i, j, k = 1, 1, 1` is undetermined;
    # we set this to zero by default. Another slant on this "problem" is that
    # λx[1, 1, 1] + λy[1, 1, 1] + λz[1, 1, 1] = 0, which yields ϕ[1, 1, 1] = Inf or NaN.
    m === 0 && CUDA.@allowscalar ϕc[1, 1, 1] = 0

    # Apply backward transforms in order
    [transform!(ϕc, solver.buffer) for transform! in solver.transforms.backward]

    copy_event = launch!(arch, solver.grid, :xyz, copy_real_component!, ϕ, ϕc, dependencies=device_event(arch))
    wait(device(arch), copy_event)

    return ϕ
end

@kernel function copy_real_component!(ϕ, ϕc)
    i, j, k = @index(Global, NTuple)
    @inbounds ϕ[i, j, k] = real(ϕc[i, j, k])
end
