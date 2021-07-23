import PencilFFTs

import Oceananigans.Solvers: poisson_eigenvalues, solve!
using Oceananigans.Solvers: copy_real_component!

struct DistributedFFTBasedPoissonSolver{A, P, F, L, λ, S}
      architecture :: A
              plan :: P
         full_grid :: F
           my_grid :: L
       eigenvalues :: λ
           storage :: S
end

function DistributedFFTBasedPoissonSolver(arch, full_grid, local_grid)
    topo = (TX, TY, TZ) = topology(full_grid)

    λx = poisson_eigenvalues(full_grid.Nx, full_grid.Lx, 1, TX())
    λy = poisson_eigenvalues(full_grid.Ny, full_grid.Ly, 2, TY())
    λz = poisson_eigenvalues(full_grid.Nz, full_grid.Lz, 3, TZ())

    I, J, K = arch.local_index
    λx = λx[(J-1)*local_grid.Ny+1:J*local_grid.Ny, :, :]

    eigenvalues = (; λx, λy, λz)

    transform = PencilFFTs.Transforms.FFT!()
    proc_dims = (arch.ranks[2], arch.ranks[3])
    plan = PencilFFTs.PencilFFTPlan(size(full_grid), transform, proc_dims, MPI.COMM_WORLD)
    storage = PencilFFTs.allocate_input(plan)

    return DistributedFFTBasedPoissonSolver(arch, plan, full_grid, local_grid, eigenvalues, storage)
end

function solve!(x, solver::DistributedFFTBasedPoissonSolver)
    arch = solver.architecture
    λx, λy, λz = solver.eigenvalues

    # Apply forward transforms.
    solver.plan * solver.storage

    # Solve the discrete Poisson equation, storing the solution
    # temporarily in xc and later extracting the real part into 
    # the solution, x.
    xc = b = solver.storage[2]
    @. xc = - b / (λx + λy + λz)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        xc[1, 1, 1] = 0
    end

    # Apply backward transforms.
    solver.plan \ solver.storage
    xc_transposed = first(solver.storage)
	
    copy_event = launch!(arch, solver.my_grid, :xyz, copy_real_component!, x, xc_transposed, dependencies=device_event(arch))
    wait(device(arch), copy_event)

    return x
end
