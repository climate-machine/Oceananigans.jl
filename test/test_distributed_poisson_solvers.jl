
function random_divergent_source_term(arch, grid)
    # Generate right hand side from a random (divergent) velocity field.
    Ru = XFaceField(arch, grid)
    Rv = YFaceField(arch, grid)
    Rw = ZFaceField(arch, grid)
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, (x, y, z) -> rand())
    set!(Rv, (x, y, z) -> rand())
    set!(Rw, (x, y, z) -> rand())

    fill_halo_regions!(Ru, arch)
    fill_halo_regions!(Rv, arch)
    fill_halo_regions!(Rw, arch)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    event = launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    return R
end

function divergence_free_poisson_solution_triply_periodic(grid_points, ranks)
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=grid_points, extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=ranks)
    model = DistributedNonhydrostaticModel(architecture=arch, grid=full_grid)

    local_grid = model.grid
    solver = DistributedFFTBasedPoissonSolver(arch, full_grid, local_grid)

    R = random_divergent_source_term(child_architecture(arch), local_grid)
    first(solver.storage) .= R

    ϕc = first(solver.storage)
    solve!(ϕc, solver)

    p_bcs = FieldBoundaryConditions(local_grid, (Center, Center, Center))
    p_bcs = inject_halo_communication_boundary_conditions(p_bcs, arch.local_rank, arch.connectivity)

    ϕ   = CenterField(child_architecture(arch), local_grid, p_bcs)  # "pressure"
    ∇²ϕ = CenterField(child_architecture(arch), local_grid, p_bcs)

    interior(ϕ) .= real(first(solver.storage))
    compute_∇²!(∇²ϕ, ϕ, arch, local_grid)

    return R ≈ interior(∇²ϕ)
end

@testset "Distributed FFT-based Poisson solver" begin
    @info "  Testing distributed FFT-based Poisson solver..."
    @test divergence_free_poisson_solution_triply_periodic((16, 16, 1), (1, 4, 1))
    @test divergence_free_poisson_solution_triply_periodic((44, 44, 1), (1, 4, 1))
end
