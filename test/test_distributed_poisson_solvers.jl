
function random_divergent_source_term(FT, arch, grid)
    # Generate right hand side from a random (divergent) velocity field.
    Ru = CenterField(FT, arch, grid, UVelocityBoundaryConditions(grid))
    Rv = CenterField(FT, arch, grid, VVelocityBoundaryConditions(grid))
    Rw = CenterField(FT, arch, grid, WVelocityBoundaryConditions(grid))
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, rand(Nx, Ny, Nz))
    set!(Rv, rand(Nx, Ny, Nz))
    set!(Rw, rand(Nx, Ny, Nz))

    # Adding (nothing, nothing) in case we need to dispatch on ::NFBC
    fill_halo_regions!(Ru, arch, nothing, nothing)
    fill_halo_regions!(Rv, arch, nothing, nothing)
    fill_halo_regions!(Rw, arch, nothing, nothing)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    event = launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    return R
end

function compute_∇²!(∇²ϕ, ϕ, arch, grid)
    fill_halo_regions!(ϕ, arch)
    child_arch = child_architecture(arch)
    event = launch!(child_arch, grid, :xyz, ∇²!, grid, ϕ.data, ∇²ϕ.data, dependencies=Event(device(child_arch)))
    wait(device(child_arch), event)
    fill_halo_regions!(∇²ϕ, arch)
    return nothing
end

function divergence_free_poisson_solution_triply_periodic(grid_points, ranks)
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=grid_points, extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=ranks)
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid)

    local_grid = model.grid
    solver = DistributedFFTBasedPoissonSolver(arch, full_grid, local_grid)

    R = random_divergent_source_term(Float64, child_architecture(arch), local_grid)
    first(solver.storage) .= R

    solve_poisson_equation!(solver)

    p_bcs = PressureBoundaryConditions(local_grid)
    p_bcs = inject_halo_communication_boundary_conditions(p_bcs, arch.my_rank, arch.connectivity)

    ϕ   = CenterField(Float64, child_architecture(arch), local_grid, p_bcs)  # "pressure"
    ∇²ϕ = CenterField(Float64, child_architecture(arch), local_grid, p_bcs)

    interior(ϕ) .= real(first(solver.storage))
    compute_∇²!(∇²ϕ, ϕ, arch, local_grid)

    return R ≈ interior(∇²ϕ)
end

@testset "Distributed FFT-based Poisson solver" begin
    @info "  Testing distributed FFT-based Poisson solver..."
    @test divergence_free_poisson_solution_triply_periodic((16, 16, 1), (1, 4, 1))
    @test divergence_free_poisson_solution_triply_periodic((64, 64, 1), (1, 4, 1))
end