using Oceananigans.Solvers: solve!, set_source_term!
using Oceananigans.Solvers: poisson_eigenvalues
using Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: _compute_w_from_continuity!
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions

function poisson_solver_instantiates(arch, grid, planner_flag)
    solver = FFTBasedPoissonSolver(arch, grid, planner_flag)
    return true  # Just making sure the FFTBasedPoissonSolver does not error/crash.
end

function random_divergent_source_term(arch, grid)
    default_bcs = FieldBoundaryConditions()
    u_bcs = regularize_field_boundary_conditions(default_bcs, grid, :u)
    v_bcs = regularize_field_boundary_conditions(default_bcs, grid, :v)
    w_bcs = regularize_field_boundary_conditions(default_bcs, grid, :w)

    Ru = CenterField(arch, grid, u_bcs)
    Rv = CenterField(arch, grid, v_bcs)
    Rw = CenterField(arch, grid, w_bcs)
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, rand(Nx, Ny, Nz))
    set!(Rv, rand(Nx, Ny, Nz))
    set!(Rw, rand(Nx, Ny, Nz))

    fill_halo_regions!(Ru, arch)
    fill_halo_regions!(Rv, arch)
    fill_halo_regions!(Rw, arch)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    event = launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    return R, U
end

function random_divergence_free_source_term(arch, grid)
    default_bcs = FieldBoundaryConditions()
    u_bcs = regularize_field_boundary_conditions(default_bcs, grid, :u)
    v_bcs = regularize_field_boundary_conditions(default_bcs, grid, :v)
    w_bcs = regularize_field_boundary_conditions(default_bcs, grid, :w)

    # Random right hand side
    Ru = CenterField(arch, grid, u_bcs)
    Rv = CenterField(arch, grid, v_bcs)
    Rw = CenterField(arch, grid, w_bcs)
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, rand(Nx, Ny, Nz))
    set!(Rv, rand(Nx, Ny, Nz))
    set!(Rw, zeros(Nx, Ny, Nz))

    fill_halo_regions!(Ru, arch, nothing, nothing)
    fill_halo_regions!(Rv, arch, nothing, nothing)
    fill_halo_regions!(Rw, arch, nothing, nothing)

    event = launch!(arch, grid, :xy, _compute_w_from_continuity!, U, grid,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    fill_halo_regions!(Rw, arch, nothing, nothing)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    event = launch!(arch, grid, :xyz, divergence!, grid, Ru.data, Rv.data, Rw.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    return R
end

#####
##### Regular rectilinear grid Poisson solver
#####

function divergence_free_poisson_solution(arch, grid, planner_flag=FFTW.MEASURE)
    ArrayType = array_type(arch)
    FT = eltype(grid)

    solver = FFTBasedPoissonSolver(arch, grid, planner_flag)
    R, U = random_divergent_source_term(arch, grid)

    p_bcs = FieldBoundaryConditions(grid, (Center, Center, Center))
    ϕ   = CenterField(arch, grid, p_bcs)  # "kinematic pressure"
    ∇²ϕ = CenterField(arch, grid, p_bcs)

    # Using Δt = 1 but it doesn't matter since velocities = 0.
    solve_for_pressure!(ϕ.data, solver, 1, U)

    compute_∇²!(∇²ϕ, ϕ, arch, grid)

    return CUDA.@allowscalar interior(∇²ϕ) ≈ R
end

#####
##### Test that Poisson solver error converges as error ~ N⁻²
#####

ψ(::Type{Bounded}, n, x) = cos(n*x/2)
ψ(::Type{Periodic}, n, x) = cos(n*x)

k²(::Type{Bounded}, n) = (n/2)^2
k²(::Type{Periodic}, n) = n^2

function analytical_poisson_solver_test(arch, N, topo; FT=Float64, mode=1)
    grid = RegularRectilinearGrid(FT, topology=topo, size=(N, N, N), x=(0, 2π), y=(0, 2π), z=(0, 2π))
    solver = FFTBasedPoissonSolver(arch, grid)

    xC, yC, zC = nodes((Center, Center, Center), grid, reshape=true)

    TX, TY, TZ = topology(grid)
    Ψ(x, y, z) = ψ(TX, mode, x) * ψ(TY, mode, y) * ψ(TZ, mode, z)
    f(x, y, z) = -(k²(TX, mode) + k²(TY, mode) + k²(TZ, mode)) * Ψ(x, y, z)

    solver.storage .= convert(array_type(arch), f.(xC, yC, zC))

    ϕc = rhs = solver.storage
    solve!(ϕc, solver, rhs)

    ϕ = real(Array(solver.storage))

    L¹_error = mean(abs, ϕ - Ψ.(xC, yC, zC))

    return L¹_error
end

function poisson_solver_convergence(arch, topo, N¹, N²; FT=Float64, mode=1)
    error¹ = analytical_poisson_solver_test(arch, N¹, topo; FT, mode)
    error² = analytical_poisson_solver_test(arch, N², topo; FT, mode)

    rate = log(error¹ / error²) / log(N² / N¹)

    TX, TY, TZ = topo
    @info "Convergence of L¹-normed error, $(typeof(arch)), $FT, ($(N¹)³ -> $(N²)³), topology=($TX, $TY, $TZ): $rate"

    return isapprox(rate, 2, rtol=5e-3)
end

#####
##### Vertically stretched Poisson solver
#####

get_grid_size(TX, TY, TZ, Nx, Ny, Nz) = (Nx, Ny, Nz)
get_grid_size(::Type{Flat}, TY, TZ, Nx, Ny, Nz) = (Ny, Nz)
get_grid_size(TX, ::Type{Flat}, TZ, Nx, Ny, Nz) = (Nx, Nz)

get_xy_interval_kwargs(TX, TY, TZ) = (x=(0, 1), y=(0, 1))
get_xy_interval_kwargs(TX, ::Type{Flat}, TZ) = (x=(0, 1),)
get_xy_interval_kwargs(::Type{Flat}, TY, TZ) = (y=(0, 1),)

function vertically_stretched_poisson_solver_correct_answer(FT, arch, topo, Nx, Ny, zF)
    Nz = length(zF) - 1
    sz = get_grid_size(topo..., Nx, Ny, Nz)
    xy_intervals = get_xy_interval_kwargs(topo...)
    vs_grid = VerticallyStretchedRectilinearGrid(FT; architecture=arch, topology=topo, size=sz, z_faces=zF, xy_intervals...)
    solver = FourierTridiagonalPoissonSolver(arch, vs_grid)

    p_bcs = FieldBoundaryConditions(vs_grid, (Center, Center, Center))
    ϕ   = CenterField(arch, vs_grid, p_bcs)  # "kinematic pressure"
    ∇²ϕ = CenterField(arch, vs_grid, p_bcs)

    R = random_divergence_free_source_term(arch, vs_grid)

    set_source_term!(solver, R)
    ϕc = solver.storage
    solve!(ϕc, solver)

    # interior(ϕ) = solution(solver) or solution!(interior(ϕ), solver)
    CUDA.@allowscalar interior(ϕ) .= real.(solver.storage)
    compute_∇²!(∇²ϕ, ϕ, arch, vs_grid)

    return CUDA.@allowscalar interior(∇²ϕ) ≈ R
end

#####
##### Run pressure solver tests
#####

PB = (Periodic, Bounded)
topos = collect(Iterators.product(PB, PB, PB))[:]

two_dimensional_topologies = [(Flat,     Bounded,  Bounded),
                              (Bounded,  Flat,     Bounded),
                              (Bounded,  Bounded,  Flat),
                              (Flat,     Periodic, Bounded),
                              (Periodic, Flat,     Bounded),
                              (Periodic, Bounded,  Flat)]

@testset "Poisson solvers" begin
    @info "Testing Poisson solvers..."

    for arch in archs
        @testset "Poisson solver instantiation [$(typeof(arch))]" begin
            @info "  Testing Poisson solver instantiation [$(typeof(arch))]..."
            for FT in float_types

                grids_3d = [RegularRectilinearGrid(FT, size=(2, 2, 2), extent=(1, 1, 1)),
                            RegularRectilinearGrid(FT, size=(1, 2, 2), extent=(1, 1, 1)),
                            RegularRectilinearGrid(FT, size=(2, 1, 2), extent=(1, 1, 1)),
                            RegularRectilinearGrid(FT, size=(2, 2, 1), extent=(1, 1, 1))]

                grids_2d = [RegularRectilinearGrid(FT, size=(2, 2), extent=(1, 1), topology=topo)
                            for topo in two_dimensional_topologies]


                grids = []
                push!(grids, grids_3d..., grids_2d...)

                for grid in grids
                    @test poisson_solver_instantiates(arch, grid, FFTW.ESTIMATE)
                    @test poisson_solver_instantiates(arch, grid, FFTW.MEASURE)
                end
            end
        end

        @testset "Divergence-free solution [$(typeof(arch))]" begin
            @info "  Testing divergence-free solution [$(typeof(arch))]..."

            for topo in topos
                for N in [7, 16]

                    grids_3d = [RegularRectilinearGrid(topology=topo, size=(N, N, N), extent=(1, 1, 1)),
                                RegularRectilinearGrid(topology=topo, size=(1, N, N), extent=(1, 1, 1)),
                                RegularRectilinearGrid(topology=topo, size=(N, 1, N), extent=(1, 1, 1)),
                                RegularRectilinearGrid(topology=topo, size=(N, N, 1), extent=(1, 1, 1))]

                    grids_2d = [RegularRectilinearGrid(size=(N, N), extent=(1, 1), topology=topo)
                                for topo in two_dimensional_topologies]

                    grids = []
                    push!(grids, grids_3d..., grids_2d...)

                    for grid in grids
                        N == 7 && @info "    Testing $(topology(grid)) topology on square grids [$(typeof(arch))]..."
                        @test divergence_free_poisson_solution(arch, grid)
                    end
                end
            end

            Ns = [11, 16]
            for topo in topos
                @info "    Testing $topo topology on rectangular grids with even and prime sizes [$(typeof(arch))]..."
                for Nx in Ns, Ny in Ns, Nz in Ns
                    grid = RegularRectilinearGrid(topology=topo, size=(Nx, Ny, Nz), extent=(1, 1, 1))
                    @test divergence_free_poisson_solution(arch, grid)
                end
            end

            # Do a couple at Float32 (since its too expensive to repeat all tests...)
            Float32_grids = [RegularRectilinearGrid(Float32, topology=(Periodic, Bounded, Bounded), size=(16, 16, 16), extent=(1, 1, 1)),
                             RegularRectilinearGrid(Float32, topology=(Bounded, Bounded, Periodic), size=(7, 11, 13), extent=(1, 1, 1))]

            for grid in Float32_grids
                @test divergence_free_poisson_solution(arch, grid)
            end
        end

        @testset "Convergence to analytic solution [$(typeof(arch))]" begin
            @info "  Testing convergence to analytic solution [$(typeof(arch))]..."
            for topo in topos
                @test poisson_solver_convergence(arch, topo, 2^6, 2^7)
                @test poisson_solver_convergence(arch, topo, 67, 131, mode=2)
            end
        end
    end

    # Vertically stretched topologies to test.
    vs_topos = [
        (Periodic, Periodic, Bounded),
        (Periodic, Bounded,  Bounded),
        (Bounded,  Periodic, Bounded),
        (Bounded,  Bounded,  Bounded),
        (Flat,     Bounded,  Bounded),
        (Flat,     Periodic, Bounded),
        (Bounded,  Flat,     Bounded),
        (Periodic, Flat,     Bounded)
    ]

    for arch in archs, topo in vs_topos
        @testset "Vertically stretched Poisson solver [FACR, $(typeof(arch)), $topo]" begin
            @info "  Testing vertically stretched Poisson solver [FACR, $(typeof(arch)), $topo]..."

            @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 8, 8, 1:8)
            @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 7, 7, 1:7)
            @test vertically_stretched_poisson_solver_correct_answer(Float32, arch, topo, 8, 8, 1:8)

            zF_even = [1, 2, 4, 7, 11, 16, 22, 29, 37]      # Nz = 8
            zF_odd  = [1, 2, 4, 7, 11, 16, 22, 29, 37, 51]  # Nz = 9

            for zF in [zF_even, zF_odd]
                @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 8,  8, zF)
                @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 16, 8, zF)
                @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 8, 16, zF)
                @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 8, 11, zF)
                @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 5,  8, zF)
                @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 7, 13, zF)
            end
        end
    end
end
