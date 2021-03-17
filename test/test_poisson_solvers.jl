using Oceananigans.Solvers: solve_for_pressure!, solve_poisson_equation!
using Oceananigans.Solvers: poisson_eigenvalues
using Oceananigans.Models.HydrostaticFreeSurfaceModels: _compute_w_from_continuity!

function poisson_solver_instantiates(arch, FT, Nx, Ny, Nz, planner_flag)
    grid = RegularRectilinearGrid(FT, size=(Nx, Ny, Nz), extent=(100, 100, 100))
    solver = FFTBasedPoissonSolver(arch, grid, planner_flag)
    return true  # Just making sure the FFTBasedPoissonSolver does not error/crash.
end

function random_divergent_source_term(FT, arch, grid)
    Ru = CenterField(FT, arch, grid, UVelocityBoundaryConditions(grid))
    Rv = CenterField(FT, arch, grid, VVelocityBoundaryConditions(grid))
    Rw = CenterField(FT, arch, grid, WVelocityBoundaryConditions(grid))
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, rand(Nx, Ny, Nz))
    set!(Rv, rand(Nx, Ny, Nz))
    set!(Rw, rand(Nx, Ny, Nz))

    # Adding (nothing, nothing) in case we need to dispatch on ::NFBC
    fill_halo_regions!(Ru)
    fill_halo_regions!(Rv)
    fill_halo_regions!(Rw)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    event = launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    return R, U
end

function random_divergence_free_source_term(FT, arch, grid)
    # Random right hand side
    Ru = CenterField(FT, arch, grid, UVelocityBoundaryConditions(grid))
    Rv = CenterField(FT, arch, grid, VVelocityBoundaryConditions(grid))
    Rw = CenterField(FT, arch, grid, WVelocityBoundaryConditions(grid))
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, rand(Nx, Ny, Nz))
    set!(Rv, rand(Nx, Ny, Nz))
    set!(Rw, zeros(Nx, Ny, Nz))

    fill_halo_regions!(Ru)
    fill_halo_regions!(Rv)
    fill_halo_regions!(Rw)

    event = launch!(arch, grid, :xy, _compute_w_from_continuity!, U, grid,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    fill_halo_regions!(Rw)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    event = launch!(arch, grid, :xyz, divergence!, grid, Ru.data, Rv.data, Rw.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    return R
end

function compute_∇²!(∇²ϕ, ϕ, arch, grid)
    fill_halo_regions!(ϕ)
    event = launch!(arch, grid, :xyz, ∇²!, grid, ϕ.data, ∇²ϕ.data, dependencies=Event(device(arch)))
    wait(device(arch), event)
    fill_halo_regions!(∇²ϕ)
    return nothing
end

#####
##### Regular rectilinear grid Poisson solver
#####

function divergence_free_poisson_solution(arch, FT, topo, Nx, Ny, Nz, planner_flag=FFTW.MEASURE)
    ArrayType = array_type(arch)
    grid = RegularRectilinearGrid(FT, topology=topo, size=(Nx, Ny, Nz), extent=(1, 1, 1))

    solver = FFTBasedPoissonSolver(arch, grid, planner_flag)
    R, U = random_divergent_source_term(FT, arch, grid)

    p_bcs = PressureBoundaryConditions(grid)
    ϕ   = CenterField(FT, arch, grid, p_bcs)  # "kinematic pressure"
    ∇²ϕ = CenterField(FT, arch, grid, p_bcs)

    # Using Δt = 1 but it doesn't matter since velocities = 0.
    solve_for_pressure!(ϕ.data, solver, arch, grid, 1, datatuple(U))

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

    solve_poisson_equation!(solver)

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

function vertically_stretched_poisson_solver_correct_answer(FT, arch, topo, Nx, Ny, zF)
    Nz = length(zF) - 1
    vs_grid = VerticallyStretchedRectilinearGrid(FT, architecture=arch, topology=topo, size=(Nx, Ny, Nz), x=(0, 1), y=(0, 1), zF=zF)
    solver = FourierTridiagonalPoissonSolver(arch, vs_grid)

    p_bcs = PressureBoundaryConditions(vs_grid)
    ϕ   = CenterField(FT, arch, vs_grid, p_bcs)  # "kinematic pressure"
    ∇²ϕ = CenterField(FT, arch, vs_grid, p_bcs)

    R = random_divergence_free_source_term(FT, arch, vs_grid)
    F = CUDA.@allowscalar reshape(vs_grid.Δzᵃᵃᶠ[1:Nz], 1, 1, Nz) .* R  # RHS needs to be multiplied by ΔzC
    solver.batched_tridiagonal_solver.f .= F

    solve_poisson_equation!(solver)

    CUDA.@allowscalar interior(ϕ) .= real.(solver.storage)
    compute_∇²!(∇²ϕ, ϕ, arch, vs_grid)

    return CUDA.@allowscalar interior(∇²ϕ) ≈ R
end

#####
##### Run pressure solver tests
#####

PB = (Periodic, Bounded)
topos = collect(Iterators.product(PB, PB, PB))[:]

@testset "Poisson solvers" begin
    @info "Testing Poisson solvers..."

    for arch in archs
        @testset "Poisson solver instantiation [$(typeof(arch))]" begin
            @info "  Testing Poisson solver instantiation [$(typeof(arch))]..."
            for FT in float_types
                @test poisson_solver_instantiates(arch, FT, 32, 32, 32, FFTW.ESTIMATE)
                @test poisson_solver_instantiates(arch, FT, 1,  32, 32, FFTW.MEASURE)
                @test poisson_solver_instantiates(arch, FT, 32,  1, 32, FFTW.ESTIMATE)
                @test poisson_solver_instantiates(arch, FT,  1,  1, 32, FFTW.MEASURE)
            end
        end

        @testset "Divergence-free solution [$(typeof(arch))]" begin
            @info "  Testing divergence-free solution [$(typeof(arch))]..."

            for topo in topos
                @info "    Testing $topo topology on square grids [$(typeof(arch))]..."
                for N in [7, 16]
                    @test divergence_free_poisson_solution(arch, Float64, topo, N, N, N)
                    @test divergence_free_poisson_solution(arch, Float64, topo, 1, N, N)
                    @test divergence_free_poisson_solution(arch, Float64, topo, N, 1, N)
                    @test divergence_free_poisson_solution(arch, Float64, topo, 1, 1, N)
                end
            end

            Ns = [11, 16]
            for topo in topos
                @info "    Testing $topo topology on rectangular grids with even and prime sizes [$(typeof(arch))]..."
                for Nx in Ns, Ny in Ns, Nz in Ns
                    @test divergence_free_poisson_solution(arch, Float64, topo, Nx, Ny, Nz)
                end
            end

            # Do a couple at Float32 (kinda expensive to repeat all tests...)
            @test divergence_free_poisson_solution(arch, Float32, (Periodic, Bounded, Periodic), 16, 16, 16)
            @test divergence_free_poisson_solution(arch, Float32, (Bounded, Periodic, Bounded), 7,  11, 13)
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
        (Bounded,  Bounded,  Bounded)
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
