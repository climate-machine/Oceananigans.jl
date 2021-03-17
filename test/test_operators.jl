function test_function_differentiation(T=Float64)
    grid = RegularRectilinearGrid(T; size=(3, 3, 3), extent=(3, 3, 3))
    ϕ = rand(T, 3, 3, 3)
    ϕ² = ϕ.^2

    ∂x_ϕ_f = ϕ²[2, 2, 2] - ϕ²[1, 2, 2]
    ∂x_ϕ_c = ϕ²[3, 2, 2] - ϕ²[2, 2, 2]

    ∂y_ϕ_f = ϕ²[2, 2, 2] - ϕ²[2, 1, 2]
    ∂y_ϕ_c = ϕ²[2, 3, 2] - ϕ²[2, 2, 2]

    ∂z_ϕ_f = ϕ²[2, 2, 2] - ϕ²[2, 2, 1]
    ∂z_ϕ_c = ϕ²[2, 2, 3] - ϕ²[2, 2, 2]

    f(i, j, k, grid, ϕ) = ϕ[i, j, k]^2

    return (
        ∂xᶜᵃᵃ(2, 2, 2, grid, f, ϕ) == ∂x_ϕ_c &&
        ∂xᶠᵃᵃ(2, 2, 2, grid, f, ϕ) == ∂x_ϕ_f &&

        ∂yᵃᶜᵃ(2, 2, 2, grid, f, ϕ) == ∂y_ϕ_c &&
        ∂yᵃᶠᵃ(2, 2, 2, grid, f, ϕ) == ∂y_ϕ_f &&

        ∂zᵃᵃᶜ(2, 2, 2, grid, f, ϕ) == ∂z_ϕ_c &&
        ∂zᵃᵃᶠ(2, 2, 2, grid, f, ϕ) == ∂z_ϕ_f
    )
end

function test_function_interpolation(T=Float64)
    grid = RegularRectilinearGrid(T; size=(3, 3, 3), extent=(3, 3, 3))
    ϕ = rand(T, 3, 3, 3)
    ϕ² = ϕ.^2

    ℑx_ϕ_f = (ϕ²[2, 2, 2] + ϕ²[1, 2, 2]) / 2
    ℑx_ϕ_c = (ϕ²[3, 2, 2] + ϕ²[2, 2, 2]) / 2

    ℑy_ϕ_f = (ϕ²[2, 2, 2] + ϕ²[2, 1, 2]) / 2
    ℑy_ϕ_c = (ϕ²[2, 3, 2] + ϕ²[2, 2, 2]) / 2

    ℑz_ϕ_f = (ϕ²[2, 2, 2] + ϕ²[2, 2, 1]) / 2
    ℑz_ϕ_c = (ϕ²[2, 2, 3] + ϕ²[2, 2, 2]) / 2

    f(i, j, k, grid, ϕ) = ϕ[i, j, k]^2

    return (
        ℑxᶜᵃᵃ(2, 2, 2, grid, f, ϕ) == ℑx_ϕ_c &&
        ℑxᶠᵃᵃ(2, 2, 2, grid, f, ϕ) == ℑx_ϕ_f &&

        ℑyᵃᶜᵃ(2, 2, 2, grid, f, ϕ) == ℑy_ϕ_c &&
        ℑyᵃᶠᵃ(2, 2, 2, grid, f, ϕ) == ℑy_ϕ_f &&

        ℑzᵃᵃᶜ(2, 2, 2, grid, f, ϕ) == ℑz_ϕ_c &&
        ℑzᵃᵃᶠ(2, 2, 2, grid, f, ϕ) == ℑz_ϕ_f
    )
end

@testset "Operators" begin
    @info "Testing operators..."

    @testset "Grid lengths, areas, and volume operators" begin
        @info "  Testing grid lengths, areas, and volume operators..."

        FT = Float64
        grid = RegularRectilinearGrid(FT, size=(1, 1, 1), extent=(π, 2π, 3π))

        @testset "Easterly lengths" begin
            @info "    Testing easterly lengths..."
            for δ in (Δx, Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ) 
                @test δ(1, 1, 1, grid) == FT(π)
            end
        end

        @testset "Westerly lengths" begin
            @info "    Testing westerly lengths..."
            for δ in (Δy, Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ) 
                @test δ(1, 1, 1, grid) == FT(2π)
            end
        end

        @testset "Vertical lengths" begin
            @info "    Testing vertical lengths..."
            for δ in (ΔzF, ΔzC)
                @test δ(1, 1, 1, grid) == FT(3π)
            end
        end

        @testset "East-normal areas in the yz-plane" begin
            @info "    Testing areas with easterly normal in the yz-plane..."
            for A in (Axᵃᵃᶜ, Axᵃᵃᶠ, Axᶠᶜᶜ)
                @test A(1, 1, 1, grid) == FT(6 * π^2)
            end
        end

        @testset "West-normal areas in the xz-plane" begin
            @info "    Testing areas with westerly normal in the xz-plane..."
            for A in (Ayᵃᵃᶜ, Ayᵃᵃᶠ, Ayᶜᶠᶜ)
                @test A(1, 1, 1, grid) == FT(3 * π^2)
            end
        end

        @testset "Horizontal areas in the xy-plane" begin
            @info "    Testing horizontal areas in the xy-plane..."
            for A in (Azᵃᵃᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ)
                @test A(1, 1, 1, grid) == FT(2 * π^2)
            end
        end

        @testset "Volumes" begin
            @info "    Testing volumes..."
            for V in (Vᵃᵃᶜ, Vᵃᵃᶠ, Vᶜᶜᶜ)
                @test V(1, 1, 1, grid) == FT(6 * π^3)
            end
        end

    end

    @testset "Function differentiation" begin
        @info "  Testing function differentiation..."
        @test test_function_differentiation()
    end

    @testset "Function interpolation" begin
        @info "  Testing function interpolation..."
        @test test_function_interpolation()
    end

    @testset "2D operators" begin
        @info "  Testing 2D operators..."

        Nx, Ny, Nz = 32, 16, 8
        Lx, Ly, Lz = 100, 100, 100

        arch = CPU()
        grid = RegularRectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))

        Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz
        Tx, Ty, Tz = Nx+2Hx, Ny+2Hy, Nz+2Hz

        A3 = OffsetArray(zeros(Tx, Ty, Tz), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1-Hz:Nz+Hz)
        @. @views A3[1:Nx, 1:Ny, 1:Nz] = rand()
        fill_halo_regions!(A3, TracerBoundaryConditions(grid), grid)

        # A yz-slice with Nx==1.
        A2yz = OffsetArray(zeros(1+2Hx, Ty, Tz), 1-Hx:1+Hx, 1-Hy:Ny+Hy, 1-Hz:Nz+Hz)
        grid_yz = RegularRectilinearGrid(size=(1, Ny, Nz), extent=(Lx, Ly, Lz))

        # Manually fill in halos for the slice.
        A2yz[0:2, 0:Ny+1, 1:Nz] .= A3[1:1, 0:Ny+1, 1:Nz]
        A2yz[:, :, 0] .= A2yz[:, :, 1]
        A2yz[:, :, Nz+1] .= A2yz[:, :, Nz]

        # An xz-slice with Ny==1.
        A2xz = OffsetArray(zeros(Tx, 1+2Hy, Tz), 1-Hx:Nx+Hx, 1-Hy:1+Hy, 1-Hz:Nz+Hz)
        grid_xz = RegularRectilinearGrid(size=(Nx, 1, Nz), extent=(Lx, Ly, Lz))

        # Manually fill in halos for the slice.
        A2xz[0:Nx+1, 0:2, 1:Nz] .= A3[0:Nx+1, 1:1, 1:Nz]
        A2xz[:, :, 0] .= A2xz[:, :, 1]
        A2xz[:, :, Nz+1] .= A2xz[:, :, Nz]

        test_indices_3d = [(4, 5, 5), (21, 11, 4), (16, 8, 4),  (30, 12, 3), (11, 3, 6), # Interior
                           (2, 10, 4), (31, 5, 6), (10, 2, 4), (17, 15, 5), (17, 10, 2), (23, 5, 7),  # Borderlands
                           (1, 5, 5), (32, 10, 3), (16, 1, 4), (16, 16, 4), (16, 8, 1), (16, 8, 8),  # Edges
                           (1, 1, 1), (32, 16, 8)] # Corners

        test_indices_2d_yz = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2),
                              (1, 1, 5), (1, 5, 1), (1, 5, 5), (1, 11, 4),
                              (1, 15, 7), (1, 15, 8), (1, 16, 7), (1, 16, 8)]

        test_indices_2d_xz = [(1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2),
                              (1, 1, 5), (5, 1, 1), (5, 1, 5), (17, 1, 4),
                              (31, 1, 7), (31, 1, 8), (32, 1, 7), (32, 1, 8)]

        for idx in test_indices_2d_yz
            @test δxᶜᵃᵃ(idx..., grid_yz, A2yz) ≈ 0
            @test δxᶠᵃᵃ(idx..., grid_yz, A2yz) ≈ 0
            @test δyᵃᶜᵃ(idx..., grid_yz, A2yz) ≈ δyᵃᶜᵃ(idx..., grid_yz, A3)
            @test δyᵃᶠᵃ(idx..., grid_yz, A2yz) ≈ δyᵃᶠᵃ(idx..., grid_yz, A3)
            @test δzᵃᵃᶜ(idx..., grid_yz, A2yz) ≈ δzᵃᵃᶜ(idx..., grid_yz, A3)
            @test δzᵃᵃᶠ(idx..., grid_yz, A2yz) ≈ δzᵃᵃᶠ(idx..., grid_yz, A3)
        end

        for idx in test_indices_2d_xz
            @test δxᶜᵃᵃ(idx..., grid_xz, A2xz) ≈ δxᶜᵃᵃ(idx..., grid_xz, A3)
            @test δxᶠᵃᵃ(idx..., grid_xz, A2xz) ≈ δxᶠᵃᵃ(idx..., grid_xz, A3)
            @test δyᵃᶜᵃ(idx..., grid_xz, A2xz) ≈ 0
            @test δyᵃᶠᵃ(idx..., grid_xz, A2xz) ≈ 0
            @test δzᵃᵃᶜ(idx..., grid_xz, A2xz) ≈ δzᵃᵃᶜ(idx..., grid_xz, A3)
            @test δzᵃᵃᶠ(idx..., grid_xz, A2xz) ≈ δzᵃᵃᶠ(idx..., grid_xz, A3)
        end
    end
end
