"""
    correct_reduced_field_size(loc, arch, grid, dims, Tx, Ty, Tz)

Test that the ReducedField at `loc`ation on `arch`itecture and `grid`
and reduced along `dims` has size `(Tx, Ty, Tz)`.
"""
correct_reduced_field_size(loc, arch, grid, dims, Tx, Ty, Tz) =
    size(parent(ReducedField(loc, arch, grid; dims=dims))) == (Tx, Ty, Tz)

function correct_reduced_field_value_was_set(arch, grid, loc, dims, val::Number)
    f = ReducedField(loc, arch, grid; dims=dims)
    set!(f, val)
    CUDA.@allowscalar return interior(f) ≈ val * ones(size(f))
end

@testset "ReducedFields" begin
    @info "Testing ReducedFields..."

    N = (4, 6, 8)
    L = (2π, 3π, 5π)
    H = (1, 1, 1)

    @testset "ReducedField initialization" begin
        @info "  Testing ReducedField initialization..."
        for arch in archs, FT in float_types

            grid = RegularRectilinearGrid(FT, size=N, extent=L, halo=H, topology=(Bounded, Bounded, Bounded))

            @test correct_reduced_field_size((Center, Center, Center), arch, grid, 1,         1,               N[2] + 2 * H[2],     N[3] + 2 * H[3])
            @test correct_reduced_field_size((Face,   Center, Center), arch, grid, 1,         1,               N[2] + 2 * H[2],     N[3] + 2 * H[3])
            @test correct_reduced_field_size((Center, Face,   Center), arch, grid, 1,         1,               N[2] + 2 * H[2] + 1, N[3] + 2 * H[3])
            @test correct_reduced_field_size((Center, Face,   Face),   arch, grid, 1,         1,               N[2] + 2 * H[2] + 1, N[3] + 2 * H[3] + 1)
            @test correct_reduced_field_size((Center, Center, Center), arch, grid, 2,         N[1] + 2 * H[1], 1,                   N[3] + 2 * H[3])
            @test correct_reduced_field_size((Center, Center, Center), arch, grid, 2,         N[1] + 2 * H[1], 1,                   N[3] + 2 * H[3])
            @test correct_reduced_field_size((Center, Center, Center), arch, grid, 3,         N[1] + 2 * H[1], N[2] + 2 * H[2],     1)
            @test correct_reduced_field_size((Center, Center, Center), arch, grid, (1, 2),    1,               1,                   N[3] + 2 * H[3])
            @test correct_reduced_field_size((Center, Center, Center), arch, grid, (2, 3),    N[1] + 2 * H[1], 1,                   1)
            @test correct_reduced_field_size((Center, Center, Center), arch, grid, (1, 2, 3), 1,               1,                   1)
        end
    end

    reduced_dims = (1, 2, 3, (1, 2), (2, 3), (1, 3), (1, 2, 3))

    int_vals = Any[0, Int8(-1), Int16(2), Int32(-3), Int64(4), Int128(-5)]
    uint_vals = Any[6, UInt8(7), UInt16(8), UInt32(9), UInt64(10), UInt128(11)]
    float_vals = Any[0.0, -0.0, 6e-34, 1.0f10]
    rational_vals = Any[1//11, -23//7]
    other_vals = Any[π]
    vals = vcat(int_vals, uint_vals, float_vals, rational_vals, other_vals)

    @testset "Setting ReducedFields" begin
        @info "  Testing ReducedField setting..."
        for arch in archs, FT in float_types

            grid = RegularRectilinearGrid(FT, size=N, extent=L, halo=H, topology=(Periodic, Periodic, Bounded))

            for dims in reduced_dims, val in vals
                @test correct_reduced_field_value_was_set(arch, grid, (Center, Center, Center), dims, val)
            end

            for dims in reduced_dims
                reduced_field = ReducedField((Center, Center, Center), arch, grid, dims=dims)
                sz = size(reduced_field)
                A = rand(FT, sz...) |> array_type(arch)
                set!(reduced_field, A)
                @test reduced_field[1, 1, 1] == A[1, 1, 1]

                fill_halo_regions!(reduced_field)

                # No-flux boundary conditions at top and bottom
                @test reduced_field[1, 1, 0] == A[1, 1, 1]
                @test reduced_field[1, 1, grid.Nz+1] == A[1, 1, end]

                # Periodic boundary conditions in the horizontal directions
                @test reduced_field[1, 0, 1] == A[1, end, 1]
                @test reduced_field[1, grid.Ny+1, 1] == A[1, 1, 1]

                @test reduced_field[0, 1, 1] == A[end, 1, 1]
                @test reduced_field[grid.Nx+1, 1, 1] == A[1, 1, 1]
            end
        end
    end
end


