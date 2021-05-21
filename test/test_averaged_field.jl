using Statistics

using Oceananigans.Fields: CenterField, ZFaceField, compute_at!
using Oceananigans.Grids: halo_size

@testset "Averaged fields" begin
    @info "Testing averaged fields..."

    for arch in archs
        @testset "Averaged fields [$(typeof(arch))]" begin
            @info "  Testing Averaged fields [$(typeof(arch))]"
            for FT in float_types

                grid = RegularRectilinearGrid(FT, topology = (Periodic, Periodic, Bounded),
                                              size = (2, 2, 2),
                                                 x = (0, 2), y = (0, 2), z = (0, 2))

                w = ZFaceField(arch, grid)
                T = CenterField(arch, grid)

                trilinear(x, y, z) = x + y + z

                set!(T, trilinear)
                set!(w, trilinear)

                @compute T̃ = AveragedField(T, dims=(1, 2, 3))

                # Note: halo regions must be *filled* prior to computing an average
                # if the average within halo regions is to be correct.
                fill_halo_regions!(T, arch)
                @compute T̅ = AveragedField(T, dims=(1, 2))

                fill_halo_regions!(T, arch)
                @compute T̂ = AveragedField(T, dims=1)

                @compute w̃ = AveragedField(w, dims=(1, 2, 3))
                @compute w̅ = AveragedField(w, dims=(1, 2))
                @compute ŵ = AveragedField(w, dims=1)

                Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

                @test T̃[1, 1, 1] ≈ 3

                @test Array(interior(T̅))[1, 1, :] ≈ [2.5, 3.5]
                @test Array(interior(T̂))[1, :, :] ≈ [[2, 3] [3, 4]]

                @test w̃[1, 1, 1] ≈ 3

                @test Array(interior(w̅))[1, 1, :] ≈ [2, 3, 4]
                @test Array(interior(ŵ))[1, :, :] ≈ [[1.5, 2.5] [2.5, 3.5] [3.5, 4.5]]

                bigger_grid = RegularRectilinearGrid(FT, topology = (Periodic, Periodic, Bounded),
                                                     size = (32, 32, 32),
                                                        x = (0, 2), y = (0, 2), z = (0, 2))

                # https://github.com/CliMA/Oceananigans.jl/issues/1684
                c = CenterField(arch, bigger_grid)
                C = AveragedField(c, dims=(1, 2))
                compute!(C)
            end
        end

        @testset "Conditional computation of AveragedFields [$(typeof(arch))]" begin
            @info "  Testing conditional computation of AveragedFields [$(typeof(arch))]"
            for FT in float_types
                grid = RegularRectilinearGrid(FT, size=(2, 2, 2), extent=(1, 1, 1))
                c = CenterField(arch, grid)

                for dims in (1, 2, 3, (1, 2), (2, 3), (1, 3), (1, 2, 3))
                    C = AveragedField(c, dims=dims)

                    @test !isnothing(C.status)

                    # Test conditional computation
                    set!(c, 1)
                    compute_at!(C, FT(1)) # will compute
                    @test all(interior(C) .== 1)
                    @test C.status.time == FT(1)

                    set!(c, 2)
                    compute_at!(C, FT(1)) # will not compute because status == 1
                    @test C.status.time == FT(1)
                    @test all(interior(C) .== 1)

                    compute_at!(C, FT(2))
                    @test C.status.time == FT(2)
                    @test all(interior(C) .== 2)
                end
            end
        end
    end
end
