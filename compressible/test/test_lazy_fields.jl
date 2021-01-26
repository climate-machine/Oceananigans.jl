using JULES: LazyVelocityFields, LazyTracerFields

@testset "Lazy fields" begin
    @info "Testing lazy fields..."

    grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
    model = CompressibleModel(grid=grid, gases=DryEarth(),
                              thermodynamic_variable=Energy())

    model.total_density.data .= π
    model.momenta.ρu.data .= 1.0
    model.momenta.ρv.data .= 2.0
    model.momenta.ρw.data .= 3.0
    model.tracers.ρe.data .= 4.0

    velocities = LazyVelocityFields(model.architecture, model.grid, model.total_density, model.momenta)
    primitive_tracers = LazyTracerFields(model.architecture, model.grid, model.total_density, model.tracers)

    @test velocities.u[1, 2, 3] == 1/π
    @test velocities.v[4, 5, 6] == 2/π
    @test velocities.w[7, 8, 9] == 3/π

    @test primitive_tracers.e[10, 11, 12] == 4/π

    grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
    model = CompressibleModel(grid=grid, gases=DryEarth(),
                              thermodynamic_variable=Entropy())

    @. model.total_density.data = randn()
    @. model.momenta.ρu.data = randn()
    @. model.momenta.ρv.data = randn()
    @. model.momenta.ρw.data = randn()
    @. model.tracers.ρs.data = randn()

    velocities = LazyVelocityFields(model.architecture, model.grid, model.total_density, model.momenta)
    primitive_tracers = LazyTracerFields(model.architecture, model.grid, model.total_density, model.tracers)

    ρ = model.total_density
    ρu, ρv, ρw = model.momenta
    @test velocities.u[1, 2, 3] == ρu[1, 2, 3] / ((ρ[0, 2, 3] + ρ[1, 2, 3]) / 2)
    @test velocities.v[4, 5, 6] == ρv[4, 5, 6] / ((ρ[4, 4, 6] + ρ[4, 5, 6]) / 2)
    @test velocities.w[7, 8, 9] == ρw[7, 8, 9] / ((ρ[7, 8, 8] + ρ[7, 8, 9]) / 2)

    ρs = model.tracers.ρs
    @test primitive_tracers.s[10, 11, 12] == ρs[10, 11, 12] / ρ[10, 11, 12]
end
