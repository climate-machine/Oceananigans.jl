#####
##### Checkpointer tests
#####

function test_model_equality(test_model, true_model)
    CUDA.@allowscalar begin
        @test all(test_model.velocities.u.data     .≈ true_model.velocities.u.data)
        @test all(test_model.velocities.v.data     .≈ true_model.velocities.v.data)
        @test all(test_model.velocities.w.data     .≈ true_model.velocities.w.data)
        @test all(test_model.tracers.T.data        .≈ true_model.tracers.T.data)
        @test all(test_model.tracers.S.data        .≈ true_model.tracers.S.data)
        @test all(test_model.timestepper.Gⁿ.u.data .≈ true_model.timestepper.Gⁿ.u.data)
        @test all(test_model.timestepper.Gⁿ.v.data .≈ true_model.timestepper.Gⁿ.v.data)
        @test all(test_model.timestepper.Gⁿ.w.data .≈ true_model.timestepper.Gⁿ.w.data)
        @test all(test_model.timestepper.Gⁿ.T.data .≈ true_model.timestepper.Gⁿ.T.data)
        @test all(test_model.timestepper.Gⁿ.S.data .≈ true_model.timestepper.Gⁿ.S.data)
        @test all(test_model.timestepper.G⁻.u.data .≈ true_model.timestepper.G⁻.u.data)
        @test all(test_model.timestepper.G⁻.v.data .≈ true_model.timestepper.G⁻.v.data)
        @test all(test_model.timestepper.G⁻.w.data .≈ true_model.timestepper.G⁻.w.data)
        @test all(test_model.timestepper.G⁻.T.data .≈ true_model.timestepper.G⁻.T.data)
        @test all(test_model.timestepper.G⁻.S.data .≈ true_model.timestepper.G⁻.S.data)
    end
    return nothing
end

"""
Run two coarse rising thermal bubble simulations and make sure

1. When restarting from a checkpoint, the restarted model matches the non-restarted
   model to machine precision.

2. When using set!(new_model) to a checkpoint, the new model matches the non-restarted
   simulation to machine precision.

3. run!(new_model, pickup) works as expected
"""
function test_thermal_bubble_checkpointer_output(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    grid = RegularRectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    closure = IsotropicDiffusivity(ν=4e-2, κ=4e-2)
    true_model = NonhydrostaticModel(architecture=arch, grid=grid, closure=closure)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    CUDA.@allowscalar true_model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    checkpointed_model = deepcopy(true_model)

    true_simulation = Simulation(true_model, Δt=Δt, stop_iteration=9)
    run!(true_simulation) # for 9 iterations

    checkpointed_simulation = Simulation(checkpointed_model, Δt=Δt, stop_iteration=5)
    checkpointer = Checkpointer(checkpointed_model, schedule=IterationInterval(5), force=true)
    push!(checkpointed_simulation.output_writers, checkpointer)

    # Checkpoint should be saved as "checkpoint_iteration5.jld" after the 5th iteration.
    run!(checkpointed_simulation) # for 5 iterations

    #####
    ##### Test `set!(model, checkpoint_file)`
    #####

    new_model = NonhydrostaticModel(architecture=arch, grid=grid, closure=closure)

    set!(new_model, "checkpoint_iteration5.jld2")

    @test new_model.clock.iteration == checkpointed_model.clock.iteration
    @test new_model.clock.time == checkpointed_model.clock.time
    test_model_equality(new_model, checkpointed_model)

    #####
    ##### Test `run!(sim, pickup=true)
    #####

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=9)

    # Pickup from explicit checkpoint path
    run!(new_simulation, pickup="checkpoint_iteration0.jld2")
    test_model_equality(new_model, true_model)

    run!(new_simulation, pickup="checkpoint_iteration5.jld2")
    test_model_equality(new_model, true_model)

    # Pickup using existing checkpointer
    new_simulation.output_writers[:checkpointer] =
        Checkpointer(new_model, schedule=IterationInterval(5), force=true)

    run!(new_simulation, pickup=true)
    test_model_equality(new_model, true_model)

    run!(new_simulation, pickup=0)
    test_model_equality(new_model, true_model)

    run!(new_simulation, pickup=5)
    test_model_equality(new_model, true_model)

    rm("checkpoint_iteration0.jld2", force=true)
    rm("checkpoint_iteration5.jld2", force=true)

    return nothing
end

function run_checkpointer_cleanup_tests(arch)
    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(architecture=arch, grid=grid)
    simulation = Simulation(model, Δt=0.2, stop_iteration=10)

    simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(3), cleanup=true)
    run!(simulation)

    [@test !isfile("checkpoint_iteration$i.jld2") for i in 1:10 if i != 9]
    @test isfile("checkpoint_iteration9.jld2")

    return nothing
end

for arch in archs
    @testset "Checkpointer [$(typeof(arch))]" begin
        @info "  Testing Checkpointer [$(typeof(arch))]..."
        test_thermal_bubble_checkpointer_output(arch)
        run_checkpointer_cleanup_tests(arch)
    end
end
