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
    true_model = IncompressibleModel(architecture=arch, grid=grid, closure=closure)

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

    restored_model = IncompressibleModel(architecture=arch, grid=grid, closure=closure)
    set!(restored_model, "checkpoint_iteration5.jld2")

    for n in 1:4
        update_state!(restored_model)
        time_step!(restored_model, Δt, euler=false) # time-step for 4 iterations
    end

    test_model_equality(restored_model, true_model)

    #####
    ##### Test `set!(model, checkpoint_file)`
    #####

    new_model = IncompressibleModel(architecture=arch, grid=grid, closure=closure)

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

function test_checkpoint_output_with_function_bcs(arch)
    grid = RegularRectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))

    @inline some_flux(x, y, t) = 2x + exp(y)
    top_u_bc = top_T_bc = FluxBoundaryCondition(some_flux)
    u_bcs = UVelocityBoundaryConditions(grid, top=top_u_bc)
    T_bcs = TracerBoundaryConditions(grid, top=top_T_bc)

    model = IncompressibleModel(architecture=arch, grid=grid, boundary_conditions=(u=u_bcs, T=T_bcs))
    set!(model, u=π/2, v=ℯ, T=Base.MathConstants.γ, S=Base.MathConstants.φ)

    checkpointer = Checkpointer(model, schedule=IterationInterval(1))
    write_output!(checkpointer, model)
    model = nothing

    restored_model = IncompressibleModel(architecture=arch, grid=grid, boundary_conditions=(u=u_bcs, T=T_bcs))
    set!(restored_model, "checkpoint_iteration0.jld2")

    CUDA.@allowscalar begin
        @test all(interior(restored_model.velocities.u) .≈ π/2)
        @test all(interior(restored_model.velocities.v) .≈ ℯ)
        @test all(interior(restored_model.velocities.w) .== 0)
        @test all(interior(restored_model.tracers.T) .≈ Base.MathConstants.γ)
        @test all(interior(restored_model.tracers.S) .≈ Base.MathConstants.φ)
    end
    restored_model = nothing

    properly_restored_model = IncompressibleModel(architecture=arch, grid=grid, boundary_conditions=(u=u_bcs, T=T_bcs))
    set!(properly_restored_model, "checkpoint_iteration0.jld2")

    CUDA.@allowscalar begin
        @test all(interior(properly_restored_model.velocities.u) .≈ π/2)
        @test all(interior(properly_restored_model.velocities.v) .≈ ℯ)
        @test all(interior(properly_restored_model.velocities.w) .== 0)
        @test all(interior(properly_restored_model.tracers.T) .≈ Base.MathConstants.γ)
        @test all(interior(properly_restored_model.tracers.S) .≈ Base.MathConstants.φ)
    end

    u, v, w = properly_restored_model.velocities
    T, S = properly_restored_model.tracers

    @test u.boundary_conditions.x.left  isa PBC
    @test u.boundary_conditions.x.right isa PBC
    @test u.boundary_conditions.y.left  isa PBC
    @test u.boundary_conditions.y.right isa PBC
    @test u.boundary_conditions.z.left  isa ZFBC
    @test u.boundary_conditions.z.right isa FBC
    @test u.boundary_conditions.z.right.condition isa ContinuousBoundaryFunction
    @test u.boundary_conditions.z.right.condition.func(1, 2, 3) == some_flux(1, 2, 3)

    @test T.boundary_conditions.x.left  isa PBC
    @test T.boundary_conditions.x.right isa PBC
    @test T.boundary_conditions.y.left  isa PBC
    @test T.boundary_conditions.y.right isa PBC
    @test T.boundary_conditions.z.left  isa ZFBC
    @test T.boundary_conditions.z.right isa FBC
    @test T.boundary_conditions.z.right.condition isa ContinuousBoundaryFunction
    @test T.boundary_conditions.z.right.condition.func(1, 2, 3) == some_flux(1, 2, 3)

    # Test that the restored model can be time stepped
    time_step!(properly_restored_model, 1)
    @test properly_restored_model isa IncompressibleModel

    return nothing
end

function run_cross_architecture_checkpointer_tests(arch1, arch2)
    grid = RegularRectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch1, grid=grid)
    set!(model, u=π/2, v=ℯ, T=Base.MathConstants.γ, S=Base.MathConstants.φ)

    checkpointer = Checkpointer(model, schedule=IterationInterval(1))
    write_output!(checkpointer, model)
    model = nothing

    restored_model = IncompressibleModel(architecture=arch2, grid=grid)
    set!(restored_model, "checkpoint_iteration0.jld2")

    @test restored_model.architecture == arch2

    ArrayType = array_type(restored_model.architecture)
    CUDA.@allowscalar begin
        @test restored_model.velocities.u.data.parent isa ArrayType
        @test restored_model.velocities.v.data.parent isa ArrayType
        @test restored_model.velocities.w.data.parent isa ArrayType
        @test restored_model.tracers.T.data.parent isa ArrayType
        @test restored_model.tracers.S.data.parent isa ArrayType

        @test all(interior(restored_model.velocities.u) .≈ π/2)
        @test all(interior(restored_model.velocities.v) .≈ ℯ)
        @test all(interior(restored_model.velocities.w) .== 0)
        @test all(interior(restored_model.tracers.T) .≈ Base.MathConstants.γ)
        @test all(interior(restored_model.tracers.S) .≈ Base.MathConstants.φ)
    end

    # Test that the restored model can be time stepped
    time_step!(restored_model, 1)
    @test restored_model isa IncompressibleModel

    return nothing
end

function run_checkpointer_cleanup_tests(arch)
    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch, grid=grid)
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
        test_checkpoint_output_with_function_bcs(arch)

        if CUDA.has_cuda()
            run_cross_architecture_checkpointer_tests(CPU(), GPU())
            run_cross_architecture_checkpointer_tests(GPU(), CPU())
        end

        run_checkpointer_cleanup_tests(arch)
    end
end
