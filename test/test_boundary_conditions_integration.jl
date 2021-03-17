using Oceananigans.BoundaryConditions: ContinuousBoundaryFunction

function test_boundary_condition(arch, FT, topo, side, field_name, boundary_condition)
    grid = RegularRectilinearGrid(FT, size=(1, 1, 1), extent=(1, π, 42), topology=topo)

    boundary_condition_kwarg = Dict(side => boundary_condition)
    field_boundary_conditions = TracerBoundaryConditions(grid; boundary_condition_kwarg...)
    bcs = NamedTuple{(field_name,)}((field_boundary_conditions,))

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT,
                                boundary_conditions=bcs)

    success = try
        time_step!(model, 1e-16, euler=true)
        true
    catch err
        @warn "test_boundary_condition errored with " * sprint(showerror, err)
        false
    end

    return success
end

function test_flux_budget(arch, FT, fldname)
    N, κ, Lz = 16, 1, 0.7
    grid = RegularRectilinearGrid(FT, size=(N, N, N), extent=(1, 1, Lz))

    bottom_flux = FT(0.3)
    flux_bc = BoundaryCondition(Flux, bottom_flux)

    if fldname == :u
        field_bcs = UVelocityBoundaryConditions(grid, bottom=flux_bc)
    elseif fldname == :v
        field_bcs = VVelocityBoundaryConditions(grid, bottom=flux_bc)
    else
        field_bcs = TracerBoundaryConditions(grid, bottom=flux_bc)
    end

    model_bcs = NamedTuple{(fldname,)}((field_bcs,))

    closure = IsotropicDiffusivity(FT, ν=κ, κ=κ)
    model = IncompressibleModel(grid=grid, closure=closure, architecture=arch, tracers=(:T, :S),
                                float_type=FT, buoyancy=nothing, boundary_conditions=model_bcs)

    field = get_model_field(fldname, model)
    @. field.data = 0

    τκ = Lz^2 / κ   # Diffusion time-scale
    Δt = 1e-6 * τκ  # Time step much less than diffusion time-scale
    Nt = 10         # Number of time steps

    for n in 1:Nt
        time_step!(model, Δt, euler= n==1)
    end

    # budget: Lz*∂<ϕ>/∂t = -Δflux = -top_flux/Lz (left) + bottom_flux/Lz (right)
    # therefore <ϕ> = bottom_flux * t / Lz
    return mean(interior(field)) ≈ bottom_flux * model.clock.time / Lz
end

function fluxes_with_diffusivity_boundary_conditions_are_correct(arch, FT)
    Lz = 1
    κ₀ = FT(exp(-3))
    bz = FT(π)
    flux = - κ₀ * bz

    grid = RegularRectilinearGrid(FT, size=(16, 16, 16), extent=(1, 1, Lz))

    buoyancy_bcs = TracerBoundaryConditions(grid, bottom=BoundaryCondition(Gradient, bz))
    κₑ_bcs = DiffusivityBoundaryConditions(grid, bottom=BoundaryCondition(Value, κ₀))
    model_bcs = (b=buoyancy_bcs, κₑ=(b=κₑ_bcs,))

    model = IncompressibleModel(
        grid=grid, architecture=arch, float_type=FT, tracers=:b, buoyancy=BuoyancyTracer(),
        closure=AnisotropicMinimumDissipation(), boundary_conditions=model_bcs
    )

    b₀(x, y, z) = z * bz
    set!(model, b=b₀)

    b = model.tracers.b
    mean_b₀ = mean(interior(b))

    τκ = Lz^2 / κ₀  # Diffusion time-scale
    Δt = 1e-6 * τκ  # Time step much less than diffusion time-scale
    Nt = 10         # Number of time steps

    for n in 1:Nt
        time_step!(model, Δt, euler= n==1)
    end

    # budget: Lz*∂<ϕ>/∂t = -Δflux = -top_flux/Lz (left) + bottom_flux/Lz (right)
    # therefore <ϕ> = bottom_flux * t / Lz
    #
    # Use an atol of 1e-6 so test passes with Float32 as there's a big cancellation
    # error due to buoyancy order of magnitude.
    #
    # Float32:
    # mean_b₀ = -1.5707965f0
    # mean(interior(b)) = -1.5708286f0
    # mean(interior(b)) - mean_b₀ = -3.20673f-5
    # (flux * model.clock.time) / Lz = -3.141593f-5
    #
    # Float64
    # mean_b₀ = -1.5707963267949192
    # mean(interior(b)) = -1.57082774272148
    # mean(interior(b)) - mean_b₀ = -3.141592656086267e-5
    # (flux * model.clock.time) / Lz = -3.141592653589793e-5
    
    return isapprox(mean(interior(b)) - mean_b₀, flux * model.clock.time / Lz, atol=1e-6)
end

test_boundary_conditions(C, FT, ArrayType) = (integer_bc(C, FT, ArrayType),
                                              float_bc(C, FT, ArrayType),
                                              irrational_bc(C, FT, ArrayType),
                                              array_bc(C, FT, ArrayType),
                                              simple_function_bc(C, FT, ArrayType),
                                              parameterized_function_bc(C, FT, ArrayType),
                                              field_dependent_function_bc(C, FT, ArrayType),
                                              parameterized_field_dependent_function_bc(C, FT, ArrayType),
                                              discrete_function_bc(C, FT, ArrayType),
                                              parameterized_discrete_function_bc(C, FT, ArrayType)
                                             )

@testset "Boundary condition integration tests" begin
    @info "Testing boundary condition integration into IncompressibleModel..."

    #=
    @testset "Boundary condition regularization" begin
        @info "  Testing boundary condition regularization in IncompressibleModel constructor..."

        FT = Float64
        arch = first(archs)

        # We use Periodic, Bounded, Bounded here because triply Bounded domains don't work on the GPU
        # yet.
        grid = RegularRectilinearGrid(FT, size=(1, 1, 1), extent=(1, π, 42), topology=(Periodic, Bounded, Bounded))

        u_boundary_conditions = UVelocityBoundaryConditions(grid; 
                                                            bottom = simple_function_bc(Value),
                                                            top    = simple_function_bc(Value),
                                                            north  = simple_function_bc(Value),
                                                            south  = simple_function_bc(Value))

        v_boundary_conditions = VVelocityBoundaryConditions(grid;
                                                            bottom = simple_function_bc(Value),
                                                            top    = simple_function_bc(Value),
                                                            north  = simple_function_bc(NormalFlow),
                                                            south  = simple_function_bc(NormalFlow))


        w_boundary_conditions = VVelocityBoundaryConditions(grid;
                                                            bottom = simple_function_bc(NormalFlow),
                                                            top    = simple_function_bc(NormalFlow),
                                                            north  = simple_function_bc(Value),
                                                            south  = simple_function_bc(Value))

        T_boundary_conditions = TracerBoundaryConditions(grid;
                                                         bottom = simple_function_bc(Value),
                                                         top    = simple_function_bc(Value),
                                                         north  = simple_function_bc(Value),
                                                         south  = simple_function_bc(Value))

        boundary_conditions = (u=u_boundary_conditions,
                               v=v_boundary_conditions,
                               w=w_boundary_conditions,
                               T=T_boundary_conditions)

        model = IncompressibleModel(architecture = arch,
                                    grid = grid,
                                    float_type = FT,
                                    boundary_conditions = boundary_conditions)

        @test location(model.velocities.u.boundary_conditions.bottom.condition) == (Face, Center, Nothing)
        @test location(model.velocities.u.boundary_conditions.top.condition)    == (Face, Center, Nothing)
        @test location(model.velocities.u.boundary_conditions.north.condition)  == (Face, Nothing, Center)
        @test location(model.velocities.u.boundary_conditions.south.condition)  == (Face, Nothing, Center)

        @test location(model.velocities.v.boundary_conditions.bottom.condition) == (Center, Face, Nothing)
        @test location(model.velocities.v.boundary_conditions.top.condition)    == (Center, Face, Nothing)
        @test location(model.velocities.v.boundary_conditions.north.condition)  == (Center, Nothing, Center)
        @test location(model.velocities.v.boundary_conditions.south.condition)  == (Center, Nothing, Center)

        @test location(model.velocities.w.boundary_conditions.bottom.condition) == (Center, Center, Nothing)
        @test location(model.velocities.w.boundary_conditions.top.condition)    == (Center, Center, Nothing)
        @test location(model.velocities.w.boundary_conditions.north.condition)  == (Center, Nothing, Face)
        @test location(model.velocities.w.boundary_conditions.south.condition)  == (Center, Nothing, Face)

        @test location(model.tracers.T.boundary_conditions.bottom.condition) == (Center, Center, Nothing)
        @test location(model.tracers.T.boundary_conditions.top.condition)    == (Center, Center, Nothing)
        @test location(model.tracers.T.boundary_conditions.north.condition)  == (Center, Nothing, Center)
        @test location(model.tracers.T.boundary_conditions.south.condition)  == (Center, Nothing, Center)
    end
    =#

    @testset "Boudnary condition time-stepping works" begin
        for arch in archs, FT in (Float64,) #float_types
            @info "  Testing that time-stepping with boundary conditions works [$(typeof(arch)), $FT]..."

            topo = arch isa CPU ? (Bounded, Bounded, Bounded) : (Periodic, Bounded, Bounded)

            for C in (Gradient, Flux, Value), boundary_condition in test_boundary_conditions(C, FT, array_type(arch))
                arch isa CPU && @test test_boundary_condition(arch, FT, topo, :east, :T, boundary_condition)

                @test test_boundary_condition(arch, FT, topo, :south, :T, boundary_condition)
                @test test_boundary_condition(arch, FT, topo, :top, :T, boundary_condition)
            end

            for boundary_condition in test_boundary_conditions(NormalFlow, FT, array_type(arch))
                 arch isa CPU && @test test_boundary_condition(arch, FT, topo, :east, :u, boundary_condition)

                @test test_boundary_condition(arch, FT, topo, :south, :v, boundary_condition)
                @test test_boundary_condition(arch, FT, topo, :top, :w, boundary_condition)
            end
        end
    end

    #=
    @testset "Budgets with Flux boundary conditions" begin
        for arch in archs, FT in float_types
            @info "  Testing budgets with Flux boundary conditions on u, v, T [$(typeof(arch)), $FT]..."
            for field_name in (:u, :v, :T)
                @test test_flux_budget(arch, FT, field_name)
            end
        end
    end

    @testset "Custom diffusivity boundary conditions" begin
        for arch in archs, FT in (Float64,) #float_types
            @info "  Testing flux budgets with diffusivity boundary conditions [$(typeof(arch)), $FT]..."
            @test fluxes_with_diffusivity_boundary_conditions_are_correct(arch, FT)
        end
    end
    =#
end
