using Printf
using Test

using Oceananigans
using Oceananigans.Coriolis
using Oceananigans.Buoyancy
using Oceananigans.TurbulenceClosures
using Oceananigans.OutputWriters: IterationInterval, NetCDFOutputWriter
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface, HydrostaticFreeSurfaceModel

initial_u(x, y, z) = cos(y) * sin(x) * exp(z)
initial_v(x, y, z) = sin(x) * cos(x) * exp(z)
initial_η(x, y, z) = exp(-x^2) * exp(-z^2)
initial_T(x, y, z) = exp(-z^2) * tanh(x)
initial_S(x, y, z) = cos(y) * sin(z)

closures = (
            IsotropicDiffusivity(ν=1, κ=1),
            AnisotropicBiharmonicDiffusivity(νh=1, κh=1),
           )

coriolises = (
              FPlane(f=1),
              BetaPlane(f₀=1, β=1),
             )

grids = (
         RegularCartesianGrid(size=(16, 16, 16), x=(-2π, 2π), y=(-2π, 2π), z=(-4π, 0)),
        )

free_surfaces = (
                 ExplicitFreeSurface(gravitational_acceleration=1),
                )

function run_hydrostatic_free_surface_model_regression_test(; architecture = CPU(),
                                                              grid_index = 1,
                                                              closure_index = 1,
                                                              coriolis_index = 1,
                                                              free_surface_index = 1)

    grid = grids[grid_index]
    coriolis = coriolises[coriolis_index]
    closure = closures[closure_index]
    free_surface = free_surfaces[free_surface_index]

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        architecture = architecture,
                                        closure = closure,
                                        coriolis = coriolis,
                                        buoyancy = SeawaterBuoyancy(gravitational_acceleration = 1,
                                                                    equation_of_state = LinearEquationOfState(α=0.7, β=0.3)))

    set!(model, u=initial_u)
    set!(model, v=initial_v)
    set!(model, η=initial_η)
    set!(model, T=initial_T)
    set!(model, S=initial_S)

    regression_test_name = @sprintf("hydrostatic_free_surface_regression_%s_%s_%s_%s.nc",
                                    typeof(model.grid).name.wrapper,
                                    typeof(model.coriolis).name.wrapper,
                                    typeof(model.closure).name.wrapper,
                                    typeof(model.free_surface).name.wrapper)

    regression_data_filepath = joinpath(dirname(@__FILE__), "data", regression_test_name)

    simulation = Simulation(model, Δt=1/256, stop_iteration=10)

    ####
    #### Uncomment the block below to generate regression data.
    ####

    @warn ("You are generating new data for the hydrostatic free surface model regression test.")

    outputs = Dict("v" => model.velocities.v,
                   "u" => model.velocities.u,
                   "w" => model.velocities.w,
                   "T" => model.tracers.T,
                   "S" => model.tracers.S)

    nc_writer = NetCDFOutputWriter(model, outputs, filepath=regression_data_filepath, schedule=IterationInterval(10))
    push!(simulation.output_writers, nc_writer)

    ####
    #### Regression test
    ####

    run!(simulation)

    ds = Dataset(regression_data_filepath, "r")

    test_fields = (u = Array(interior(model.velocities.u)),
                   v = Array(interior(model.velocities.v)),
                   w = Array(interior(model.velocities.w)),
                   T = Array(interior(model.tracers.T)),
                   S = Array(interior(model.tracers.S)))

    correct_fields = (u = ds["u"][:, :, :, end],
                      v = ds["v"][:, :, :, end],
                      w = ds["w"][:, :, :, end],
                      T = ds["T"][:, :, :, end],
                      S = ds["S"][:, :, :, end])

    summarize_regression_test(test_fields, correct_fields)

    @test all(test_fields.u .≈ correct_fields.u)
    @test all(test_fields.v .≈ correct_fields.v)
    @test all(test_fields.w .≈ correct_fields.w)
    @test all(test_fields.T .≈ correct_fields.T)
    @test all(test_fields.S .≈ correct_fields.S)

    return nothing
end
