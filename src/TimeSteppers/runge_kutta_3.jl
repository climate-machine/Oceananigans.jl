using Oceananigans: fields

"""
    RungeKutta3TimeStepper{FT, TG} <: AbstractTimeStepper

Holds parameters and tendency fields for a low storage, third-order Runge-Kutta-Wray
time-stepping scheme described by Le and Moin (1991).
"""
struct RungeKutta3TimeStepper{FT, TG, TI} <: AbstractTimeStepper
                 γ¹ :: FT
                 γ² :: FT
                 γ³ :: FT
                 ζ² :: FT
                 ζ³ :: FT
                 Gⁿ :: TG
                 G⁻ :: TG
    implicit_solver :: TI
end

"""
    RungeKutta3TimeStepper(arch, grid, tracers,
                           Gⁿ = TendencyFields(arch, grid, tracers),
                           G⁻ = TendencyFields(arch, grid, tracers))

Return an `RungeKutta3TimeStepper` object with tendency fields on `arch` and
`grid`. The tendency fields can be specified via optional kwargs.
"""
function RungeKutta3TimeStepper(arch, grid, tracers;
                                implicit_solver::TI = nothing,
                                Gⁿ::TG = TendencyFields(arch, grid, tracers),
                                G⁻ = TendencyFields(arch, grid, tracers)) where {TI, TG}

    !isnothing(implicit_solver) &&
        @warn("Implicit-explicit time-stepping with RungeKutta3TimeStepper is not tested. " * 
              "\n implicit_solver: $(typeof(implicit_solver))")

    γ¹ = 8 // 15
    γ² = 5 // 12
    γ³ = 3 // 4

    ζ² = -17 // 60
    ζ³ = -5 // 12

    FT = eltype(grid)

    return RungeKutta3TimeStepper{FT, TG, TI}(γ¹, γ², γ³, ζ², ζ³, Gⁿ, G⁻, implicit_solver)
end

#####
##### Time steppping
#####

"""
    time_step!(model::AbstractModel{<:RungeKutta3TimeStepper}, Δt; euler=false)

Step forward `model` one time step `Δt` with a 3rd-order Runge-Kutta method.
The 3rd-order Runge-Kutta method takes three intermediate substep stages to
achieve a single timestep. A pressure correction step is applied at each intermediate
stage.
"""
function time_step!(model::AbstractModel{<:RungeKutta3TimeStepper}, Δt)
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model)

    γ¹ = model.timestepper.γ¹
    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    first_stage_Δt  = γ¹ * Δt
    second_stage_Δt = (γ² + ζ²) * Δt
    third_stage_Δt  = (γ³ + ζ³) * Δt

    #
    # First stage
    #

    calculate_tendencies!(model)

    rk3_substep!(model, Δt, γ¹, nothing)

    calculate_pressure_correction!(model, first_stage_Δt)
    pressure_correct_velocities!(model, first_stage_Δt)

    tick!(model.clock, first_stage_Δt; stage=true)
    store_tendencies!(model)
    update_state!(model)
    update_particle_properties!(model, first_stage_Δt)

    #
    # Second stage
    #

    calculate_tendencies!(model)

    rk3_substep!(model, Δt, γ², ζ²)

    calculate_pressure_correction!(model, second_stage_Δt)
    pressure_correct_velocities!(model, second_stage_Δt)

    tick!(model.clock, second_stage_Δt; stage=true)
    store_tendencies!(model)
    update_state!(model)
    update_particle_properties!(model, second_stage_Δt)

    #
    # Third stage
    #

    calculate_tendencies!(model)
    
    rk3_substep!(model, Δt, γ³, ζ³)

    calculate_pressure_correction!(model, third_stage_Δt)
    pressure_correct_velocities!(model, third_stage_Δt)

    tick!(model.clock, third_stage_Δt)
    update_state!(model)
    update_particle_properties!(model, third_stage_Δt)

    return nothing
end

#####
##### Time stepping in each substep
#####

stage_Δt(Δt, γⁿ, ζⁿ) = Δt * (γⁿ + ζⁿ)
stage_Δt(Δt, γⁿ, ::Nothing) = Δt * γⁿ

function rk3_substep!(model, Δt, γⁿ, ζⁿ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    substep_field_kernel! = rk3_substep_field!(device(model.architecture), workgroup, worksize)

    model_fields = prognostic_fields(model)

    events = []

    for (i, field) in enumerate(model_fields)

        field_event = substep_field_kernel!(field, Δt, γⁿ, ζⁿ,
                                            model.timestepper.Gⁿ[i],
                                            model.timestepper.G⁻[i],
                                            dependencies=barrier)

        # TODO: function tracer_index(model, field_index) = field_index - 3, etc...
        tracer_index = i - 3 # assumption

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.clock,
                       stage_Δt(Δt, γⁿ, ζⁿ),
                       model.closure,
                       tracer_index,
                       model.diffusivities,
                       dependencies = field_event)

        push!(events, field_event)
    end

    wait(device(model.architecture), MultiEvent(Tuple(events)))

    return nothing
end

"""
Time step fields via the 3rd-order Runge-Kutta method

    `U^{m+1} = U^m + Δt (γⁿ G^{m} + ζⁿ G^{m-1})`,

where `m` denotes the substage.
"""

"""
Time step velocity fields with a 3rd-order Runge-Kutta method.
"""
@kernel function rk3_substep_field!(U, Δt, γⁿ, ζⁿ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U[i, j, k] += Δt * (γⁿ * Gⁿ[i, j, k] + ζⁿ * G⁻[i, j, k])
    end
end

"""
Time step velocity fields with a 3rd-order Runge-Kutta method.
"""
@kernel function rk3_substep_field!(U, Δt, γ¹, ::Nothing, G¹, G⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U[i, j, k] += Δt * γ¹ * G¹[i, j, k]
    end
end


