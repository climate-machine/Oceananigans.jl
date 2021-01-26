using Oceananigans.TimeSteppers: AbstractTimeStepper

struct WickerSkamarockTimeStepper{S, F, I} <: AbstractTimeStepper
      slow_source_terms :: S
      fast_source_terms :: F
    intermediate_fields :: I

    function WickerSkamarockTimeStepper(arch, grid, tracers;
        slow_source_terms = SlowSourceTermFields(arch, grid, tracers),
        fast_source_terms = FastSourceTermFields(arch, grid, tracers),
      intermediate_fields = FastSourceTermFields(arch, grid, tracers))
  
      return new{typeof(slow_source_terms), typeof(fast_source_terms),
        typeof(intermediate_fields)}(slow_source_terms, fast_source_terms, intermediate_fields)
  end
end

function SlowSourceTermFields(arch, grid, tracernames)
    ρu = XFaceField(arch, grid)
    ρv = YFaceField(arch, grid)
    ρw = ZFaceField(arch, grid)
    tracers = TracerFields(tracernames, arch, grid, ())
    return (ρu = ρu, ρv = ρv, ρw = ρw, tracers = tracers)
end

function FastSourceTermFields(arch, grid, tracernames)
    ρu = XFaceField(arch, grid)
    ρv = YFaceField(arch, grid)
    ρw = ZFaceField(arch, grid)
    tracers = TracerFields(tracernames, arch, grid, ())
    return (ρu = ρu, ρv = ρv, ρw = ρw, tracers = tracers)
end
