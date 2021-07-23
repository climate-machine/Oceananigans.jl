struct Callback{F, S}
    func :: F
    schedule :: S
end

(callback::Callback)(sim) = callback.func(sim)

"""
    Callback(func; schedule=IterationInterval(1))

Return `Callback` that executes `func(sim::Simulation)` on `schedule`.
"""
Callback(func; schedule=IterationInterval(1)) = Callback(func, schedule)

#####
##### Utilities for run!
#####

default_callback_name(n) = Symbol(:callback, n)

function add_callbacks!(sim, callbacks)
    n_existing_callbacks = length(sim.callbacks)

    for (i, callback) in enumerate(callbacks)
        name = default_callback_name(n_existing_callbacks + i)
        sim.callbacks[name] = callback
    end

    return nothing
end

