using Printf
using Oceananigans
using Oceananigans: Utils, Units
using Oceananigans.OutputWriters
using Oceanostics: SingleLineProgressMessenger

grid = RegularRectilinearGrid(size=(4, 4, 4), extent=(1,1,1))
model = IncompressibleModel(architecture = CPU(), grid = grid)

start_time = 1e-9*time_ns()
simulation = Simulation(model, Δt=1, stop_time=50, iteration_interval=5,
                        progress=SingleLineProgressMessenger(LES=false, initial_wall_time_seconds=start_time),
                        )
println("\n", simulation,"\n",)

@info "Setting up chk writer"
simulation.output_writers[:chk_writer] = Checkpointer(model; dir=".",
                                         prefix = "chk.test",
                                         schedule = TimeInterval(5),
                                         force = true, cleanup = true,
                                         )
                                         
println("\n", simulation,"\n",)          

@printf("---> Starting run!\n")
run!(simulation, pickup=true)
