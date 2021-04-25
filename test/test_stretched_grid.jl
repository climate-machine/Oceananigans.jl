using Revise

using Oceananigans

@inline    linear(x, L, N) = x .+ L*collect(range(0, 1, length=N+1))
@inline quadratic(x, L, N) = x .+ L*collect(range(0, 1, length=N+1)).^2
@inline     cubic(x, L, N) = x .+ L*collect(range(0, 1, length=N+1)).^3

stretch = Dict([("x", linear), ("y", quadratic), ("z", cubic)])

topologys = ( (Bounded, Bounded, Bounded),(Bounded, Bounded, Flat),
              (Bounded, Flat,    Flat),   (Flat,    Flat,    Flat) )

label1 = ( x=(0, 8), y=(0, 8), z=(0, 8) )
label2 = ( x=(0, 8), y=(0, 8) )
label3 = ( x=(0, 8) )
label4 = ()

#for topo in topologys
#       print("Building model with topology = ", topo, "...\n")#
#end

grid1 = StretchedRectilinearGrid(size=(8, 8, 8), 
                                    x=(0, 8), y=(0, 8), z=(0, 8), 
                                    halo = (3, 3, 3),
                             topology=(Bounded, Bounded, Bounded),
                              stretch=stretch)

grid2 = StretchedRectilinearGrid(size=(8, 8), 
                                    x=(0, 8), y=(0, 8),
                                halo = (3, 3),
                             topology=(Bounded, Bounded, Flat),
                              stretch=stretch)

grid3 = StretchedRectilinearGrid(size=(8), 
                                    x=(0, 8),
                                halo = (3),
                             topology=(Bounded, Flat, Flat),
                              stretch=stretch)

grid4 = StretchedRectilinearGrid(size=(),  
                             topology=(Flat, Flat, Flat),
                                halo = (),
                              stretch=stretch)
                            
show(grid1)
show(grid2)
show(grid3)
show(grid4)

using Oceananigans.Models

print("\n")

#model1 = IncompressibleModel(grid = grid1) 
#set!(model1, u=0)
#print("Build model with topology = ", topologys[1], "\n")

model2 = ShallowWaterModel(grid = grid2, gravitational_acceleration=1) 
set!(model2, h=1)
print("Build model with topology = ", topologys[2], "\n")

#simulation = Simulation(model2, Δt =1, stop_iteration=1)
#run!(simulation)

model3 = ShallowWaterModel(grid = grid3, gravitational_acceleration=1) 
set!(model3, h=1)
print("Build model with topology = ", topologys[3], "\n")

model4 = ShallowWaterModel(grid = grid4, gravitational_acceleration=1) 
set!(model4, h=1)
print("Build model with topology = ", topologys[4], "\n")
