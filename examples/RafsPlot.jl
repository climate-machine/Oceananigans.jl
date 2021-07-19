
file = jldopen(simulation.output_writers[:fields].filepath)
iterations = parse.(Int, keys(file["timeseries/t"]))
# "./rh.jld2"

##
using Oceananigans, JLD2
file = jldopen("./rh_nonlinear.jld2")
iterations = parse.(Int, keys(file["timeseries/t"]))

λ, ϕ, r = nodes(model.free_surface.η, reshape=true)
ϕind = 140
nλ , _ = size(file["timeseries/u/" * string(0)])
u_time = zeros(nλ, length(iterations))
for (i,iter) in enumerate(iterations)
    u_time[:,i] = file["timeseries/u/" * string(iter)][:, ϕind, 1]
end
t = [file["timeseries/t/" * string(iter)] for iter in iterations] ./ 86400
iterations
##
file = jldopen("./rh_liner.jld2")
iterations = parse.(Int, keys(file["timeseries/t"]))

λ, ϕ, r = nodes(model.free_surface.η, reshape=true)
ϕind = 140
nλ , _ = size(file["timeseries/u/" * string(0)])
u_l_time = zeros(nλ, length(iterations))
for (i,iter) in enumerate(iterations)
    u_l_time[:,i] = file["timeseries/u/" * string(iter)][:, ϕind, 1]
end
t = [file["timeseries/t/" * string(iter)] for iter in iterations] ./ 86400
iterations
##
using Printf
titlelabel = "Zonal Velocity at ϕ = " * @sprintf("%0.2f", ϕ[ϕind]) * " degrees"
fig = Figure(resolution  = (1466, 918))
ax1 = fig[1, 1] = Axis(fig, title = titlelabel, titlesize = 50)
heatmap!(ax1, λ[:], t, u_time, colormap = :balance)
ax1.xlabel = "Longitude in Degrees"
ax1.ylabel = "Time in Days"
ax1.xlabelsize = 30
ax1.ylabelsize = 30       
display(fig)
##
function findlocalmaxima(signal)
   inds = Int[]
   if length(signal)>1
       if signal[1]>signal[2]
           push!(inds,1)
       end
       for i=2:length(signal)-1
           if signal[i-1]<signal[i]>signal[i+1]
               push!(inds,i)
           end
       end
       if signal[end]>signal[end-1]
           push!(inds,length(signal))
       end
   end
   inds
 end

lmaxt = [findlocalmaxima(u_time[:,i]) for i in 1:length(t)]
troughsinds1 = [lmaxt[i][2] for i in 1:length(t)]
troughsinds2 = [lmaxt[i][3] for i in 1:length(t)]
troughsinds3 = [lmaxt[i][4] for i in 1:length(t)]
troughsinds4 = [lmaxt[i][5] for i in 1:length(t)]

 ax = lines(t, troughsinds1)
 lines!(ax.figure.scene, t, troughsinds2)
 ax.axis.title = "Amplitude"
 ax.axis.xlabel = "time in days"
 ax.axis.ylabel = "longitude"             



[troughsinds1[i+1] - troughsinds1[i] for i in 1:length(t)-1]

(troughsinds1[1] - troughsinds1[end])/(t[1] - t[end])
(troughsinds2[1] - troughsinds2[end])/(t[1] - t[end])
(troughsinds3[1] - troughsinds3[end])/(t[1] - t[end])
(troughsinds4[1] - troughsinds4[end])/(t[1] - t[end])