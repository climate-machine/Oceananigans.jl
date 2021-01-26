using NCDatasets
using Interpolations
using Makie

#####
##### Read data
#####

ds = NCDataset("dry_convection.nc")

nx, ny, nz, nt = size(ds["ρe"])

ρe₀ = Array(ds["ρe₀"]);
ρe  = Array(ds["ρe"]);
ρe′ = similar(ρe);

for n in 1:length(ds["time"])
    @. ρe′[:, :, :, n] = ρe[:, :, :, n] - ρe₀;
end

#####
##### Interpolate to a finer grid with S::Int > 1
#####

function get_data(n, S=1)
    if S == 1
        return ρe′[:, :, :, n]
    else
        itp = interpolate(ρe′[:, :, :, n], BSpline(Cubic(Periodic(OnCell()))));
        n = size(ρe′[:, :, :, n])
        N = S .* n
        ℑρe′ = zeros(N)

        i_inds = range(1, n[1], length=N[1])
        j_inds = range(1, n[2], length=N[2])
        k_inds = range(1, n[3], length=N[3])

        i_inds = reshape(i_inds, (N[1], 1, 1))
        j_inds = reshape(j_inds, (1, N[2], 1))
        k_inds = reshape(k_inds, (1, 1, N[3]))

        @. ℑρe′ = itp(i_inds, j_inds, k_inds)

        return ℑρe′
    end
end

#####
##### Plot!
#####

α(ξ) = ξ  # Opacity/alpha along the cmap (0 <= ξ <= 1)

cmap_rgb = to_colormap(:ice)
A = α.(range(0, 1, length=length(cmap_rgb)))
cmap_rgba = RGBAf0.(cmap_rgb, A)

time_index = Node(1)
data = @lift get_data($time_index, 3)

function make_movie(scene)
    n_frames = nt
    θ = 2π / 360

    record(scene, "dry_convection_oceananigans_makie.mp4", 1:n_frames, framerate=30) do n
        @info "frame $n/$n_frames"
        time_index[] = n
        n == 1 && zoom!(scene, (0, 0, 0), -1.25, false)
        rotate_cam!(scene, (θ, 0, 0))
    end
end

scene = volume(0..nx, 0..ny, 0..nz, data, colorrange=(0, 400), colormap=cmap_rgba, algorithm=:absorption, absorption=10.0f0,
               backgroundcolor=cmap_rgb[2], show_axis=false, resolution = (1920, 1080))

make_movie(scene)

