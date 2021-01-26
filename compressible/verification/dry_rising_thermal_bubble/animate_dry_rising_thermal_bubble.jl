using Printf
using NCDatasets
using PyCall
using PyPlot

using Oceananigans
using Oceananigans.Utils
using JULES

cmocean = pyimport("cmocean")

const plt = PyPlot
plt.ioff()

km = kilometers
Tvar = Energy

ds = NCDataset("dry_rising_thermal_bubble_$(Tvar).nc")

for n in 1:length(ds["time"])
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=true, sharey=true, figsize=(16, 9), dpi=200)

    xC, xF = ds["xC"] / km, ds["xF"] / km
    zC, zF = ds["zC"] / km, ds["zF"] / km

    ρ₀ = ds["ρ₀"][:, 1, :]'
    ρe₀ = ds["ρe₀"][:, 1, :]'

    ρ  = ds["ρ"][:, 1, :, n]'
    ρw = ds["ρw"][:, 1, 2:end, n]'
    ρu = ds["ρu"][:, 1, :, n]'
    ρe = ds["ρe"][:, 1, :, n]'

    fig.suptitle("Dry rising thermal bubble: time = $(prettytime(ds["time"][n]))")

    ax_u = axes[1, 1]
    im = ax_u.pcolormesh(xF, zC, ρu ./ ρ, cmap=cmocean.cm.balance, vmin=-10, vmax=10)
    ax_u.set_title("u(x, z)")
    ax_u.set_ylabel("z (km)")
    fig.colorbar(im, ax=ax_u, label="m/s")

    ax_w = axes[1, 2]
    im = ax_w.pcolormesh(xC, zC, ρw ./ ρ, cmap=cmocean.cm.balance, vmin=-10, vmax=10)
    ax_w.set_title("w(x, z)")
    fig.colorbar(im, ax=ax_w, label="m/s")

    ax_ρ = axes[2, 1]
    im = ax_ρ.pcolormesh(xC, zC, ρ - ρ₀, cmap=cmocean.cm.dense, vmin=-0.007, vmax=0)
    ax_ρ.set_xlabel("x (km)")
    ax_ρ.set_ylabel("z (km)")
    ax_ρ.set_title("ρ′(x, z)")
    fig.colorbar(im, ax=ax_ρ, label="kg/m³")

    ax_e = axes[2, 2]
    im = ax_e.pcolormesh(xC, zC, (ρe .- ρe₀) ./ ρ, cmap=cmocean.cm.thermal, vmin=0, vmax=600)
    ax_e.set_xlabel("x (km)")
    ax_e.set_title("e′(x, z)")
    fig.colorbar(im, ax=ax_e, extend="max", label="J/kg")

    plt.xlim(-10, 10)
    plt.ylim(0, 10)

    filename = @sprintf("dry_rising_thermal_bubble_%s_%03d.png", Tvar, n)
    @info "Saving $filename..."
    plt.savefig(filename)
    plt.close(fig)
end

close(ds)

run(`ffmpeg -r 15 -f image2 -i dry_rising_thermal_bubble_$(Tvar)_%03d.png -vcodec libx264 -crf 22 -pix_fmt yuv420p dry_rising_thermal_bubble_$Tvar.mp4`)
