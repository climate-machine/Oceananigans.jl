from copy import copy

import joblib
import xarray as xr
import xarray.ufuncs as xu
import matplotlib.pyplot as plt
import cmocean
import ffmpeg


def plot_cylinder_frame(n):
    ds = xr.open_dataset("tmp.nc", chunks={"time": 1})

    fig, axes = plt.subplots(nrows=2, figsize=(16, 9), dpi=200)

    fig.suptitle(f"Flow around a cylinder, t = {ds.time[n].values:.2f}", fontsize=16)

    u = ds.u.isel(time=n).squeeze()
    w = ds.w.isel(time=n).squeeze()

    u = u.where(ds.xF**2 + ds.zC**2 > 1)
    w = w.where(ds.xC**2 + ds.zF**2 > 1)

    palette_u = copy(cmocean.cm.speed)
    palette_w = copy(cmocean.cm.balance)
    palette_u.set_bad("gray", 1.0)
    palette_w.set_bad("gray", 1.0)

    ax_u, ax_w = axes
    img_u = u.plot.pcolormesh(ax=ax_u, vmin=-2.0, vmax=2.0, cmap=palette_w, add_colorbar=False)
    img_w = w.plot.pcolormesh(ax=ax_w, vmin=-2.0, vmax=2.0, cmap=palette_w, add_colorbar=False)

    ax_u.set_title("u-velocity")
    ax_w.set_title("w-velocity")

    fig.colorbar(img_u, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(img_w, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")
        ax.label_outer()

    print(f"Saving frame {n}/{ds.time.size-1}...")
    plt.savefig(f"flow_around_cylinder_{n:05d}.png")
    plt.close("all")

ds = xr.open_dataset("flow_around_cylinder.nc")

joblib.Parallel(n_jobs=-1)(
    joblib.delayed(plot_cylinder_frame)(n)
    for n in range(ds.time.size-1)
)

(
    ffmpeg
    .input("flow_around_cylinder_%05d.png", framerate=30)
    .output("flow_around_cylinder.mp4", crf=15, pix_fmt='yuv420p')
    .overwrite_output()
    .run()
)

