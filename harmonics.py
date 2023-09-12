# Edited version of Keaton Bell's script
# https://github.com/keatonb/sphericalharmonics/blob/master/shanimate.py
import sys
import argparse
import subprocess
import numpy as np
import scipy.special as sp
import cartopy.crs as ccrs

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation


def main(args):
    assert np.abs(args.m) <= args.ell
    assert np.abs(args.inc) <= 180
    outfile = args.outfile
    if outfile is None:
        outfile = f"l{args.ell}m{args.m}.gif"

    plotcrs = ccrs.Orthographic(0, 90 - args.inc)

    lon = np.linspace(0, 2 * np.pi, args.nlon) - np.pi
    lat = np.linspace(-np.pi / 2, np.pi / 2, args.nlat)
    colat = lat + np.pi / 2
    d = np.zeros((len(lon), len(colat)), dtype=np.complex64)
    for j, yy in enumerate(colat):
        for i, xx in enumerate(lon):
            d[i, j] = sp.sph_harm(args.m, args.ell, xx, yy)

    fig = plt.figure(
            figsize=(args.size, args.size),
            )

    ax = plt.subplot(projection=plotcrs)
    fig.tight_layout(pad=0.1)

    drm = np.transpose(np.real(d))
    vlim = np.max(np.abs(drm))
    ax.pcolormesh(
        lon * 180 / np.pi,
        lat * 180 / np.pi,
        drm,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
    )
    ax.relim()
    ax.autoscale_view()

    def init():
        return

    def animate(i):
        drm = np.transpose(
            np.real(
                d * np.exp(-1.0j * (2.0 * np.pi * float(i) / np.float64(args.nframes)))
            )
        )
        sys.stdout.write("\rFrame {0} of {1}".format(i + 1, args.nframes))
        sys.stdout.flush()
        ax.clear()
        ax.pcolormesh(
            lon * 180 / np.pi,
            lat * 180 / np.pi,
            drm,
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=-vlim,
            vmax=vlim,
        )
        ax.relim()
        ax.autoscale_view()
        fig.patch.set_visible(False)
        ax.patch.set_visible(False)
        return

    interval = args.duration / np.float64(args.nframes)

    """
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=args.nframes, interval=interval, blit=False
    )
    anim.save(
        outfile,
        dpi=args.dpi,
        fps=1.0 / interval,
        writer="imagemagick",
        savefig_kwargs={"transparent": True},
    )
    """
    for i in range(args.nframes):
        animate(i)
        plt.savefig(f"./tmp/harm_{i:02d}.png", transparent=True)
    subprocess.call(f"convert -delay 5 -loop 0 -dispose Background ./tmp/*.png {outfile}", shell=True)

    print('\nWrote '+outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate animated gif of spherical harmonic."
    )
    parser.add_argument("ell", type=int, help="spherical degree")
    parser.add_argument("m", type=int, help="azimuthal order")
    parser.add_argument("-o", "--outfile", type=str, help="output gif filename")
    parser.add_argument(
        "-i", "--inc", type=float, default=90, help="inclination (degrees from pole)"
    )
    parser.add_argument(
        "-s", "--size", type=float, default=1, help="image size (inches)"
    )
    parser.add_argument(
        "-n", "--nframes", type=int, default=64, help="number of frames in animation"
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=2, help="animation duration (seconds)"
    )
    parser.add_argument(
        "--nlon", type=int, default=200, help="number of longitude samples"
    )
    parser.add_argument(
        "--nlat", type=int, default=500, help="number of latitude samples"
    )
    parser.add_argument("--dpi", type=float, default=300, help="dots per inch")
    args = parser.parse_args()

    main(args)
