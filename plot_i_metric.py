import numpy as np
import pandas as pd
import xarray as xr
xr.set_options(keep_attrs=True)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.path as mpath
import os, sys
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as patch


import pyxpcm
from pyxpcm.models import pcm
import run_throughs.sithom_plot_style as sps
import run_through_gmm as rtg


@rtg.timeit
def plot_da(da, time_i, K, pca):

    map_proj = ccrs.SouthPolarStereo()
    carree = ccrs.PlateCarree()

    pairs = da.coords['pair'].values.shape[0]

    gs = GridSpec(nrows=2, ncols=pairs,
                  width_ratios=[1 / pairs
                                    for x in range(pairs)],
                  height_ratios=[1, 0.05],
                  wspace=0.15)

    fig = plt.gcf()

    ax1 = fig.add_subplot(gs[0, :], projection=map_proj)
    cbar_axes = [fig.add_subplot(gs[1, i]) for i in range(pairs)]

    fig.subplots_adjust(bottom=0.05, top=0.95,
                        left=0.04, right=0.95,
                        wspace=0.02)

    ax1.set_extent([-180, 180, -90, -30], carree)

    def plot_boundary():
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.515 # , 0.45
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax1.set_boundary(circle, transform=ax1.transAxes)

    plot_boundary()

    cmap_list = sps.return_list_of_colormaps(pairs, fade_to_white=False)

    for i in range(pairs):
        im = da.isel(pair=i, time=time_i).plot(cmap=cmap_list[i], vmin=0, vmax=1,
                                               ax=ax1,
                                               add_colorbar=False,
                                               transform=carree,
                                               subplot_kws={"projection": map_proj},
                                               )
        cbar = plt.colorbar(im, cax=cbar_axes[i],
                            orientation='horizontal',
                            ticks=[0,  1])
        cbar.set_label( da.coords['pair'].values[i])
    plt.suptitle('')
    plt.title('')
    ax1.set_title('')
    ax1.coastlines()
    plt.tight_layout()
    ts = pd.to_datetime(str(da.coords['time'].values[time_i]))

    plt.savefig(rtg._return_plot_folder(K, pca)
                + ts.strftime('%Y-%m-%d') +'_'
                + '.png', dpi=600, bbox_inches='tight')

    plt.clf()
