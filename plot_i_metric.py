import numpy as np
import numpy.linalg as la
import xarray as xr
xr.set_options(keep_attrs=True)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.path as mpath
import os, sys
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as patch

#  sys.path.insert(0, os.path.abspath('../../pyxpcm/'))

import pyxpcm
from pyxpcm.models import pcm
import run_throughs.sithom_plot_style as sps
import run_throughs.run_through_gmm as rtg


def plot_da(da, time_i):
    
    map_proj = ccrs.SouthPolarStereo()
    carree = ccrs.PlateCarree()
    
    pairs = da.coords['pair'].values.shape[0]
    
    gs = GridSpec(nrows=2, ncols=pairs,
                  width_ratios=[1 / pairs
                                    for x in range(pairs)],
                  height_ratios=[1, 0.05])
    
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
                                               # the data's projection
                                               subplot_kws={"projection": map_proj},  
                                               # the plot's projection
                                               )
        cbar = plt.colorbar(im, cax=cbar_axes[i],
                            orientation='horizontal',
                            ticks=[0,  1]
                            )
        cbar.set_label( da.coords['pair'].values[i])
    plt.suptitle('')
    plt.title('')
    ax1.coastlines()
    plt.tight_layout()
    # plt.savefig()
    plt.show()
