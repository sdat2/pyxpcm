import numpy as np
import xarray as xr
xr.set_options(keep_attrs=True)
import translate_nc as tnc
import plot_i_metric as pim
import run_through_gmm as rtg


@rtg.timeit
def run_through_plot():
    link_to_netcdf = 'nc/i-metric-joint-k-5.nc'
    ds = xr.open_dataset(link_to_netcdf)
    print(ds.__str__())
    ### start loop
    batch_size = 2
    
    for i in range(40, 60, batch_size):
        print('running', i)
        da = tnc.pair_i_metric(
           ds.isel(time=slice(i, i + batch_size)))
        for j in range(batch_size):
            pim.plot_da(da, j)       


run_through_plot()

