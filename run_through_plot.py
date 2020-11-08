import xarray as xr
import numpy as np


def run_through_plot():
    link_to_netcdf = 'nc/i-metric-joint-k-5.nc'
    ds = xr.open_dataset(link_to_netcdf)
    print(ds.__str__())
    ### start loop
    for i in range(0, 60, 10):
        print('running', i)
        print(ds(slice(0, 10))


run_through_plot()

