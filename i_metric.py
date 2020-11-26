import numpy as np
import xarray as xr
xr.set_options(keep_attrs=True)
import translate_nc as tnc
import plot_i_metric as pim
import run_through_gmm as rtg



@rtg.timeit
def run_through_plot(K=5, pca=3, save_nc=False):
    
    #link_to_netcdf = rtg._return_name(K, pca) + '.nc'
    #ds = xr.open_dataset(link_to_netcdf)
    #print(ds.__str__())
     
    batch_size = 2
    
    for i in range(14, 16, batch_size):
        print('running', i)
        if save_nc:
            da = tnc.pair_i_metric(
            ds.isel(time=slice(i, i + batch_size)),
            threshold=0.00)
        if save_nc:
             da.rename('pair_i_metric').to_dataset().to_netcdf(
             rtg._return_pair_folder(K, pca) + str(i) + '.nc'
             )
        else:
            da = xr.open_dataset(rtg._return_pair_folder(K, pca) + str(i) + '.nc').to_array()
        
        print(da)
        for j in range(batch_size):
            pim.plot_da(da, j, K, pca)       


for K in [#5, 
          #4, 
          #2,
          20
           ]:
    run_through_plot(K=K)


