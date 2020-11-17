import numpy as np
import xarray as xr
xr.set_options(keep_attrs=True)
import run_through_gmm as rtg


@rtg.timeit
def pair_i_metric(ds, threshold=0.05):
    
    sorted_version = np.sort(ds.A_B.values, axis=0)
    # (2, 12, 60, 240)
    # rank, time, YC, XC

    i_metric = ds.IMETRIC.isel(Imetric=0).values

    list_no = [i for i in range(int(np.nanmax(sorted_version)) + 1)]

    cart_prod = [np.array([a, b]) for a in list_no for b in list_no if a <= b and a != b]
    
    pair_i_metric_list = []

    pair_list = []

    for pair in cart_prod:
        
        shape = np.shape(sorted_version)
        
        pair_i_metric = np.empty([shape[1], shape[2], shape[3]])
        pair_i_metric[:] = np.nan
        
        at_least_one_point = False
         
        for i in range(shape[1]):
            for j in range(shape[2]):
                for k in range(shape[3]):
                    
                    if np.array_equal(pair, sorted_version[:, i, j, k]):
                        if i_metric[i, j, k] >= threshold:
                            
                            pair_i_metric[i, j, k] = i_metric[i, j, k]
                            at_least_one_point = True
                
        if at_least_one_point:
            
            pair_i_metric_list.append(pair_i_metric)
            pair_list.append(pair)
            
    print(pair_list)

    pair_i_metric_array = np.zeros([len(pair_i_metric_list), shape[1], shape[2], shape[3]])

    for i in range(len(pair_i_metric_list)):
            
        pair_i_metric_array[i, :, :, :] = pair_i_metric_list[i][:, :, :]
        
    pair_str_list = []

    for i in range(len(pair_list)):
        pair_str_list.append(str(pair_list[i][0] + 1 ) 
                             + ' to ' + str(pair_list[i][1] + 1))

    da = xr.DataArray(pair_i_metric_array,
                      dims=['pair', 'time', 
                            'YC', 'XC'], ##
                      coords={'XC': ds.coords['XC'].values,
                              'YC': ds.coords['YC'].values,
                              'time': ds.coords['time'].values, ##
                              'pair': pair_str_list})
    
    return da


def get_pair_i_metric_da(ds, threshold=0.05):
    da  = pair_i_metric(ds)
    print(da.__str__())
    return da


