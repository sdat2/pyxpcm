# ! /usr/local/bin/python3
# %load_ext autoreload
# %autoreload 2
import os
import numpy as np
import xarray as xr
xr.set_options(keep_attrs=True)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.path as mpath
import time
from functools import wraps
import pyxpcm
from pyxpcm.models import pcm


def timeit(method):
    """
    timeit is a wrapper for performance analysis which should
    return the time taken for a function to run,
    :param method: the function that it takes as an input
    :return: timed
    example usage:
    tmp_log_data={}
    part = spin_forward(400, co, particles=copy.deepcopy(particles),
                        log_time=tmp_log_d)
    # chuck it into part to stop interference.
    assert part != particles
    spin_round_time[key].append(tmp_log_data['SPIN_FORWARD'])
    TODO make this function user friendly for getting the data from.
    """
    @wraps(method)
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = (te - ts)
        else:
            print('%r  %2.5f s\n' % (method.__name__, (te - ts)))
        return result
    return timed


@timeit
def _return_name(K, pca):

    return 'nc/i-metric-joint-k-' + str(K) + '-d-' + str(pca)


@timeit
def _return_plot_folder(K, pca):
    folder =  ('../FBSO-Report/images/i-metric-joint-k-'
               + str(K) + '-d-' + str(pca) + '/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


@timeit
def _return_folder(K, pca):

    folder = _return_name(K, pca) + '/'

    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def _return_pair_name(K, pca):
    return 'nc/pair-i-metric-k-' + str(K) + '-d-' + str(pca)

def _return_pair_folder(K, pca):
    folder = 'nc/pair-i-metric-k-' + str(K) + '-d-'+ str(pca) + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


@timeit
def train_on_interpolated_year(time_i=42, K=5, maxvar=3, min_depth=300,
                               max_depth=2000, separate_pca=True,
                               remove_s_and_t=True):

    main_dir = '/Users/simon/bsose_monthly/'
    salt = main_dir + 'bsose_i106_2008to2012_monthly_Salt.nc'
    theta = main_dir + 'bsose_i106_2008to2012_monthly_Theta.nc'

    z = np.arange(-min_depth, -max_depth, -10.)
    features_pcm = {'THETA': z, 'SALT': z}
    features = {'THETA': 'THETA', 'SALT': 'SALT'}
    fname = 'interp.nc'
    if not os.path.isfile(fname):
        salt_nc = xr.open_dataset(salt).isel(time=slice(time_i, time_i+12))
        theta_nc = xr.open_dataset(theta).isel(time=slice(time_i, time_i+12))
        big_nc = xr.merge([salt_nc, theta_nc])
        both_nc = big_nc.where(big_nc.coords['Depth'] >
                           max_depth).drop(['iter', 'Depth',
                                            'rA', 'drF', 'hFacC'])

        lons_new = np.linspace(both_nc.XC.min(), both_nc.XC.max(), 60*4)
        lats_new = np.linspace(both_nc.YC.min(), both_nc.YC.max(), 60)


        ds = both_nc.interp(coords={'YC': lats_new, 'XC': lons_new}) #, method='cubic')
        ds.to_netcdf(fname)
    else:
        ds = xr.open_dataset(fname)
    m = pcm(K=K, features=features_pcm,
            separate_pca=separate_pca,
            maxvar=maxvar,
            timeit=True, timeit_verb=1)

    m.fit(ds, features=features, dim='Z')

    m.add_pca_to_xarray(ds, features=features,
                        dim='Z', inplace=True)

    m.find_i_metric(ds, inplace=True)
    m.predict(ds, features=features, dim='Z',inplace=True)

    del ds.PCA_VALUES.attrs['_pyXpcm_cleanable']
    del ds.IMETRIC.attrs['_pyXpcm_cleanable']
    del ds.A_B.attrs['_pyXpcm_cleanable']
    del ds.PCM_LABELS.attrs['_pyXpcm_cleanable']

    if remove_s_and_t:
        ds = ds.drop(['THETA', 'SALT'])

    return m, ds


@timeit
def pca_from_interpolated_year(m, pca=2, K=5, time_i=42, max_depth=2000):

    main_dir = '/Users/simon/bsose_monthly/'
    salt = main_dir + 'bsose_i106_2008to2012_monthly_Salt.nc'
    theta = main_dir + 'bsose_i106_2008to2012_monthly_Theta.nc'
    features = {'THETA': 'THETA', 'SALT': 'SALT'}

    salt_nc = xr.open_dataset(salt).isel(time=time_i)
    theta_nc = xr.open_dataset(theta).isel(time=time_i)
    big_nc = xr.merge([salt_nc, theta_nc])
    both_nc = big_nc.where(big_nc.coords['Depth'] >
                           max_depth).drop(['iter', 'Depth',
                                            'rA', 'drF', 'hFacC'])

    attr_d = {}

    for coord in both_nc.coords:
        attr_d[coord] = both_nc.coords[coord].attrs

    ds = both_nc

    ds = m.find_i_metric(ds, inplace=True)

    ds = m.add_pca_to_xarray(ds, features=features,
                            dim='Z', inplace=True)


    def sanitize():
        del ds.IMETRIC.attrs['_pyXpcm_cleanable']
        del ds.A_B.attrs['_pyXpcm_cleanable']
        del ds.PCA_VALUES.attrs['_pyXpcm_cleanable']

    for coord in attr_d:
        ds.coords[coord].attrs = attr_d[coord]

    sanitize()

    ds = ds.drop(['SALT', 'THETA'])

    ds = ds.expand_dims(dim='time', axis=None)

    ds = ds.assign_coords({"time":
                           ("time",
                            [salt_nc.coords['time'].values])})

    ds.coords['time'].attrs = salt_nc.coords['time'].attrs

    ds.to_netcdf(_return_folder(K, pca) + str(time_i)
                 + '.nc', format='NETCDF4')

    #return ds


@timeit
def run_through_joint_two(K=5, pca=3):
    m, ds = train_on_interpolated_year(time_i=42, K=K,
                                       maxvar=pca, min_depth=300,
                                       max_depth=2000, separate_pca=False)

    # m.to_netcdf('nc/pc-joint-m.nc')

    for time_i in range(60):
        pca_from_interpolated_year(m, K=K, pca=pca, time_i=time_i)


# run_through_joint_two()


@timeit
def merge_and_save_joint(K=5, pca=3):

    pca_ds = xr.open_mfdataset(_return_folder(K, pca) + '*.nc',
                               concat_dim="time",
                               combine='by_coords',
                               chunks={'time': 1},
                               data_vars='minimal',
                               # parallel=True,
                               coords='minimal',
                               compat='override')   # this is too intense for memory

    xr.save_mfdataset([pca_ds], [_return_name(K, pca) + '.nc'], format='NETCDF4')


@timeit
def merge_pair(K=5, pca=3):
    ds = xr.open_mfdataset(_return_pair_folder(K, pca) + '*.nc',
                           concat_dim='time',
                           combine='by_coords',
                           chunks={'time': 1},
                           data_vars='minimal',
                           coords='minimal',
                           compat='override')
    xr.save_mfdataset([ds], [_return_pair_name(K, pca) + '..nc' ] , format='NETCDF4')


#for K in [20]:
#    merge_pair(K)


@timeit
def run_through():
    K_list =  [4, 2, 10]
    for K in K_list:
        run_through_joint_two(K=K)
    for K in K_list:
        merge_and_save_joint(K=K)


# run_through()


def one_fit(ds, K, features, features_pcm, separate_pca, maxvar):

    print('K =', K)

    ds2 = ds.copy()

    m = pcm(K=K, features=features_pcm,
            separate_pca=separate_pca,
            maxvar=maxvar,
            # timeit=True,
            # timeit_verb=1
            )

    m.fit(ds2, features=features, dim='Z')

    bic = m.bic(ds2, features=features, dim='Z')

    print('bic ', bic)

    # del m

    # del ds2

    return bic


def make_interp(time_i=42, min_depth=300, max_depth=2000):

    main_dir = '/Users/simon/bsose_monthly/'
    salt = main_dir + 'bsose_i106_2008to2012_monthly_Salt.nc'
    theta = main_dir + 'bsose_i106_2008to2012_monthly_Theta.nc'
    salt_nc = xr.open_dataset(salt).isel(time=slice(time_i, time_i + 12))
    theta_nc = xr.open_dataset(theta).isel(time=slice(time_i, time_i + 12))
    big_nc = xr.merge([salt_nc, theta_nc])

    both_nc = big_nc.where(big_nc.coords['Depth'] >
                           max_depth).drop(['iter', 'Depth',
                                            'rA', 'drF', 'hFacC'])

    lons_new = np.linspace(both_nc.XC.min(), both_nc.XC.max(), 60 * 4)
    lats_new = np.linspace(both_nc.YC.min(), both_nc.YC.max(), 60)
    ds = both_nc.interp(coords={'YC': lats_new, 'XC': lons_new})

    ds.to_netcdf('interp.nc')

    return ds


def run_k_on_interpolated_year(time_i=42, min_depth=300,
                               max_depth=2000,  maxvar=3,
                               separate_pca=True):

    # ds = make_interp(time_i=time_i, min_depth=min_depth, max_depth=max_depth)
    ds = xr.open_dataset('interp.nc')

    z = np.arange(-min_depth, -max_depth, -10.)
    features_pcm = {'THETA': z, 'SALT': z}
    features = {'THETA': 'THETA', 'SALT': 'SALT'}

    bic_list = []

    for K in range(2, 20):

        bic_list.append(one_fit(ds, K,
                                features, features_pcm,
                                separate_pca, maxvar))
    # for K in range(3, 10):
    print(bic_list)


# run_k_on_interpolated_year(separate_pca=False)

if __name__ == "__main__":
    merge_and_save_joint(K=5)


    # run_through()
