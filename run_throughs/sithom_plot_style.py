"""
sithom_plotting_style.py
========================
import sithom_plotting_style as sps
usage

ds = xr.open_dataset('example.nc')

sps.ds_for_grahing(ds).plot()
"""

import numpy as np
import numpy.linalg as la
import re
# import cartopy.crs as ccrs
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import xarray as xr


def mpl_params(quality='high'):
    """
    Apply my plotting style to produce nice looking figures.
    Call this at the start of a script which uses matplotlib,
    and choose the correct setting.
    :return:
    """
    if quality == 'high':
        matplotlib.style.use('seaborn-colorblind')
        param_set = {"pgf.texsystem": "pdflatex",
                     "text.usetex": True,
                     "font.family": "serif",
                     "font.serif": [],
                     "font.sans-serif": ["DejaVu Sans"],
                     "font.monospace": [],
                     "lines.linewidth": 0.75,
                     "axes.labelsize": 10,  # 10
                     "font.size": 8,
                     "legend.fontsize": 9,
                     "xtick.labelsize": 10,  # 10,
                     "ytick.labelsize": 10,  # 10,
                     "scatter.marker": '+',
                     "image.cmap": 'RdYlBu_r',
                     "pgf.preamble": [r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc}"]
                     }
    else:
        matplotlib.style.use('seaborn-colorblind')
        param_set = {"text.usetex": False,
                     "lines.linewidth": 0.75,
                     "font.family": "sans-serif",
                     "font.serif": [],
                     "font.sans-serif": ["DejaVu Sans"],
                     "axes.labelsize": 10,
                     "font.size": 6,
                     "legend.fontsize": 8,
                     "xtick.labelsize": 10,
                     "ytick.labelsize": 10,
                     "image.cmap": 'RdYlBu_r',
                     }

    matplotlib.rcParams.update(param_set)


def tex_escape(text):
    """
    It is better to plot in TeX, but this involves escaping strings.
    from:
        https://stackoverflow.com/questions/16259923/
        how-can-i-escape-latex-special-characters-inside-django-templates
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    # removed unicode(key) from re.escape because this seemed an unnecessary,
      and was throwing an error.
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(key) for key in sorted(conv.keys(), key=lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


def proper_units(text):
    conv = {
        'degK': r'K',
        'degC': r'$^{\circ}C$',
        'degrees\_celsius': r'$^{\circ}$C',
        'degrees\_north': r'$^{\circ}$N',
        'degrees\_east': r'$^{\circ}$E',
        'degrees\_west': r'$^{\circ}$W',
        'I metric': '$\mathcal{I}$--metric'
    }
    regex = re.compile('|'.join(re.escape(key) for key in sorted(conv.keys(), key=lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


def ds_for_graphing(dsA):
    ds = dsA.copy()

    for varname, da in ds.data_vars.items():
        for attr in da.attrs:
            if attr in ['units', 'long_name']:
                da.attrs[attr] = proper_units(tex_escape(da.attrs[attr]))

    for coord in ds.coords:
        if coord not in ['Z']:
            for attr in ds.coords[coord].attrs:
                if attr in ['units', 'long_name']:
                    da.coords[coord].attrs[attr] = proper_units(tex_escape(da.coords[coord].attrs[attr]))

    return ds


mpl_params(quality='high')



def _fading_colormap_name(from_name, fade_to_white=True):
    """
    Takes a python color name and returns a fading color map.
    :param from_name:
    :return:
    """
    red, green, blue, alpha = colors.to_rgba(from_name)

    return _fading_colormap_rgb((red, green, blue), fade_to_white=fade_to_white)


def _fading_colormap_hex(from_hex, fade_to_white=True):
    """
    Takes a hex string as input and returns a fading color map as output.
    :param from_hex:
    :return:
    """
    hex_number = from_hex.lstrip('#')
    return _fading_colormap_rgb(tuple(int(hex_number[i:i + 2], 16) for i in (0, 2, 4)),
                                fade_to_white=fade_to_white)


def _fading_colormap_rgb(from_rgb, fade_to_white=True):
    """
    Takes an r g b tuple and returns a fading color map.
    :param from_rgb: an r g b tuple
    :return:
    """

    # from color r,g,b
    red1, green1, blue1 = from_rgb

    # to color r,g,b
    red2, green2, blue2 = 1, 1, 1

    if fade_to_white:
        cdict = {'red': ((0, red1, red1),
                         (1, red2, red2)),
                 'green': ((0, green1, green1),
                           (1, green2, green2)),
                 'blue': ((0, blue1, blue1),
                          (1, blue2, blue2))}
    else:
        cdict = {'red': ((0, red2, red2),
                         (1, red1, red1)),
                 'green': ((0, green2, green2),
                           (1, green1, green1)),
                 'blue': ((0, blue2, blue2),
                          (1, blue1, blue1))}

    cmap = colors.LinearSegmentedColormap('custom_cmap', cdict)

    return cmap


def fading_colormap(from_color, fade_to_white=True):
    """
    Takes a hex or color name, and returns a fading color map.
    example usage:

    # cmap_a = fading_colormap('blue')
    # cmap_b = fading_colormap('#96f97b')

    :param from_color: either a hex or a name
    :return: cmap --> a colormap that can be used as a parameter in a plot.
    """
    if from_color.startswith('#'):
        cmap = _fading_colormap_hex(from_color, fade_to_white=fade_to_white)
    else:
        cmap = _fading_colormap_name(from_color, fade_to_white=fade_to_white)
    return cmap


def replacement_color_list(number_of_colors):
    """

    :param number_of_colors:
    :return:
    """
    assert isinstance(number_of_colors, int)

    color_d = {2: ['b', 'r'],
               3: ['b', 'green', 'r'],
               4: ['b', 'green', 'orange', 'r'],
               5: ['navy', 'b', 'green', 'orange', 'r'],
               6: ['navy', 'b', 'green', 'orange', 'r', 'darkred'],
               7: ['navy', 'b', 'green', 'yellow', 'orange', 'r', 'darkred'],
               8: ['navy', 'b', 'green', 'yellow', 'orange', 'r', 'darkred', '#fe019a'],
               9: ['navy', 'b', '#b8ffeb', 'green', 'yellow', 'orange', 'r', 'darkred', '#fe019a'],
               10: ['navy', 'b', '#b8ffeb', 'green', '#ccfd7f', 'yellow', 'orange', 'r', 'darkred', '#fe019a'],
               11: ['navy', 'b', '#b8ffeb', 'green', '#ccfd7f', 'yellow', 'orange', 'r', 'darkred', '#cf0234',
                    '#fe019a'],
               12: ['navy', 'b', '#b8ffeb', 'green', '#ccfd7f', 'yellow', 'orange', 'r', 'darkred', '#cf0234',
                    '#6e1005', '#fe019a'],
               13: ['navy', 'b', '#b8ffeb', 'green', '#ccfd7f', 'yellow', '#fdb915', 'orange', 'r', 'darkred',
                    '#cf0234', '#6e1005', '#fe019a'],
               }
    if number_of_colors in color_d:
        color_list = color_d[number_of_colors]
    else:
        color_list = color_d[13]
    return color_list


def return_list_of_colormaps(number, fade_to_white=True):
    color_list = replacement_color_list(number)
    cmap_list = []
    for i in range(number):
        cmap_list.append(fading_colormap(color_list[i %
                                                         len(color_list)],
                                         fade_to_white=fade_to_white))
    return cmap_list


def key_to_cmap(data_d,
                fade_to_white=True):
    """

    :param data_d:
    :return:
    """

    sorted_key_list = mmf.sort_keys_by_avg_lat(data_d)

    base_color_list = replacement_color_list(len(sorted_key_list))

    cmap_d = {}

    for i in range(len(sorted_key_list)):
        cmap_d[sorted_key_list[i]] = fading_colormap(base_color_list[i % len(base_color_list)],
                                                     fade_to_white=fade_to_white)
    return cmap_d


def plot_ellipsoid_trial():
    """
    https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib

    :return:
    """

    # your ellipsoid's covariance_matrix and mean in matrix form
    covariance_matrix = np.array([[1, 0.5, 0],
                                  [0.2, 2, 0],
                                  [0, 0, 10]])
    covariance_matrix1 = np.array([[1, 0.1, 0],
                                   [0.2, 8, 0],
                                   [0, 0, 1]])

    mean = [-0.2, 0.3, 0.1]
    mean1 = [0, 0, 0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig = plot_ellipsoid(fig, ax, covariance_matrix, mean, 1, 'b')
    plot_ellipsoid(fig, ax, covariance_matrix1, mean1, 1, 'g')
    plt.show()


def plot_ellipsoid(fig, ax, covariance_matrix, mean,
                   weight, color, print_properties=False,
                   additional_rotation=np.identity(3)):
    """
    A function for drawing 3d-multivariate guassians with method initially from:
    https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib

    :param fig: The figure matplotlib.pyplot object
    :param ax: The axis matplotlib.pylot object with Axes3D extension
    :param covariance_matrix: A covariance matrix input from the multivariate guassian to be plotted
    :param mean: ditto
    :param weight: ditto
    :param color: ditto
    :return: fig and ax so that they can be used by further plotting steps.
    """

    # I arbitrarily choose some levels to in the multivariate Gaussian to plot.

    if print_properties == True:
        print('weight', weight)
        print('mean', mean)
        print('covariance matrix', covariance_matrix)

    for frac_sigma, alpha in [[1, 0.2*weight*0.5],
                              [1/np.e, 0.4*weight*0.5],
                              [1/(np.e**2), 0.6*weight*0.5]]:
        # find the rotation matrix and radii of the axes
        U, s, rotation = la.svd(covariance_matrix)
        # Singular Value Decomposition from numpy.linalg finds the variance vector s when the covariance
        # matrix has been rotated so that it is diagonal
        radii = np.sqrt(s/frac_sigma)
        # s is the sigma*2 in each of the principal axes directions

        # now carry on with EOL's answer
        u = np.linspace(0.0, 2.0 * np.pi, 100)  # AZIMUTHAL ANGLE (LONGITUDE)
        v = np.linspace(0.0, np.pi, 100)  # POLAR ANGLE (LATITUDE)
        # COORDINATES OF THE SURFACE PRETENDING THAT THE GAUSSIAN IS AT THE CENTRE & NON ROTATED
        x = radii[0] * np.outer(np.cos(u), np.sin(v))  # MESH FOR X
        y = radii[1] * np.outer(np.sin(u), np.sin(v))  # MESH FOR Y
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))  # MESH FOR Z

        # move so that the gaussian is actually rotated and on the right point.
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + mean
                [x[i, j], y[i, j], z[i, j]] = np.dot(additional_rotation, [x[i, j], y[i, j], z[i, j]])

        # plot the surface in a reasonable partially translucent way
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha=alpha)

    return fig, ax

def plot_ellipse(fig, ax, covariance_matrix, mean,
                 weight, color, print_properties=False,):
    """
    A function for drawing 3d-multivariate guassians with method initially from:
    https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib

    :param fig: The figure matplotlib.pyplot object
    :param ax: The axis matplotlib.pylot object with Axes3D extension
    :param covariance_matrix: A covariance matrix input from the multivariate guassian to be plotted
    :param mean: ditto
    :param weight: ditto
    :param color: ditto
    :return: fig and ax so that they can be used by further plotting steps.
    """

    # I arbitrarily choose some levels to in the multivariate Gaussian to plot.

    if print_properties == True:
        print('weight', weight)
        print('mean', mean)
        print('covariance matrix', covariance_matrix)

    for frac_sigma, alpha in [[1, 0.2*weight*0.5],
                              [1/np.e, 0.4*weight*0.5],
                              [1/(np.e**2), 0.6*weight*0.5]]:

        # find the rotation matrix and radii of the axes
        U, s, rotation = la.svd(covariance_matrix)
        # Singular Value Decomposition from numpy.linalg finds
        # the variance vector s when the covariance
        # matrix has been rotated so that it is diagonal
        radii = np.sqrt(s/frac_sigma)
        # s is the sigma*2 in each of the principal axes directions

        # now carry on with EOL's answer
        u = np.linspace(0.0, 2.0 * np.pi, 100)  # AZIMUTHAL ANGLE (LONGITUDE)
        v = np.linspace(0.0, np.pi, 100)  # POLAR ANGLE (LATITUDE)

        # COORDINATES OF THE SURFACE PRETENDING THAT THE GAUSSIAN IS AT THE CENTRE & NON ROTATED

        x = radii[0] * np.outer(np.cos(u), np.sin(v))  # MESH FOR X
        y = radii[1] * np.outer(np.sin(u), np.sin(v))  # MESH FOR Y
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))  # MESH FOR Z

        # move so that the gaussian is actually rotated and on the right point.
        for i in range(len(x)):
            for j in range(len(x)):

                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + mean
                [x[i, j], y[i, j], z[i, j]] = np.dot(additional_rotation, [x[i, j], y[i, j], z[i, j]])

        # plot the surface in a reasonable partially translucent way
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha=alpha)

    return fig, ax
