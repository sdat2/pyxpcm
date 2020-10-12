"""
sithom_plotting_style.py
========================
import sithom_plotting_style as sps
usage

ds = xr.open_dataset('example.nc')

sps.ds_for_grahing(ds).plot()
"""

import numpy as np
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

def rerturn_list_of_colormaps(number, fade_to_white=True):
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
