#!/usr/bin/env python3

import argparse
import collections
import itertools
import glob
import os

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import xarray as xr
import yaml

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

mpl.rcParams['savefig.dpi'] = 400
mpl.rcParams['figure.dpi'] = 109
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.figsize'] = [3.2, 2.4]
mpl.rcParams['font.size'] = 7.0
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['pdf.fonttype'] = 42

iminterpol = 'nearest'  # 'antialiased'

_lw = None

_ls = collections.namedtuple('LineStyle', ['color', 'style'])
cci_plot_style = {  # Day, Night
    'N2':  (_ls('#EE0600', '-'),  _ls('#82191D', '-')),
    'N3':  (_ls('#84B8FD', ''),   _ls('#000037', '-')),
    'D2':  (_ls('#EE0600', '--'), _ls('#82191D', '--')),
    'D3':  (_ls('#84B8FD', ''),   _ls('#000037', '--')),
    'N2a': (_ls('#35EAB9', '-'),  _ls('#115423', '-')),
    'N2b': (_ls('#35EAB9', '-'),  _ls('#115423', '-')),
    'N1':  (_ls('#EE0600', '-'),  _ls('#82191D', '-')),
    'N1b': (_ls('#35EAB9', '-'),  _ls('#115423', '-')),
    'N4':  (_ls('#84B8FD', ''),   _ls('#000037', '-')),
    'N2*': (_ls('#35EAB9', ''),   _ls('#35EAB9', '-')),
    'D2*': (_ls('#35EAB9', ''),   _ls('#35EAB9', '--')),
    'L4':  (_ls('#EE0600', '-'),  _ls('#000037', '-')),
    None:  (_ls('', ''),          _ls('', '')),
    '2 chan': (_ls('#EE0600', '-'),  _ls('#82191D', '-')),
    '3 chan': (_ls('#84B8FD', ''),   _ls('#000037', '-')),
    'ref':    (_ls('#35EAB9', ''),   _ls('#35EAB9', '-')),
    }


_insitu_names = {
    'argo': 'Argo',
    'argosurf': 'Argo surface',
    'bottle': 'Bottle',
    'ctd': 'CTD',
    'drifter': 'Drifter',
    'drifter_cmems': 'Drifter',
    'gtmba': 'GTMBA',
    'gtmba2': 'GTMBA2',
    'mbt': 'MBT',
    'mooring': 'Mooring',
    'ship': 'Ship',
    'xbt': 'XBT',
    'noship': 'Non-ship',
    'merged': 'Merged',
    'ref': 'Reference',
    }


def set_scale(scale=1.0):
    lines_context = {
        'axes.labelpad': 4.0,
        'axes.linewidth': 0.8,
        'axes.titlepad': 6.0,
        'boxplot.boxprops.linewidth': 1.0,
        'boxplot.capprops.linewidth': 1.0,
        'boxplot.flierprops.linewidth': 1.0,
        'boxplot.flierprops.markeredgewidth': 1.0,
        'boxplot.flierprops.markersize': 6.0,
        'boxplot.meanprops.linewidth': 1.0,
        'boxplot.meanprops.markersize': 6.0,
        'boxplot.medianprops.linewidth': 1.0,
        'boxplot.whiskerprops.linewidth': 1.0,
        'boxplot.whiskers': 1.5,
        'errorbar.capsize': 0.0,
        'grid.linewidth': 0.8,
        'hatch.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'lines.markeredgewidth': 1.0,
        'lines.markersize': 6.0,
        'patch.linewidth': 1.0,
        }
    # axis tick sizes in points
    ticks_context = {
        'xtick.major.pad': 3.5,
        'xtick.major.size': 3.5,
        'xtick.major.width': 0.8,
        'xtick.minor.pad': 3.4,
        'xtick.minor.size': 2.0,
        'xtick.minor.width': 0.6,
        'ytick.major.pad': 3.5,
        'ytick.major.size': 3.5,
        'ytick.major.width': 0.8,
        'ytick.minor.pad': 3.4,
        'ytick.minor.size': 2.0,
        'ytick.minor.width': 0.6,
        }
    params = {k: v*scale for k, v in lines_context.items()}
    plt.rcParams.update(params)
    params = {k: v*scale for k, v in ticks_context.items()}
    plt.rcParams.update(params)


def figsize(size):
    """Return figure sizes relative to current default.

    If size is two element array-like, then scales the default size

    If size is 'double', return a double width figsize

    short: reduce height to give a widescreen 16:9 ratio
    """
    sz = mpl.rcParams['figure.figsize'].copy()
    if np.shape(size) == (2,):
        return [i*j for i, j in zip(size, sz)]
    elif size == 'double':
        sz[0] *= 2
        return sz
    elif size == 'short':
        sz[1] = sz[0] * 9 / 16
    return sz


def tight_layout(fig):
    # Don't use tight_layout if constrained is active
    if not mpl.rcParams['figure.constrained_layout.use']:
        fig.tight_layout()


def set_simple(sat, ver=''):
    global cci_plot_style
    # Hide N2 night-time retrieval for CDR2.0 plots
    if sat != 'atsr-e1' and ver.startswith('cdr2-0'):
        cci_plot_style['N2'] = cci_plot_style['N2'][0], _ls('', '')
    cci_plot_style['D2'] = cci_plot_style['N2']
    cci_plot_style['D3'] = cci_plot_style['N3']


def rsd(x, *args, **kwargs):
    """
    Compute the robust standard deviation of x
    """
    return 1.4826 * (abs(x-x.median(*args, **kwargs)).median(*args, **kwargs))


def _gauss(x, a, mu, sigma):
    """Gaussian function"""
    return a*np.exp(-(x-mu)**2/(2.*sigma**2))


def getlabel(da, default=''):
    """Get axis label from CF metadata if available"""
    label = getattr(da, 'long_name', default)
    if hasattr(da, 'units'):
        label += ' / ' + da.units
    return label


def pvir_plot_hist(data, ret, xaxis, xbins=80, xrange=(-4, 4), day=False,
                   legend=True, histtype=None, showstats=True, title=''):
    diff = data[xaxis]
    label = ret
    ds, ns = cci_plot_style[ret]
    color, style = ds if day else ns
    if not style:
        color = None
        style = None
    hist, edge = np.histogram(diff, xbins, xrange)
    xval = (edge[:-1] + edge[1:])/2
    fig, axs = plt.subplots()
    axs.ticklabel_format(axis='y', scilimits=(-2, 2))
    axs.yaxis.set_major_locator(mpl.ticker.MaxNLocator(
                nbins='auto', steps=[1, 2, 4, 5, 10], integer=True))
    axs.plot(xval, hist, label=label, color=color, linestyle=style, lw=_lw)
    try:
        p0 = [hist.max(), 0.0, 1.0]
        coef, vmat = scipy.optimize.curve_fit(_gauss, xval.astype(float),
                                              hist.astype(float), p0=p0)
        axs.plot(xval, _gauss(xval, *coef), color=color, linestyle='--')
    except RuntimeError:
        coef = None
    stats = {'n': np.isfinite(diff.values).sum(),
             'mean': np.nanmean(diff),
             'sdev': np.nanstd(diff),
             'med': np.nanmedian(diff),
             'rsd': rsd(diff).data[()],
             'g_x': coef[1],
             'g_s': coef[2],
             }
    if showstats and np.isfinite(diff).sum() > 3:
        color = 'none'
        m, s = stats['mean'], stats['sdev']
        axs.axvline(m, color=color, linestyle=':',
                    label=f'Normal:\n{m:+5.2f} ({s:4.2f})')
        m, s = stats['med'], stats['rsd']
        axs.axvline(m, color=color, linestyle='-.',
                    label=f'Robust:\n{m:+5.2f} ({s:4.2f})')
        if coef is not None:
            m, s = coef[1], coef[2]
            axs.axvline(m, color=color, linestyle='--',
                        label=f'Gaussian fit:\n{m:+5.2f} ({s:4.2f})')
    if legend:
        axs.legend(loc='upper right')
    axs.set_ylim(0)
    axs.set_title(title.format(retrieval=''))
    axs.set_xlim(*xrange)
    axs.set_xlabel('Satellite - in situ / K')
    axs.set_ylabel('Number of matches')
    tight_layout(fig)
    print(f"""{title} ({label}) {stats['n']}
  Normal   {stats['mean']:+5.2f} ({stats['sdev']:4.2f})
  Robust   {stats['med']:+5.2f} ({stats['rsd']:4.2f})
  Gaussian {stats['g_x']:+5.2f} ({stats['g_s']:4.2f})""")
    return fig


def calc_bins(bins=10, range=None, step=None, sample=None, adjust_right=True):
    """
    Calculate bin edges for histogram / binning routines.

    Parameters
    ----------
    bins : int or sequence of scalars, optional
        Number of bins, or bin edges
    range : (float, float), optional
        Data range. If not supplied will use bins or sample range.
    step : float, optional
        Bin width for axis. Only used if range is also supplied
    sample : array_like, optional
        Sample data used to guess data range
    adjust_right : bool, optional
        If true (default) then the right most bin will be extended by the
        smallest possible amount (via np.nextafter)

    Returns
    -------
    edges : np.ndarray
        Bin edges suitable for use with np.digitize
    centre : np.ndarray
        Centre of each bin
    range : (lower_edge, upper_edge)
        First and last bin edges

    Note - if adjust_right is True then the last value in edges has been
    extended. However, this does not affect the centre points or the range
    """
    if range and step:
        nbins = int((range[1] - range[0]) / step)
        first, last = range
        edges = np.linspace(first, last, nbins+1)
    elif step and sample is not None:
        first, last = sample.min(), sample.max()
        nbins = int((last-first) / step)
        edges = np.linspace(first, last, nbins+1)
    elif np.ndim(bins) == 0:
        nbins = bins
        if range:
            first, last = range
        elif sample is not None:
            first, last = sample.min(), sample.max()
        else:
            first, last = 0.0, 1.0
        edges = np.linspace(first, last, nbins+1)
    elif np.ndim(bins) == 1:
        nbins = None
        if range:
            first, last = range
        else:
            first, last = bins.min(), bins.max()
        edges = np.asarray(bins)
    centre = 0.5 * (edges[:-1] + edges[1:])
    if adjust_right:
        # Extent rightmost bin by small amount to include rightmost edge
        edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges, centre, [first, last]


def logtick(x, pos):
    """Tick formatter for simple log10 axis"""
    return f'{10**x}'


def logtick2(x, pos):
    """Tick formatter for simple log10 axis"""
    return f'10$^{{{x:.0f}}}$'


def calc_binned_stats(data, axis, bins, dim='matchup', funcs=['mean', 'std']):
    """
    Calculate stats stratified along a binned axis.

    Parameters
    ----------
    data : xarray.DataArray
        Input data.
    axis : string
        Name of axis we want to use for binning the data
    bins : int or array-like
        Number of bins, or bin edges passed to xarray groupby_bins
    dim : string
        Axis along which we want to calculate the statistics
    funcs: sequence
        List of strings indicating which statistics to calculate for each bin

    Returns
    -------
    stats : xarray.Dataset
        Dataset containing the requested stats stratified by named axis.
    """
    group = data.groupby_bins(axis, bins)
    names = {'mean': 'Mean',
             'median': 'Median',
             'std': 'SD',
             'rsd': 'RSD',
             }
    stats = xr.Dataset()
    # dim is required for the CCI case where we have a dimension
    # for the different retreival types
    for f in funcs:
        if f == 'rsd':
            stats[f] = group.map(rsd, dim=dim)
            stats[f].attrs.update(data.attrs)
        else:
            # Rely on built-in xarray functions
            stats[f] = getattr(group, f)(dim)
        if f in names:
            stats[f].attrs['long_name'] = names[f]

    # xarray sets the binned coordinate "xyz_bins" to an array of Interval
    # objects. These can be used by xarray internal plotting code, but are
    # useless for basic matplotlib etc.
    bnd_name = f'{axis}_bins'
    # First rename the bound variable so the dimension does not end with bnd
    stats = stats.rename({bnd_name: axis})
    stats[axis].attrs.update(data[axis].attrs)
    # Copy the bounds coordinate back to the original name
    stats = stats.assign_coords({bnd_name: stats[axis]})
    # Now replace the dimension coordinate with numeric data
    stats = stats.assign_coords({axis: [x.mid for x in stats[axis].values]})
    stats[axis].attrs.update(data[axis].attrs)

    return stats


def pvir_plot_dependence(ds, xaxis, yaxis='tdiff', xbins=10, xrange=None,
                         xstep=None, title='', xtitle=None, legend=True,
                         robust=True, yrange=[], erange=[], ret='N',
                         wide=False):
    """
    Generate a PVIR dependence plot

    Parameters
    ----------
    ds : xarray.Dataset
        Input data.
    xaxis : string
        Name of xaxis coordinate variable
    yaxis : string
        Name of data variable to plot (e.g. sst_diff)
    xbins : int or sequence of scalars, optional
        Number of xaxis bins, or bin edges
    xrange : (float, float), optional
        X axis data range. If not supplied will use xbins or data range.
    xstep : float, optional
        Bin width for xaxis. Only used if xrange is also supplied
    xtitle : string, optional
        X axis title. If not supplied we use the dataset metadata
    legend : bool, optional
        If ``True`` then add a matplotlib legend.
    robust : bool, optional
        If ``True`` then use robust (median, RSD) statistics.
    yrange: (float, float), optional
        Y axis data range for main (mean/meadian) plot
    erange: (float, float), optional
        Y axis data range for upper (rsd/std.dev) plot
    """
    funcs = {
        True: ['rsd', 'median'],
        False: ['std', 'mean'],
        }
    data = ds.set_coords(xaxis)[yaxis]
    xedges, xcentre, xrange = calc_bins(xbins, xrange, xstep, data[xaxis].values)
    day = calc_binned_stats(data.where(ds.day), xaxis, xedges, funcs=funcs[robust])
    ngt = calc_binned_stats(data.where(ds.ngt), xaxis, xedges, funcs=funcs[robust])
    sz = figsize('double') if wide else None
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=sz,
                                   gridspec_kw={'height_ratios': (1, 2)})
    ds = cci_plot_style[ret+'2'][0]
    ns = cci_plot_style[ret+'3'][1]
    v1, v2 = funcs[robust]
    ax1.plot(day[xaxis], day[v1], color=ds.color, linestyle=ds.style, lw=_lw)
    ax2.plot(day[xaxis], day[v2], color=ds.color, linestyle=ds.style, lw=_lw,
             label=ret+'2 (day)')
    ax1.plot(day[xaxis], ngt[v1], color=ns.color, linestyle=ns.style, lw=_lw)
    ax2.plot(day[xaxis], ngt[v2], color=ns.color, linestyle=ns.style, lw=_lw,
             label=ret+'3')
    ax1.set_ylabel(getlabel(day[v1]))
    ax2.set_ylabel(getlabel(day[v2]))
    if not xtitle:
        xtitle = getlabel(data[xaxis])
    ax2.set_xlabel(xtitle)
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_xlim(*xrange)
    ax2.set_ylim(*yrange)
    ax1.set_ylim(*erange)
    if legend:
        ax2.legend(loc='best', ncol=2)
    ax1.set_title(title.format(retrieval=''))
    if data[xaxis].attrs.get('xlog'):
        ax2.xaxis.set_major_formatter(logtick)
        ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator())
    else:
        plt.ticklabel_format(useOffset=False)
        if xaxis == 'year':
            ax2.xaxis.set_major_locator(mpl.ticker.MaxNLocator(
                    nbins='auto', steps=[1, 2, 5, 10], integer=True))
    tight_layout(fig)
    if not mpl.rcParams['figure.constrained_layout.use']:
        fig.subplots_adjust(hspace=0.1)
    return fig


def pvir_plot_dependence1(ds, xaxis, yaxis='tdiff', xbins=10, xrange=None,
                          xstep=None, title='', xtitle=None, legend=False,
                          robust=True, yrange=[], erange=[], ret='N',
                          wide=False):
    """
    Generate a PVIR dependence plot

    Parameters
    ----------
    ds : xarray.Dataset
        Input data.
    xaxis : string
        Name of xaxis coordinate variable
    yaxis : string
        Name of data variable to plot (e.g. sst_diff)
    xbins : int or sequence of scalars, optional
        Number of xaxis bins, or bin edges
    xrange : (float, float), optional
        X axis data range. If not supplied will use xbins or data range.
    xstep : float, optional
        Bin width for xaxis. Only used if xrange is also supplied
    xtitle : string, optional
        X axis title. If not supplied we use the dataset metadata
    legend : bool, optional
        If ``True`` then add a matplotlib legend.
    robust : bool, optional
        If ``True`` then use robust (median, RSD) statistics.
    yrange: (float, float), optional
        Y axis data range for main (mean/meadian) plot
    erange: (float, float), optional
        Y axis data range for upper (rsd/std.dev) plot
    """
    funcs = {
        True: ['rsd', 'median'],
        False: ['std', 'mean'],
        }
    data = ds.set_coords(xaxis)[yaxis]
    xedges, xcentre, xrange = calc_bins(xbins, xrange, xstep, data[xaxis].values)
    day = calc_binned_stats(data, xaxis, xedges, funcs=funcs[robust])
    sz = figsize('double') if wide else None
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=sz,
                                   gridspec_kw={'height_ratios': (1, 2)})
    ds = cci_plot_style[ret][1]
    v1, v2 = funcs[robust]
    ax1.plot(day[xaxis], day[v1], color=ds.color, linestyle=ds.style, lw=_lw)
    ax2.plot(day[xaxis], day[v2], color=ds.color, linestyle=ds.style, lw=_lw,
             label=ret)
    ax1.set_ylabel(getlabel(day[v1]))
    ax2.set_ylabel(getlabel(day[v2]))
    if not xtitle:
        xtitle = getlabel(data[xaxis])
    ax2.set_xlabel(xtitle)
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_xlim(*xrange)
    ax2.set_ylim(*yrange)
    ax1.set_ylim(*erange)
    if legend:
        ax2.legend(loc='best', ncol=2)
    ax1.set_title(title.format(retrieval=''))
    if data[xaxis].attrs.get('xlog'):
        ax2.xaxis.set_major_formatter(logtick)
        ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator())
    else:
        plt.ticklabel_format(useOffset=False)
        if xaxis == 'year':
            ax2.xaxis.set_major_locator(mpl.ticker.MaxNLocator(
                    nbins='auto', steps=[1, 2, 5, 10], integer=True))
    tight_layout(fig)
    if not mpl.rcParams['figure.constrained_layout.use']:
        fig.subplots_adjust(hspace=0.1)
    return fig


def groupby_2d(data, varname, xaxis, xrange, xstep, yaxis, yrange, ystep,
               robust=False, count=False, std=False):
    """
    Perform an xarray groupby operation in 2 dimensions. This is done by
    digitizing along the two requested axes, and converting the 2d indices to a
    flattened index passed to the xarray groupby function.
    """
    xedges, xcentre, xrange = calc_bins(range=xrange, step=xstep)
    yedges, ycentre, yrange = calc_bins(range=yrange, step=ystep)
    xbin = len(xcentre)
    ybin = len(ycentre)
    xi = np.digitize(data[xaxis], xedges) - 1
    yi = np.digitize(data[yaxis], yedges) - 1
    ind = np.ravel_multi_index([yi, xi], [ybin, xbin], mode='clip')
    data['index'] = ['matchup'], ind
    data = data.set_coords(['index'])
    vars = ['index', varname]
    temp = np.full(xbin*ybin, np.nan)
    try:
        if robust and std:
            avg = data[vars].groupby('index').apply(rsd, dim='matchup')
        elif robust:
            avg = data[vars].groupby('index').median('matchup')
        elif std:
            avg = data[vars].groupby('index').std('matchup')
        else:
            avg = data[vars].groupby('index').mean('matchup')
        temp[avg.index.data] = avg[varname]
    except (StopIteration, ValueError):
        pass
    out = np.reshape(temp, [ybin, xbin])
    if count:
        num = np.reshape(np.bincount(ind, minlength=xbin*ybin), [ybin, xbin])
        return out, num
    else:
        return out


def pvir_plot_spatial(data, varname, title='', robust=False,
                      wide=False, squeeze=True):
    """
    Parameters
    ----------
    wide : bool, optional
        If true we produce a plot twice the standard width (i.e. will double
        figsize[0])
    squeeze : bool, optional
        Note - ignored if wide is true
        When true (default) we reduce the figure height to produce a 16:9 plot
    """
    day = data.where(data.day, drop=True)
    ngt = data.where(data.ngt, drop=True)
    yrange = int(data.year.min()), int(data.year.max()+1)
    mapn = groupby_2d(ngt, varname, 'sat_lon', [-180, 180], 2.5,
                                    'sat_lat', [-90, 90], 2.5, robust=robust)
    mapd = groupby_2d(day, varname, 'sat_lon', [-180, 180], 2.5,
                                    'sat_lat', [-90, 90], 2.5, robust=robust)
    hovn = groupby_2d(ngt, varname, 'year', yrange, 1./12,
                                    'sat_lat', [-90, 90], 2, robust=robust)
    hovd = groupby_2d(day, varname, 'year', yrange, 1./12,
                                    'sat_lat', [-90, 90], 2, robust=robust)
    proj = ccrs.PlateCarree()
    extent = (-180, 180, -90, 90)
    figs = {}
    sz_map = figsize('short') if squeeze else None
    sz_hov = figsize('double') if wide else sz_map

    def mapplot(d, t=None):
        fig = plt.figure(figsize=sz_map)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_global()
        ax.gridlines()
        ax.imshow(d, vmin=-1, vmax=1, cmap='coolwarm', interpolation=iminterpol,
                  transform=proj, extent=extent, origin='lower')
        ax.set_title(t)
        tight_layout(fig)
        return fig
    figs['spatial-day'] = mapplot(mapd, title.format(retrieval=' (day)'))
    figs['spatial-night'] = mapplot(mapn, title.format(retrieval=' (night)'))
    extent = yrange + (-90, 90)

    def hovplot(d, t=None):
        fig = plt.figure(figsize=sz_hov)
        ax = plt.axes()
        ax.imshow(d, vmin=-1, vmax=1, cmap='coolwarm', interpolation=iminterpol,
                  extent=extent, origin='lower', aspect='auto')
        ax.set_xlabel('Year')
        ax.set_ylabel('Latitude')
        ax.set_xlim(*yrange)
        ax.set_ylim(-90, 90)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(
                    nbins='auto', steps=[1, 2, 5, 10], integer=True))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(30))
        ax.set_title(t)
        ax.grid()
        tight_layout(fig)
        return fig
    figs['hov-day'] = hovplot(hovd, title.format(retrieval=' (day)'))
    figs['hov-night'] = hovplot(hovn, title.format(retrieval=' (night)'))
    return figs


def pvir_plot_spatial_l4(data, varname, title='', robust=False,
                         wide=False, squeeze=True):
    """
    Parameters
    ----------
    wide : bool, optional
        If true we produce a plot twice the standard width (i.e. will double
        figsize[0])
    squeeze : bool, optional
        Note - ignored if wide is true
        When true (default) we reduce the figure height to produce a 16:9 plot
    """
    yrange = int(data.year.min()), int(data.year.max()+1)
    mapn = groupby_2d(data, varname, 'sat_lon', [-180, 180], 2.5,
                                     'sat_lat', [-90, 90], 2.5, robust=robust)
    hovn = groupby_2d(data, varname, 'year', yrange, 1./12,
                                     'sat_lat', [-90, 90], 2, robust=robust)
    proj = ccrs.PlateCarree()
    extent = (-180, 180, -90, 90)
    figs = {}
    sz_map = figsize('short') if squeeze else None
    sz_hov = figsize('double') if wide else sz_map

    fig = plt.figure(figsize=sz_map)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()
    ax.gridlines()
    ax.imshow(mapn, vmin=-1, vmax=1, cmap='coolwarm', interpolation=iminterpol,
              transform=proj, extent=extent, origin='lower')
    ax.set_title(title.format(retrieval=''))
    tight_layout(fig)
    figs['spatial'] = fig

    extent = yrange + (-90, 90)
    fig = plt.figure(figsize=sz_hov)
    ax = plt.axes()
    ax.imshow(hovn, vmin=-1, vmax=1, cmap='coolwarm', interpolation=iminterpol,
              extent=extent, origin='lower', aspect='auto')
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(
                nbins='auto', steps=[1, 2, 5, 10], integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(30))
    ax.grid()
    ax.set_xlabel('Year')
    ax.set_ylabel('Latitude')
    ax.set_xlim(*yrange)
    ax.set_ylim(-90, 90)
    ax.set_title(title.format(retrieval=''))
    tight_layout(fig)
    figs['hov'] = fig
    return figs


def pvir_plot_ldist(data, varname, title='', wide=False, robust=False):
    if len(data.matchup) == 0:
        return None
    yrange = int(data.year.min()), int(data.year.max()+1)
    avg = groupby_2d(data, varname, 'year', yrange, 1./12,
                                    'sat_lland', [0, 3.5], 0.1, robust=robust)
    std = groupby_2d(data, varname, 'year', yrange, 1./12,
                                    'sat_lland', [0, 3.5], 0.1, std=True, robust=robust)
    extent = yrange + (0, 3.5)

    sz = figsize('double') if wide else None
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=sz)
    axs[0].imshow(std, vmin=0, vmax=2, interpolation=iminterpol,
                  extent=extent, origin='lower', aspect='auto')
    axs[1].imshow(avg, vmin=-1, vmax=1, cmap='coolwarm', interpolation=iminterpol,
                  extent=extent, origin='lower', aspect='auto')
    axs[0].grid()
    axs[1].grid()
    axs[1].set_xlabel('Year')
    fig.supylabel('Distance to Land / km', fontsize='medium')
    axs[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(
                nbins='auto', steps=[1, 2, 5, 10], integer=True))
    axs[0].yaxis.set_major_formatter(logtick2)
    axs[0].yaxis.set_major_locator(mpl.ticker.MultipleLocator())
    title = title.format(retrieval='')  # Assume we're always level 4 for now
    if robust:
        axs[0].set_title(f'{title} (RSD)')
        axs[1].set_title(f'{title} (median)')
    else:
        axs[0].set_title(f'{title} (SD)')
        axs[1].set_title(f'{title} (mean)')
    axs[1].set_xlim(*yrange)
    axs[1].set_ylim(0, 3.5)
    tight_layout(fig)
    return fig


def pvir_plot_uncert(ds, xaxis, yaxis, xbins=None, xrange=[0.0, 1.0], legend=True,
                     xstep=0.05, ins=None, min_data=100, title='', robust=True,
                     separate=True):
    insitu_u = {
        'drifter': 0.2,
        'drifter_cmems': 0.2,
        'mooring': 0.5,
        'ship':    1.0,
        'gtmba':   0.1,
        'argo':    0.005,
        'argosurf': 0.005,
        }
    if isinstance(ins, (float, int)):
        u_ins = ins
    else:
        u_ins = insitu_u.get(ins, np.median(ds.ins_sst_comb_unc))
    data = ds.set_coords(xaxis)[yaxis]
    xedges, xcentre, xrange = calc_bins(xbins, xrange, xstep, data[xaxis].values)
    theo = np.sqrt(xedges**2 + u_ins**2)
    ymax = max(theo.max(), *xrange)
    figs = {}
    funcs = {
        True: ['rsd', 'median', 'count', 'std'],
        False: ['std', 'mean', 'count'],
        }

    if separate:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        figs['uncertainty-day'] = fig1
        figs['uncertainty-night'] = fig2
        axes = {True: ax1, False: ax2}
        # So we can get the dpi below
        fig = fig1
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize('double'))
        figs['uncertainty'] = fig
        axes = {True: axs[0], False: axs[1]}

    for day in [True, False]:
        if day:
            dsel = data.where(ds.day, drop=True)
        else:
            dsel = data.where(ds.ngt, drop=True)
        try:
            # Ensure that first bin aligs with minimum value
            offset = dsel[xaxis].values.min()
            stats = calc_binned_stats(dsel, xaxis, xedges+offset, funcs=funcs[robust])
            if robust:
                bias = stats['median']
                uncr = stats['rsd']
            else:
                bias = stats['mean']
                uncr = stats['std']
            numb = stats['count'].fillna(0)
            if robust:
                stde = 1.253 * stats['std'] / np.sqrt(numb)
            else:
                stde = uncr / np.sqrt(numb)
            bias[numb < min_data] = np.nan
            uncr[numb < min_data] = 0
        except (StopIteration, ValueError):
            pass
        else:
            axs = axes[day]
            axs.set_xlim(xrange[0], xrange[1])
            axs.set_ylim(-ymax, ymax)
            axs.set_xlabel("Estimated uncertainty / K")
            axs.set_ylabel("Discrepancy / K")
            # Observed uncertainty is a polygon with points at bin centres.
            # We need to add start / ending points at the min and max x values
            # to align with the violin plot
            x1 = np.insert(stats[xaxis], 0, offset).values
            y1 = np.insert(uncr, 0, uncr[0]).values
            umax = dsel[xaxis].values.max()
            ind = np.searchsorted(x1, umax)
            if ind > 0 and ind < len(x1):
                x1[ind] = umax
                y1[ind:] = np.nan
                y1[ind] = y1[ind-1]
            axs.fill_between(x1, -y1, y1, fc='C7', ec='k', alpha=0.5, label='Observed')
            axs.axhline(0, color='k', linestyle='--')
            axs.plot(xedges, theo, color='C0', linestyle='-', label='Predicted')[0]
            axs.plot(xedges, -theo, color='C0', linestyle='-')
            # Estimate capsize
            trans = axs.transData.transform
            s = (trans((xstep, 0)) - trans((0, 0)))[0] * 72 / fig.dpi / 2
            axs.errorbar(stats[xaxis], bias, yerr=stde, fmt='none', capsize=s, color='C1', label='Bias')
            # Violon plot of the distribution of estimated uncertainties
            parts = axs.violinplot(dsel[xaxis].values,  vert=False,
                                   positions=[-ymax*5/6], widths=ymax/3)
            for p in parts:
                if p == 'bodies':
                    for pc in parts[p]:
                        pc.set_facecolor('C2')
                        pc.set_edgecolor('C2')
                else:
                    parts[p].set_edgecolor('C2')
            # axs.set_xlim(xrange[0], xrange[1])
            # axs.set_ylim(-ymax, ymax)
            # axs.set_xlabel("Estimated uncertainty / K")
            # axs.set_ylabel("Discrepancy / K")
            # Just callin gset_label will mess up the legend order. So need to
            # grab the current handle order, then explicitly append at the end
            handles, labels = axs.get_legend_handles_labels()
            parts['bodies'][0].set_label('Frequency')
            handles.append(parts['bodies'][0])
            if legend:
                axs.legend(handles=handles, loc='best')
            r = ' (day)' if day else ' (night)'
            axs.set_title(title.format(retrieval=r))
            tight_layout(fig)
    if not legend:
        p1 = mpl.patches.Patch(fc='C7', ec='k', alpha=0.5, label='Observed')
        p2 = mpl.lines.Line2D([], [], color='C0', label='Predicted')
        p3 = handles[2]
        p4 = mpl.patches.Patch(fc='C2', ec='C2', alpha=0.3, label='Frequency')

        fig = plt.figure(figsize=figsize([0.6, 0.2]))
        fig.legend(handles=[p1, p2, p3, p4], ncols=2, loc='center')
        figs['legend'] = fig
    return figs


def basic_plots(ds, prefix='', title=''):
    title = title.format(retrieval='')
    fig, axs = plt.subplots()
    axs.ticklabel_format(axis='y', scilimits=(-2, 2))
    axs.hist(ds.dtime, 40, range=[-2, 2])
    axs.set_xlim(-2, 2)
    axs.set_xlabel("Time difference / hour")
    axs.set_ylabel('Number of matches')
    if title:
        axs.set_title(title + r': SST$_\mathrm{0.2m}$ @ 10:30')
    tight_layout(fig)
    fig.savefig(f'{prefix}-matchup_tdiff.png')
    fig.savefig(f'{prefix}-matchup_tdiff.svg')
    plt.close(fig)
    fig, axs = plt.subplots()
    axs.ticklabel_format(axis='y', scilimits=(-2, 2))
    axs.hist(ds.dtime, 48, range=[-12, 12])
    axs.set_xlim(-12, 12)
    axs.set_xlabel("Time difference / hour")
    axs.set_ylabel('Number of matches')
    if title:
        axs.set_title(title + r': SST$_\mathrm{0.2m}$ @ 10:30')
    tight_layout(fig)
    fig.savefig(f'{prefix}-matchup_tdiff12.png')
    fig.savefig(f'{prefix}-matchup_tdiff12.svg')
    plt.close(fig)
    fig = plt.figure()
    axs = plt.axes(projection=ccrs.PlateCarree())
    axs.coastlines()
    axs.set_global()
    axs.gridlines()
    axs.plot(ds.ins_longitude, ds.ins_latitude, ',')
    axs.set_title(title)
    tight_layout(fig)
    fig.savefig(f'{prefix}-matchup_position.png')
    # fig.savefig(f'{prefix}-matchup_position.svg')  # Too big
    plt.close(fig)
    n = np.histogram2d(ds.ins_longitude, ds.ins_latitude, [180, 90], [[-180, 180], [-90, 90]])
    n = np.where(n[0] > 0, n[0], np.nan).T
    fig = plt.figure()
    axs = plt.axes(projection=ccrs.PlateCarree())
    im = axs.imshow(n, extent=(-180, 180, -90, 90), origin='lower', norm=mpl.colors.LogNorm(), interpolation=iminterpol)
    fig.colorbar(im, location='bottom', label='Number')
    axs.coastlines()
    axs.gridlines()
    axs.set_title(title)
    tight_layout(fig)
    fig.savefig(f'{prefix}-matchup_position2.png')
    fig.savefig(f'{prefix}-matchup_position2.svg')
    plt.close(fig)
    fig, axs = plt.subplots()
    axs.ticklabel_format(axis='y', scilimits=(-2, 2))
    # axs.hist(ds.ins_time, 365*2)
    # Aim for approx 30-day bins
    axs.hist(ds.ins_time, np.ptp(ds.sat_time.values) // np.timedelta64(30, 'D'))
    axs.set_xlabel("Date")
    axs.set_ylabel('Number of matches')
    axs.set_title(title)
    fig.savefig(f'{prefix}-matchup_time.png')
    fig.savefig(f'{prefix}-matchup_time.svg')
    tight_layout(fig)
    plt.close(fig)
    if 'sat_lland' in ds:
        lbins = np.logspace(0, 3.5, 15)
        fig, axs = plt.subplots()
        axs.ticklabel_format(axis='y', scilimits=(-2, 2))
        axs.hist(ds.sat_land, lbins)
        axs.set_xscale('log')
        # axs.set_xlim(1e-2, 1e4)
        axs.set_xlim(1e0, 10**3.5)
        axs.set_xlabel('Distance to land / km')
        axs.set_ylabel('Number of matches')
        axs.set_title(title)
        tight_layout(fig)
        fig.savefig(f'{prefix}-matchup_landdist.png')
        fig.savefig(f'{prefix}-matchup_landdist.svg')
        plt.close(fig)
    if 'sat_unc' in ds:
        fig, axs = plt.subplots()
        axs.ticklabel_format(axis='y', scilimits=(-2, 2))
        axs.hist(ds.sat_unc, 30, range=[0, 3])
        axs.set_xlim(0, 3)
        axs.set_xlabel('Estimated uncertainty / K')
        axs.set_ylabel('Number of matches')
        axs.set_title(title)
        tight_layout(fig)
        fig.savefig(f'{prefix}-matchup_unc.png')
        fig.savefig(f'{prefix}-matchup_unc.svg')
        plt.close(fig)
    xe, xc, xr = calc_bins(range=[int(ds.year.min().item()), int(ds.year.max().item()+.999)], step=1/12)
    ye, yc, yr = calc_bins(range=[-4,4], step=1/12)
    d1 = ds[['year', 'dtime']].where(ds.day)
    d2 = ds[['year', 'dtime']].where(~ds.day)
    n1, _, _ = np.histogram2d(d1.year, d1.dtime, [xe, ye])
    n1[n1==0] = np.nan
    n2, _, _ = np.histogram2d(d2.year, d2.dtime, [xe, ye])
    n2[n2==0] = np.nan
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    axs[0].imshow(n1.T, extent=[xr[0], xr[1], yr[0], yr[1]], aspect='auto', vmin=0, vmax=50, interpolation='nearest')
    axs[1].imshow(n2.T, extent=[xr[0], xr[1], yr[0], yr[1]], aspect='auto', vmin=0, vmax=50, interpolation='nearest')
    axs[0].set_title(f'{title} (day)')
    axs[1].set_title(f'{title} (night)')
    axs[1].set_xlabel('Year')
    axs[0].set_ylabel('Time difference / hour')
    axs[1].set_ylabel('Time difference / hour')
    tight_layout(fig)
    fig.savefig(f'{prefix}-matchup_time_tdiff.png')
    fig.savefig(f'{prefix}-matchup_time_tdiff.svg')
    plt.close(fig)


def get_unit_offset(input_unit):
    """
    Check satellite input units and add offset if needed.

    Parameters
    ----------
    input_unit : string
        Satellite dataset units
    """

    input_unit = str(input_unit)

    if input_unit in ["kelvin", "K"]:
        unit_corr = -273.15
    elif input_unit in ["celsius", "C"]:
        unit_corr = 0.0
    else:
        raise Exception(f"ERROR: Satellite data unit: {input_unit} not recognised!")
    return unit_corr


def add_features(axs, features, color='k', offsets=[-0.25, -0.35, -0.45]):
    """Add volcanic eruptions or other features to timeseries plots.

    Parameters
    ----------
    dims : int
        Set to 1 to add vertical lines to a timeseries; or 2 to add symbols at
        lat, time to a hovmoller plot
    axs : matplotlib axis
        Axis
    """
    if isinstance(features, dict):
        for k, v in sorted(features.items(), key=lambda x: x[1]):
            axs.axvline(v)
            axs.annotate(k, (v, offsets[0]))
            offsets.append(offsets.pop(0))
    elif isinstance(features, list):
        for x in features:
            if isinstance(x, list):
                axs.axvline(x[0], color=x[1], linestyle='--')
            else:
                axs.axvline(x, color=color, linestyle='--')
    elif len(np.shape(features)) == 2:
        size = 2 * mpl.rcParams['lines.markersize']
        axs.plot(*features, 'x', color=color, markersize=size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Plotting config file')
    parser.add_argument('--files', nargs='*', help="MMD file[s] to process")
    parser.add_argument('--title', default=None, help='Plot title (default is product name)')
    parser.add_argument('--dir', help='Output directory')
    parser.add_argument('--quality', help='L2 quality levels to use')
    parser.add_argument('--min_filequal', type=int, help='Minimum file quality level')
    parser.add_argument('--prefix')
    parser.add_argument('--product', help='Override product name')
    parser.add_argument('--insitu_unc', help='Assumed in situ uncertainty')
    parser.add_argument('--sirds_type', help='Override sirds type in plots filename')
    parser.add_argument('--dtime', type=float, help='Delta-time limit (hours)')
    args = parser.parse_args()

    conf = yaml.safe_load(open(args.config))
    # Override config file with command line arguments
    conf.update({k: v for k, v in vars(args).items() if v is not None})
    set_scale(0.5)

    if 'files' in conf:
        # Expand any wild cards in the config file list
        conf['files'] = list(itertools.chain(*(glob.glob(f) for f in conf['files'])))
    else:
        raise Exception('Must specify input files in config or command line')

    ds = xr.open_mfdataset(conf['files'], combine='nested', concat_dim='matchup', decode_timedelta=False)
    ds.load()

    conf['sirds_type'] = conf.get('sirds_type', ds.attrs.get('sirds_type', ''))
    conf['product'] = conf.get('product', ds.attrs.get('product', 'SST'))
    conf['dir'] = conf.get('dir', ds.attrs.get('gbcsver', ''))
    conf['title'] = conf.get('title', conf['product']+'{retrieval}')

    # Specifying an assumed in situ uncertainty for the plots has some special cases
    u = conf.get('insitu_unc')
    if u == 'sirds_type':
        u = conf['sirds_type']
    try:
        u = float(u)
    except (ValueError, TypeError):
        pass
    conf['insitu_unc'] = u

    ql = conf.get('quality', [4, 5])
    if isinstance(ql, str):
        ql = [int(i) for i in ql.split(',')]
    conf['quality'] = ql

    # Default filenaming scheme
    prefix = '{product}-{sirds_type}'
    if 'regions' in conf:
        prefix = '{region}-' + prefix
    prefix = conf.get('prefix', prefix)

    if conf['dir']:
        os.makedirs(conf['dir'], exist_ok=True)
        prefix = os.path.join(conf['dir'], prefix)

    conf['prefix'] = prefix.format(region='{region}', **conf)

    if 'processing_level' not in ds.attrs:
        if 'sat_l2p_flags' in ds:
            ds.attrs['processing_level'] = 'L3'
        else:
            ds.attrs['processing_level'] = 'L4'

    if ds.attrs['processing_level'] == 'L4':
        ret = 'L4'
    elif 'AVHRR' in ds.product.upper():
        ret = 'N'
    else:
        ret = 'D'

    # Apply in situ SST corrections
    # Fix for inconsistent masking of _FillValue in earlier processings
    ds['ins_sst_type_corr'] = ds.ins_sst_type_corr.where(ds.ins_sst_type_corr != -9999)
    ds['ins_sst_plat_corr'] = ds.ins_sst_plat_corr.where(ds.ins_sst_plat_corr != -9999)
    print(f'Applying SST_TYPE_CORR: {ds.ins_sst_type_corr.values.min():.2f} to {ds.ins_sst_type_corr.values.max():.2f}')
    print(f'Applying SST_PLAT_CORR: {ds.ins_sst_plat_corr.values.min():.2f} to {ds.ins_sst_plat_corr.values.max():.2f}')
    ds['ins_sst'] = ds.ins_sst + ds.ins_sst_type_corr.fillna(0) + ds.ins_sst_plat_corr.fillna(0)

    print('In situ combined uncertainty:')
    ins_unc = ds.ins_sst_comb_unc.values
    print(f' median: {np.median(ins_unc):.2f} range: {ins_unc.min():.2f}-{ins_unc.max():.2f}')

    # Fractional year based on day
    ds['year'] = ds.sat_time.dt.year + (ds.sat_time.dt.dayofyear-1)/365.25
    # Fractional year based on only month (so plots etc will align on month )
    # ds['year'] = ds.sat_time.dt.year + (ds.sat_time.dt.month-0.5)/12
    ds['year'].attrs['long_name'] = 'Year'
    # Correct SST unit values if needed.
    sat_unit_corr = get_unit_offset(ds.sat_sst.units)
    ds.sat_sst.data = ds.sat_sst.data + sat_unit_corr
    ds['tdiff'] = ds.sat_sst - ds.ins_sst
    ds['tdiff'].attrs['long_name'] = 'depth-depth'
    ds['tdiff'].attrs['units'] = 'K'
    ds['dtime'] = (ds.sat_time - ds.ins_time) / np.timedelta64(1, 'h')
    ds['dtime'].attrs['long_name'] = 'Time difference'
    ds['dtime'].attrs['units'] = 'hour'
    if 'sat_land' in ds:
        # Set minimum distance to land to 1km
        ds['sat_land'] = ds.sat_land.where(ds.sat_land>1, 1)
        ds['sat_lland'] = np.log10(ds.sat_land)
        ds['sat_lland'].attrs['xlog'] = True
    if ds.attrs['processing_level'] == 'L4':
        ds['day'] = ds.year < 0
        ds['day'].attrs['long_name'] = 'Day flag'
        ds['ngt'] = np.invert(ds.day)
    else:
        ds['day'] = ds.sat_l2p_flags.astype(int) & 256 > 0
        ds['day'].attrs['long_name'] = 'Day flag'
        ds['ngt'] = np.invert(ds.day)

    msk = ds.ins_qc1 == 0
    if 'min_filequal' in conf:
        if 'file_quality_level' in ds:
            msk = msk & (ds.file_quality_level >= conf['min_filequal'])
        else:
            print('Dataset does not contain file_quality_level - ignoring min file qual request')
    if 'dtime' in conf:
        # 0.5 hour time difference is suitable for recent MDs, but will need
        # larger window for historical data
        print(f'Removing matches with dtime > {conf["dtime"]}')
        msk = msk & (np.abs(ds.dtime) < conf['dtime'])
    if ds.attrs['processing_level'] == 'L4':
        msk = msk & (ds.sat_mask == 1) & np.isfinite(ds.sat_sst)
    else:
        msk = msk & ds.sat_quality_level.isin(conf['quality'])

    if ds.sirds_type in ['bottle', 'mbt', 'ship']:
        umax = 1.5
        dmax = 1.0
    else:
        umax = 0.75
        dmax = 0.5

    umax = conf.get('umax', umax)
    dmax = conf.get('dmax', dmax)

    def tofracyear(x):
        if isinstance(x, list):
            x[0] = x[0].year + (x[0].timetuple().tm_yday-1)/365.25
        else:
            x = x.year + (x.timetuple().tm_yday-1)/365.25
        return x

    # Convert feature items to fractional year
    features = conf.get('features', {})
    if isinstance(features, dict):
        features = {k: tofracyear(v) for k, v in features.items()}
    elif isinstance(features, list):
        features = [tofracyear(v) for v in features]
    else:
        raise Exception(f'Unexpected features type: {type(features)}')
    conf['features'] = features

    # And same for located features (time, latitude)
    if 'features2' in conf:
        features = conf.get('features2', [])
        conf['features2'] = np.array([[tofracyear(x), y] for x, y in features]).T
    else:
        conf['features2'] = conf.get('features')

    set_simple('')

    # Create a set of basic matchup plots before any filtering is applied to dataset
    basic_plots(ds, conf['prefix'].format(region='ALL')+'-0',
                title=conf['title'].format(sirds_name=_insitu_names[conf['sirds_type']],
                                           region='', retrieval='', **conf))

    # Use an empty "global" region if none have been specified in config file
    if 'regions' not in conf:
        conf['regions'] = {'global': {}}

    for region in conf['regions']:
        print(f'Processing region: {region}')
        prefix = conf['prefix'].format(region=region)
        rdef = conf['regions'][region]
        ptitle = conf['title'].format(sirds_name=_insitu_names[conf['sirds_type']],
                                      region=rdef.get('name', region),
                                      retrieval='{retrieval}',
                                      **conf)
        rmsk = msk.copy()
        # Latitude range
        if 'lat' in rdef:
            lim = rdef['lat']
            rmsk = rmsk & (ds.sat_lat >= lim[0]) & (ds.sat_lat < lim[1])
        # Longitude range (with wrapping)
        if 'lon' in rdef:
            lim = rdef['lon']
            if lim[0] > lim[1]:
                rmsk = rmsk & ((ds.sat_lon >= lim[0]) | (ds.sat_lon < lim[1]))
            else:
                rmsk = rmsk & (ds.sat_lon >= lim[0]) & (ds.sat_lon < lim[1])
        # Distance to land
        if 'min_land' in rdef:
            rmsk = rmsk & (ds['sat_land'] >= rdef['min_land'])
        if 'max_land' in rdef:
            rmsk = rmsk & (ds['sat_land'] <= rdef['max_land'])
        if 'time' in rdef:
            lim = [np.datetime64(d) for d in rdef['time']]
            if lim[0] > lim[1]:
                rmsk = rmsk & ((ds.sat_time >= lim[0]) | (ds.sat_time < lim[1]))
            else:
                rmsk = rmsk & (ds.sat_time >= lim[0]) & (ds.sat_time < lim[1])

        if not np.any(rmsk):
            print("No matches available")
            continue

        qc5 = ds.isel(matchup=rmsk)

        basic_plots(qc5, prefix, title=ptitle)

        print("Histogram plots")
        if ret == 'L4':
            fig = pvir_plot_hist(qc5, ret, 'tdiff', title=ptitle)
            fig.savefig(f'{prefix}-histogram.png')
            fig.savefig(f'{prefix}-histogram.svg')
            plt.close(fig)
        else:
            fig = pvir_plot_hist(qc5.where(qc5.ngt), ret+'3', 'tdiff', title=ptitle)
            fig.savefig(f'{prefix}-histogram-night-d3.png')
            fig.savefig(f'{prefix}-histogram-night-d3.svg')
            plt.close(fig)
            fig = pvir_plot_hist(qc5.where(qc5.day), ret+'2', 'tdiff', title=ptitle)
            fig.savefig(f'{prefix}-histogram-day-d2.png')
            fig.savefig(f'{prefix}-histogram-day-d2.svg')
            plt.close(fig)

        print("Dependence plots")
        plotopts = dict(yrange=[-dmax, dmax],
                        erange=[0, umax],
                        ret=ret,
                        title=ptitle)
        yrange = int(qc5.year.min()), int(qc5.year.max()+1)
        wide = (yrange[1] - yrange[0]) > 19
        if ret == 'L4':
            pfunc = pvir_plot_dependence1
        else:
            pfunc = pvir_plot_dependence

        fig = pfunc(qc5, 'year', xrange=yrange, xstep=1/12, wide=wide, **plotopts)
        add_features(fig.get_axes()[0], conf['features'])
        add_features(fig.get_axes()[1], conf['features'])
        fig.savefig(f'{prefix}-dependence-month.png')
        fig.savefig(f'{prefix}-dependence-month.svg')
        plt.close(fig)

        fig = pfunc(qc5, 'sat_lat', xrange=[-75, 80], xstep=5.0, **plotopts)
        fig.savefig(f'{prefix}-dependence-sat_lat.png')
        fig.savefig(f'{prefix}-dependence-sat_lat.svg')
        plt.close(fig)

        if 'sat_lland' in qc5:
            fig = pfunc(qc5, 'sat_lland', xrange=[0, 3.5], xstep=0.1, **plotopts)
            fig.savefig(f'{prefix}-dependence-sat_land.png')
            fig.savefig(f'{prefix}-dependence-sat_land.svg')
            plt.close(fig)

        if ret == 'L4':
            fig = pfunc(qc5, 'dtime', xrange=[-12, 12], xstep=1, **plotopts)
            fig.savefig(f'{prefix}-dependence-tdiff.png')
            fig.savefig(f'{prefix}-dependence-tdiff.svg')
            plt.close(fig)

            fig = pfunc(qc5, 'sat_unc', xrange=[0, 3], xstep=0.1,
                        xtitle='Estimated uncertainty / K', **plotopts)
            fig.savefig(f'{prefix}-dependence-unc.png')
            fig.savefig(f'{prefix}-dependence-unc.svg')
            plt.close(fig)

        else:
            fig = pfunc(qc5, 'dtime', xrange=[-2, 2], xstep=0.25, **plotopts)
            fig.savefig(f'{prefix}-dependence-tdiff.png')
            fig.savefig(f'{prefix}-dependence-tdiff.svg')
            plt.close(fig)

            fig = pfunc(qc5, 'sat_wind_speed', xrange=[0, 25], xstep=0.25, **plotopts)
            fig.savefig(f'{prefix}-dependence-wind.png')
            fig.savefig(f'{prefix}-dependence-wind.svg')
            plt.close(fig)

        print("Spatial plots")
        if ret == 'L4':
            spatial = pvir_plot_spatial_l4(qc5, 'tdiff', title=ptitle, wide=wide, robust=True)
            add_features(spatial['hov'].get_axes()[0], conf['features2'], offsets=[-89, -82, -75])
        else:
            spatial = pvir_plot_spatial(qc5, 'tdiff', title=ptitle, wide=wide, robust=True)
        for name, fig in spatial.items():
            try:
                add_features(fig.get_axes()[0], conf['features2'], offsets=[-89, -82, -75])

                fig.savefig(f'{prefix}-{name}.png')
                fig.savefig(f'{prefix}-{name}.svg')

                # Special case add colorbar to pdf
                fig.set_figheight(fig.get_figheight() + 0.5)
                fig.colorbar(fig.axes[0].images[0], orientation='horizontal',
                             label='Satellite - in situ / K', aspect=15)
                fig.savefig(f'{prefix}-{name}.pdf')
                plt.close(fig)
            except AttributeError:
                # Spatial plots fail if there is no data.
                plt.close(fig)

        if ret == 'L4':
            spatial = pvir_plot_spatial_l4(qc5, 'tdiff', title=ptitle, wide=wide, robust=False)
            add_features(spatial['hov'].get_axes()[0], conf['features2'], offsets=[-89, -82, -75])
        else:
            spatial = pvir_plot_spatial(qc5, 'tdiff', title=ptitle, wide=wide, robust=False)
        for name, fig in spatial.items():
            try:
                fig.savefig(f'{prefix}-{name}-nr.png')
                fig.savefig(f'{prefix}-{name}-nr.svg')
                plt.close(fig)
            except AttributeError:
                plt.close(fig)

        if 'sat_lland' in qc5:
            fig = pvir_plot_ldist(qc5, 'tdiff', ptitle, wide=wide, robust=True)
            add_features(fig.get_axes()[0], conf['features'], offsets=[-89, -82, -75])
            add_features(fig.get_axes()[1], conf['features'], offsets=[-89, -82, -75])
            if fig:
                fig.savefig(f'{prefix}-hov-dist.png')
                fig.savefig(f'{prefix}-hov-dist.svg')
                plt.close(fig)

            fig = pvir_plot_ldist(qc5, 'tdiff', ptitle, wide=wide)
            add_features(fig.get_axes()[0], conf['features'], offsets=[-89, -82, -75])
            add_features(fig.get_axes()[1], conf['features'], offsets=[-89, -82, -75])
            if fig:
                fig.savefig(f'{prefix}-hov-dist-nr.png')
                fig.savefig(f'{prefix}-hov-dist-nr.svg')
                plt.close(fig)

        print("Uncertainty plots")
        if ret == 'L4':
            urange = [0, 2]
            ustep = 0.05
        else:
            urange = [0, 1]
            ustep = 0.05
        ulegend = False   # Generate a separate legend plot
        uncer = pvir_plot_uncert(qc5, 'sat_unc', 'tdiff', ins=conf['insitu_unc'],
                                 min_data=100, title=ptitle, xrange=urange, xstep=ustep,
                                 legend=ulegend)

        if not ulegend:
            fig = uncer['legend']
            fig.savefig(os.path.join(conf['dir'], 'legend_uncert.png'))
            fig.savefig(os.path.join(conf['dir'], 'legend_uncert.svg'))
            plt.close(fig)

        for name, fig in uncer.items():
            if name == 'legend':
                continue
            try:
                fig.savefig(f'{prefix}-{name}.png')
                fig.savefig(f'{prefix}-{name}.svg')
                fig.set_size_inches(2.5, 2.0)
                fig.savefig(f'{prefix}-{name}-small.svg')
                plt.close(fig)
            except AttributeError:
                plt.close(fig)

        # For pdf we want a combined plot
        uncer = pvir_plot_uncert(qc5, 'sat_unc', 'tdiff', ins=conf['insitu_unc'],
                                 min_data=100, title=ptitle, xrange=urange, xstep=ustep,
                                 legend=ulegend, separate=False)
        name = 'uncertainty'
        try:
            fig = uncer[name]
            fig.savefig(f'{prefix}-{name}.pdf')
            fig.set_size_inches(5.0, 2.0)
            fig.savefig(f'{prefix}-{name}-small.pdf')
            plt.close(fig)
        except AttributeError:
            plt.close(fig)

        uncer = pvir_plot_uncert(qc5, 'sat_unc', 'tdiff', ins=conf['insitu_unc'],
                                 min_data=100, title=ptitle, xrange=urange, xstep=ustep,
                                 legend=ulegend, robust=False)
        for name, fig in uncer.items():
            if name == 'legend':
                continue
            try:
                fig.savefig(f'{prefix}-{name}-nr.png')
                fig.savefig(f'{prefix}-{name}-nr.svg')
                plt.close(fig)
            except AttributeError:
                plt.close(fig)

        # Color bar
        # fig, axs = plt.subplots(figsize=(6.4, 1.6))
        fig, axs = plt.subplots(figsize=(3.2, 0.5))
        scl = mpl.cm.ScalarMappable(mpl.colors.Normalize(-1, 1), 'coolwarm')
        fig.colorbar(scl, cax=axs, orientation='horizontal',
                     label='Satellite - in situ / K')
        tight_layout(fig)
        fig.savefig(os.path.join(conf['dir'], 'colourbar.png'))
        fig.savefig(os.path.join(conf['dir'], 'colourbar.svg'))
        plt.close(fig)

        fig, axs = plt.subplots(figsize=(3.2, 0.5))
        scl = mpl.cm.ScalarMappable(mpl.colors.Normalize(0, 2))
        fig.colorbar(scl, cax=axs, orientation='horizontal',
                     label='Standard Deviation / K')
        tight_layout(fig)
        fig.savefig(os.path.join(conf['dir'], 'colourbar_sd.png'))
        fig.savefig(os.path.join(conf['dir'], 'colourbar_sd.svg'))
        plt.close(fig)

        fig, axs = plt.subplots(figsize=(3.2, 0.5))
        scl = mpl.cm.ScalarMappable(mpl.colors.Normalize(0, 2))
        fig.colorbar(scl, cax=axs, orientation='horizontal',
                     label='Robust Standard Deviation / K')
        tight_layout(fig)
        fig.savefig(os.path.join(conf['dir'], 'colourbar_rsd.png'))
        fig.savefig(os.path.join(conf['dir'], 'colourbar_rsd.svg'))
        plt.close(fig)
