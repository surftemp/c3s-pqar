#!/usr/bin/env python3

import argparse
import collections
import os

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import scipy.optimize
import xarray as xr

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

mpl.rcParams['savefig.dpi'] = 200

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


def pvir_plot_hist(data, ret, xaxis, xbins=80, xrange=(-4, 4), day=False,
                   legend=True, histtype=None, showstats=True, title=None):
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
    axs.plot(xval, hist, label=label, color=color, linestyle=style, lw=2)
    try:
        p0 = [hist.max(), 0.0, 1.0]
        coef, vmat = scipy.optimize.curve_fit(_gauss, xval.astype(float),
                                              hist.astype(float), p0=p0)
        axs.plot(xval, _gauss(xval, *coef), color=color, linestyle='--')
    except RuntimeError:
        coef = None
    if showstats and np.isfinite(diff).sum() > 3:
        color = 'none'
        m, s = np.nanmean(diff), np.nanstd(diff)
        axs.axvline(m, color=color, linestyle=':',
                    label='Normal:\n{:+5.2f} ({:4.2f})'.format(m, s))
        m, s = np.nanmedian(diff), rsd(diff).data[()]
        axs.axvline(m, color=color, linestyle='-.',
                    label='Robust:\n{:+5.2f} ({:4.2f})'.format(m, s))
        if coef is not None:
            m, s = coef[1], coef[2]
            axs.axvline(m, color=color, linestyle='--',
                        label='Gaussian fit:\n{:+5.2f} ({:4.2f})'.format(m, s))
    if legend:
        axs.legend(loc='best')
        axs.set_ylim(0)
    if title:
        axs.set_title(title)
    axs.set_xlim(*xrange)
    axs.set_xlabel('Satellite - in situ / K')
    fig.tight_layout()
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


def pvir_plot_dependence(data, xaxis, yaxis='tdiff', xbins=10, xrange=None,
                         xstep=None, title=None, xtitle=None, legend=True,
                         robust=True, yrange=None, erange=None, ret='N'):
    """
    Generate a PVIR dependence plot

    Parameters
    ----------
    data : xarray.Dataset
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
    vars = [xaxis, yaxis, 'matchup']
    xedges, xcentre, xrange = calc_bins(xbins, xrange, xstep, data[xaxis].data)
    day = data[vars].where(data.day).groupby_bins(xaxis, xedges)
    ngt = data[vars].where(data.ngt).groupby_bins(xaxis, xedges)
    if robust:
        day2 = day.median('matchup')
        day1 = day.apply(rsd, dim='matchup')
        ngt2 = ngt.median('matchup')
        ngt1 = ngt.apply(rsd, dim='matchup')
    else:
        day2 = day.mean('matchup')
        day1 = day.std('matchup')
        ngt2 = ngt.mean('matchup')
        ngt1 = ngt.std('matchup')
    fig, (ax1, ax2) = plt.subplots(2, sharex=True,
                                   gridspec_kw={'height_ratios': (1, 2)})
    ds = cci_plot_style[ret+'2'][0]
    ns = cci_plot_style[ret+'3'][1]
    ax1.plot(xcentre, day1[yaxis], color=ds.color, linestyle=ds.style, lw=2)
    ax2.plot(xcentre, day2[yaxis], color=ds.color, linestyle=ds.style, lw=2,
             label=ret+'2 (day)')
    ax1.plot(xcentre, ngt1[yaxis], color=ns.color, linestyle=ns.style, lw=2)
    ax2.plot(xcentre, ngt2[yaxis], color=ns.color, linestyle=ns.style, lw=2,
             label=ret+'3')
    if robust:
        ax1.set_ylabel('RSD / K')
        ax2.set_ylabel('Median / K')
    else:
        ax1.set_ylabel('Standard Deviation / K')
        ax2.set_ylabel('Mean / K')
    if not xtitle:
        xtitle = data[xaxis].long_name
        if 'units' in data[xaxis].attrs:
            xtitle = xtitle + ' / ' + data[xaxis].units
    ax2.set_xlabel(xtitle)
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_xlim(*xrange)
    if yrange:
        ax2.set_ylim(*yrange)
    if erange:
        ax1.set_ylim(*erange)
    if legend:
        ax2.legend(loc='best', ncol=2)
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    plt.ticklabel_format(useOffset=False)
    return fig


def pvir_plot_dependence1(data, xaxis, yaxis='tdiff', xbins=10, xrange=None,
                          xstep=None, title=None, xtitle=None, legend=True,
                          robust=True, yrange=None, erange=None, ret='N'):
    """
    Generate a PVIR dependence plot

    Parameters
    ----------
    data : xarray.Dataset
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
    vars = [xaxis, yaxis, 'matchup']
    xedges, xcentre, xrange = calc_bins(xbins, xrange, xstep, data[xaxis].data)
    day = data[vars].groupby_bins(xaxis, xedges)
    if robust:
        day2 = day.median('matchup')
        day1 = day.apply(rsd, dim='matchup')
    else:
        day2 = day.mean('matchup')
        day1 = day.std('matchup')
    fig, (ax1, ax2) = plt.subplots(2, sharex=True,
                                   gridspec_kw={'height_ratios': (1, 2)})
    ds = cci_plot_style[ret][1]
    ax1.plot(xcentre, day1[yaxis], color=ds.color, linestyle=ds.style, lw=2)
    ax2.plot(xcentre, day2[yaxis], color=ds.color, linestyle=ds.style, lw=2,
             label=ret)
    if robust:
        ax1.set_ylabel('RSD / K')
        ax2.set_ylabel('Median / K')
    else:
        ax1.set_ylabel('Standard Deviation / K')
        ax2.set_ylabel('Mean / K')
    if not xtitle:
        xtitle = data[xaxis].long_name
        if 'units' in data[xaxis].attrs:
            xtitle = xtitle + ' / ' + data[xaxis].units
    ax2.set_xlabel(xtitle)
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_xlim(*xrange)
    if yrange:
        ax2.set_ylim(*yrange)
    if erange:
        ax1.set_ylim(*erange)
    if legend:
        ax2.legend(loc='best', ncol=2)
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    plt.ticklabel_format(useOffset=False)
    return fig


def groupby_2d(data, varname, xaxis, xrange, xstep, yaxis, yrange, ystep,
               robust=False, count=False):
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
    data['index'] = xr.DataArray(ind, coords=[data.matchup])
    vars = ['index', varname]
    temp = np.full(xbin*ybin, np.nan)
    try:
        if robust:
            avg = data[vars].groupby('index').median('matchup')
            # day1 = data[vars].groupby('index').apply(rsd, dim='matchup')
        else:
            avg = data[vars].groupby('index').mean('matchup')
            # day1 = data[vars].groupby('index').std('matchup')
        temp[avg.index.data] = avg[varname]
    except (StopIteration, ValueError):
        pass
    out = np.reshape(temp, [ybin, xbin])
    if count:
        num = np.reshape(np.bincount(ind, minlength=xbin*ybin), [ybin, xbin])
        return out, num
    else:
        return out


def pvir_plot_spatial(data, varname, title=None):
    day = data.where(data.day, drop=True)
    ngt = data.where(data.ngt, drop=True)
    yrange = int(data.year.min()), int(data.year.max()+1)
    mapn = groupby_2d(ngt, varname, 'sat_lon', [-180, 180], 2.5,
                                    'sat_lat', [-90, 90], 2.5)
    mapd = groupby_2d(day, varname, 'sat_lon', [-180, 180], 2.5,
                                    'sat_lat', [-90, 90], 2.5)
    hovn = groupby_2d(ngt, varname, 'year', yrange, 1./12,
                                    'sat_lat', [-90, 90], 2)
    hovd = groupby_2d(day, varname, 'year', yrange, 1./12,
                                    'sat_lat', [-90, 90], 2)
    proj = ccrs.PlateCarree()
    extent = (-180, 180, -90, 90)
    figs = {}

    def mapplot(d, t=None):
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_global()
        ax.imshow(d, vmin=-1, vmax=1, cmap='coolwarm', interpolation='none',
                  transform=proj, extent=extent, origin='lower')
        if t:
            ax.set_title(t)
        fig.tight_layout()
        return fig
    figs['spatial-day'] = mapplot(mapd, f'{title} day')
    figs['spatial-night'] = mapplot(mapn, f'{title} night')
    extent = yrange + (-90, 90)

    def hovplot(d, t=None):
        fig = plt.figure()
        ax = plt.axes()
        ax.imshow(d, vmin=-1, vmax=1, cmap='coolwarm', interpolation='none',
                  extent=extent, origin='lower', aspect='auto')
        ax.set_xlabel('Year')
        ax.set_ylabel('Latitude')
        if t:
            ax.set_title(t)
        fig.tight_layout()
        return fig
    figs['hov-day'] = hovplot(hovd, f'{title} day')
    figs['hov-night'] = hovplot(hovn, f'{title} night')
    return figs


def pvir_plot_uncert(data, xaxis, yaxis, xbins=None, xrange=[0.0, 1.0],
                     xstep=0.05, ins=None, min_data=100, title=None):
    if ins == 'drifter':
        u_ins = 0.2
    elif ins == 'moored':
        u_ins = 0.5
    elif ins == 'ship':
        u_ins = 1.0
    elif ins == 'gtmba':
        u_ins = 0.1
    elif ins == 'argo':
        u_ins = 0.005
    else:
        u_ins = np.median(data.ins_sst_comb_unc)
    vars = [xaxis, yaxis, 'matchup']
    xedges, xcentre, xrange = calc_bins(xbins, xrange, xstep, data[xaxis].data)
    theo = np.sqrt(xcentre**2 + u_ins**2)
    ymax = max(theo.max(), *xrange)
    figs = {}
    dsel = data[vars]
    for day in [True, False]:
        try:
            if day:
                grp = dsel.where(data.day).groupby_bins(xaxis, xedges)
            else:
                grp = dsel.where(data.ngt).groupby_bins(xaxis, xedges)
            bias = grp.median('matchup')[yaxis]
            uncr = grp.apply(rsd, dim='matchup')[yaxis]
            numb = grp.count('matchup')[yaxis]
        except (StopIteration, ValueError):
            pass
        else:
            stde = uncr / np.sqrt(numb)
            bias[numb < min_data] = np.nan
            uncr[numb < min_data] = np.nan
            fig, axs = plt.subplots()
            axs.axhline(0, color='k', linestyle='--')
            axs.plot(xcentre, theo, color='g', linestyle='--')
            axs.plot(xcentre, -theo, color='g', linestyle='--')
            axs.errorbar(xcentre, 0*xcentre, fmt='none', ecolor='c', yerr=uncr)
            axs.errorbar(xcentre, bias, fmt='none', ecolor='r', yerr=stde,
                         elinewidth=2, xerr=0.005)
            axs.set_xlim(xrange[0], xrange[1])
            axs.set_ylim(-ymax, ymax)
            axs.set_xlabel("SST uncertainty / K")
            axs.set_ylabel("Measure of discrepancy / K")
            if title:
                axs.set_title(title)
            fig.tight_layout()
            if day:
                figs['uncertainty-day'] = fig
            else:
                figs['uncertainty-night'] = fig
    return figs


def basic_plots(ds, prefix='', title=None):
    fig, axs = plt.subplots()
    axs.hist(ds.dtime, 40, [-2, 2])
    axs.set_xlim(-2, 2)
    axs.set_xlabel("Time difference / hour")
    if title:
        axs.set_title(title + ': SST$_\mathrm{0.2m}$ @ 10:30')
    fig.tight_layout()
    fig.savefig(prefix+'matchup_tdiff.png')
    plt.close(fig)
    fig, axs = plt.subplots()
    axs.hist(ds.dtime, 48, [-12, 12])
    axs.set_xlim(-12, 12)
    axs.set_xlabel("Time difference / hour")
    if title:
        axs.set_title(title + ': SST$_\mathrm{0.2m}$ @ 10:30')
    fig.tight_layout()
    fig.savefig(prefix+'matchup_tdiff12.png')
    plt.close(fig)
    fig = plt.figure()
    axs = plt.axes(projection=ccrs.PlateCarree())
    axs.coastlines()
    axs.set_global()
    axs.plot(ds.ins_longitude, ds.ins_latitude, ',')
    if title:
        axs.set_title(title)
    fig.tight_layout()
    fig.savefig(prefix+'matchup_position.png')
    plt.close(fig)
    fig, axs = plt.subplots()
    axs.hist(ds.ins_time, 365*2)
    axs.set_xlabel("Date")
    if title:
        axs.set_title(title)
    fig.savefig(prefix+'matchup_time.png')
    fig.tight_layout()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help="MMD file[s] to process")
    parser.add_argument('--sat', default=None, help="Sensor name (for plot titles)")
    parser.add_argument('--dir', default='', help="Output directory")
    parser.add_argument('--quality', default='4,5', help="quality_level to validate (L3 only)")
    parser.add_argument('--filequal', type=int, default=3, help='Minimum file quality level')
    parser.add_argument('--prefix', help="prefix for output files")
    parser.add_argument('--extra', help="suffix for output files (excluding file extension)")
    parser.add_argument('--insitu_type', default='auto', help="In situ type (for uncertainty plots)")
    parser.add_argument('--sirds_type', help='Override sirds type in plot filenames')
    parser.add_argument('--dtime', type=float, help='Delta-time limit (hours)')
    args = parser.parse_args()
    ds = xr.open_mfdataset(args.files, combine='nested', concat_dim='matchup')
    ds.load()
    ptitle = args.sat or ds.product
    prefix = args.prefix or (ds.product + '-' + (args.sirds_type or ds.sirds_type) + '-')
    extra = args.extra or ''
    odir = args.dir
    if 'gbcsver' in ds.attrs:
        odir = odir or ds.attrs['gbcsver']
    if odir:
        os.makedirs(odir, exist_ok=True)
        prefix = os.path.join(odir, prefix)
    saveopts = dict(prefix=prefix, extra=extra)
    if 'OSTIA' in ptitle.upper():
        ret = 'L4'
    elif 'AVHRR' in ptitle.upper():
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

    ds['year'] = ds.sat_time.dt.year + (ds.sat_time.dt.dayofyear-1)/365.
    ds['year'].attrs['long_name'] = 'Year'
    # Correct SST unit values if needed.
    sat_unit_corr = get_unit_offset(ds.sat_sst.units)
    ds.sat_sst.data = ds.sat_sst.data + sat_unit_corr
    ds['tdiff'] = ds.sat_sst - ds.ins_sst
    ds['tdiff'].attrs['long_name'] = 'depth-depth'
    ds['dtime'] = (ds.sat_time - ds.ins_time) / np.timedelta64(1, 'h')
    ds['dtime'].attrs['long_name'] = 'Time difference'
    ds['dtime'].attrs['units'] = 'hour'
    if ds.product == 'OSTIA':
        ds['day'] = ds.year < 0
        ds['day'].attrs['long_name'] = 'Day flag'
        ds['ngt'] = np.invert(ds.day)
    else:
        ds['day'] = ds.sat_l2p_flags.astype(int) & 256 > 0
        ds['day'].attrs['long_name'] = 'Day flag'
        ds['ngt'] = np.invert(ds.day)

    msk = ds.ins_qc1 == 0
    if args.filequal is not None:
        if 'file_quality_level' in ds:
            msk = msk & (ds.file_quality_level >= args.filequal)
        else:
            print('Dataset does not contain file_quality_level - ignoring min file qual request')
    if args.dtime:
        # 0.5 hour time difference is suitable for recent MDs, but will need
        # larger window for historical data
        msk = msk & (np.abs(ds.dtime) < args.dtime)
    if ds.product == 'OSTIA':
        msk = msk & (ds.sat_mask == 1) & np.isfinite(ds.sat_sst)
    else:
        ql = [int(i) for i in args.quality.split(',')]
        msk = msk & ds.sat_quality_level.isin(ql)

    # qc5 = ds.where(msk, drop=True)
    qc5 = ds.isel(matchup=msk)
    set_simple('')

    basic_plots(ds, prefix, title=ptitle)

    print("Histogram plots")
    if ret == 'L4':
        fig = pvir_plot_hist(qc5, ret, 'tdiff', title=ptitle)
        fig.savefig('{prefix}histogram{extra}.png'.format(**saveopts))
        plt.close(fig)
    else:
        fig = pvir_plot_hist(qc5.where(qc5.ngt), ret+'3', 'tdiff', title=ptitle)
        fig.savefig('{prefix}histogram-night-{0}{extra}.png'.format('d3', **saveopts))
        plt.close(fig)
        fig = pvir_plot_hist(qc5.where(qc5.day), ret+'2', 'tdiff', title=ptitle)
        fig.savefig('{prefix}histogram-day-{0}{extra}.png'.format('d2', **saveopts))
        plt.close(fig)

    print("Dependence plots")
    umax = 0.75
    if ds.sirds_type == 'ship':
        umax = 1.5
    plotopts = dict(yrange=[-.5, .5],
                    erange=[0, umax],
                    ret=ret,
                    title=ptitle)
    yrange = int(qc5.year.min()), int(qc5.year.max()+1)
    if ret == 'L4':
        pfunc = pvir_plot_dependence1
    else:
        pfunc = pvir_plot_dependence
    fig = pfunc(qc5, 'year', xrange=yrange, xstep=0.125, **plotopts)
    fig.savefig('{prefix}dependence-{0}{extra}.png'.format('year', **saveopts))
    plt.close(fig)

    fig = pfunc(qc5, 'year', xrange=yrange, xstep=1/12, **plotopts)
    fig.savefig('{prefix}dependence-{0}{extra}.png'.format('month', **saveopts))
    plt.close(fig)

    fig = pfunc(qc5, 'sat_lat', xrange=[-75, 80], xstep=5.0, **plotopts)
    fig.savefig('{prefix}dependence-{0}{extra}.png'.format('sat_lat', **saveopts))
    plt.close(fig)

    if ret == 'L4':
        fig = pfunc(qc5, 'dtime', xrange=[-12, 12], xstep=1, **plotopts)
        fig.savefig('{prefix}dependence-{0}{extra}.png'.format('tdiff', **saveopts))
        plt.close(fig)

    else:
        fig = pfunc(qc5, 'dtime', xrange=[-2, 2], xstep=0.25, **plotopts)
        fig.savefig('{prefix}dependence-{0}{extra}.png'.format('tdiff', **saveopts))
        plt.close(fig)

        fig = pfunc(qc5, 'sat_wind_speed', xrange=[0, 25], xstep=0.25, **plotopts)
        fig.savefig('{prefix}dependence-{0}{extra}.png'.format('wind', **saveopts))
        plt.close(fig)

    print("Spatial plots")
    spatial = pvir_plot_spatial(qc5, 'tdiff', title=ptitle)
    for name, fig in spatial.items():
        try:
            fig.savefig('{prefix}{0}{extra}.png'.format(name, **saveopts))
            plt.close(fig)
        except AttributeError:
            # Spatial plots fail if there is no data.
            plt.close(fig)

    print("Uncertainty plots")
    if ret == 'L4':
        urange = [0, 3]
    else:
        urange = [0, 1]
    uncer = pvir_plot_uncert(qc5, 'sat_unc', 'tdiff', ins=args.insitu_type,
                             min_data=100, title=ptitle, xrange=urange)
    for name, fig in uncer.items():
        try:
            fig.savefig('{prefix}{0}{extra}.png'.format(name, **saveopts))
            plt.close(fig)
        except AttributeError:
            plt.close(fig)

    # Color bar
    fig, axs = plt.subplots(figsize=(6.4, 1.6))
    scl = mpl.cm.ScalarMappable(mpl.colors.Normalize(-1, 1), 'coolwarm')
    fig.colorbar(scl, cax=axs, orientation='horizontal',
                 label='Satellite - in situ / K')
    fig.tight_layout()
    fig.savefig(os.path.join(odir, 'colourbar.png'))
    plt.close(fig)
