import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm

def bin_data(df, variable, n_bins=15):
    bins = np.linspace(df[variable].min(), df[variable].max(), n_bins)
    df[variable + '_bin'] = pd.cut(df[variable], bins=bins, labels=bins[:-1])
    binned_data = df.groupby(variable + '_bin').agg({'OmB': ['mean', 'std']})
    binned_data.columns = ['OmB_mean', 'OmB_std']
    binned_data = binned_data.reset_index()
    return binned_data

def flatten_array(A):
    tmp_out = A.flatten()
    return tmp_out[~np.isnan(tmp_out)]


def geo_map_vis(data_list, labels, cmaps,
                lon, lat,
                date, unit,
                vmin, vmax, shrink,
                log=False,
                save=False,
                font=14,
                out_path='.', name='figure',
                axes=None, fig=None):
    """
    Plot 1–4 panels of geographic data on Basemap, with a shared colorbar
    that honors your vmin/vmax and colormap even when you pass in axes.
    """
    n = len(data_list)
    if not (1 <= n <= 4):
        raise ValueError("data_list must have between 1 and 4 elements")

    # 1) decide whether we created the figure & axes or were given them
    if axes is None:
        # standalone: build own fig & axes
        if n == 1:
            nrows, ncols, figsize = 1, 1, (6, 5)
        elif n == 2:
            nrows, ncols, figsize = 1, 2, (12, 5)
        elif n == 3:
            nrows, ncols, figsize = 1, 3, (15, 12)
        else:
            nrows, ncols, figsize = 1, 4, (20, 12)

        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        fig.set_facecolor('white')
        plt.rcParams.update({'font.size': font})
        fig.subplots_adjust(wspace=0.01, hspace=0.0)
        # flatten
        axs_flat = axs.flatten().tolist() if n>1 else [axs]
        standalone = True
    else:
        # using user-supplied axes
        axs_flat = axes
        fig = fig or plt.gcf()
        standalone = False

    # 2) normalization & tick locations
    if log:
        vmin = max(vmin, 1e-10)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        ticks = np.logspace(np.log10(vmin), np.log10(vmax), 5)
    else:
        norm = None
        step = 0.25 * (vmax - vmin)
        ticks = np.arange(vmin, vmax + step, step)

    # map bounds
    lon0, lat0 = 13.8, 47.4
    llon, ulon = lon0 - 8.8, lon0 + 10.0
    llat, ulat = lat0 - 4.4, lat0 + 3.4

    # 3) plot each panel
    for idx, (data, title, cmap) in enumerate(zip(data_list, labels, cmaps)):
        ax = axs_flat[idx]
        m = Basemap(projection='lcc',
                    lon_0=lon0, lat_0=lat0,
                    llcrnrlon=llon, llcrnrlat=llat,
                    urcrnrlon=ulon, urcrnrlat=ulat,
                    resolution='h', ax=ax)
        m.drawcountries(); m.drawcoastlines()
        x, y = m(lon, lat)
        p = m.pcolormesh(x, y, data,
                         shading='nearest',
                         cmap=cmap,
                         vmin=vmin, vmax=vmax,
                         norm=norm)
        ax.set_title(f"{title} {date}", fontsize=font)

        parallels = np.arange(llat+1, ulat, 3)
        if idx == 0:
            m.drawparallels(parallels, labels=[1,0,0,0], fontsize=font-2)
        else:
            m.drawparallels(parallels, labels=[0,0,0,0], fontsize=font-2)

        meridians = np.arange(llon+1, ulon, 5)
        m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=font-2)

        if idx > 0:
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False)
        ax.set_facecolor('yellow')
    # 4) ALWAYS add the shared colorbar, even if axes were passed in
    cbar = fig.colorbar(p,
                        ax=axs_flat[:n],
                        orientation='vertical',
                        pad=0.005,
                        shrink=shrink)
    cbar.set_label(unit, fontsize=font)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    cbar.ax.tick_params(labelsize=font)

    # 5) standalone: save/show/close
    if standalone:
        if save:
            fig.savefig(f"{out_path}/{name}.png",
                        dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    return axs_flat