import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm


def flatten_array(A):
    tmp_out = A.flatten()
    return tmp_out[~np.isnan(tmp_out)]

import numpy as np
import proplot as pplt

def plot_simulations(sims,x_lims, y_lims,
                     columns=None,
                     bins=None,
                     range_=None,
                     colors=None,
                     alpha=0.7,
                     linewidth=1.2,
                     log_x=False,
                     log_y=False,
                     legend_loc='ur',
                     title='Overlaid histograms',
                     xlabel='distribution',
                     ylabel='count', save=None,name=None):
    """
    Plot overlaid histograms of the specified columns in `sims`,
    with optional log scaling.

    Parameters
    ----------
    sims : pandas.DataFrame
        DataFrame whose columns are the series to histogram.
    columns : list of str, optional
        Which columns of `sims` to plot. Defaults to all.
    bins : array‐like, optional
        Bin edges (default np.arange(-0.01, 1, 0.2)).
    colors : sequence of color specs, optional
        Edge colors for each series. Auto‐chosen from 'tab10' or 'tab20'.
    alpha : float, optional
        Line transparency.
    linewidth : float, optional
        Thickness of histogram edges.
    log_x : bool, optional
        If True, set x-axis to log scale.
    log_y : bool, optional
        If True, set y-axis to log scale.
    legend_loc : str, optional
        Proplot legend location code.
    title, xlabel, ylabel : str, optional
        Suptitle and axes labels.
    """
    # pick columns
    if columns is None:
        columns = list(sims.columns)
    data = sims[columns].values
    n_series = len(columns)

    # defaults
    if colors is None:
        cmap_name = 'tab10' if n_series <= 10 else 'tab20'
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i) for i in range(n_series)]

    # draw
    fig, ax = pplt.subplots(refwidth=4, refaspect=(4, 3))
    plt.rcParams.update({'font.size': 10})
    ax.format(suptitle=title, xlabel=xlabel, ylabel=ylabel)
    ax.hist(
        data,
        bins,
        filled=False,
        alpha=alpha,
        cycle=colors,range= range_,
        linewidth=linewidth,
        labels=columns,
    )
    ax.legend(loc=legend_loc, bbox_to_anchor=(0.5, 1.08),
          ncol=3, fancybox=True, shadow=True, fontsize=9)
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    # apply log scales if requested
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    if save:
        fig.savefig(name+".png", dpi=1200)
    return fig, ax
def cloud_impact_av(cloudiness_threshold, B, O):
    
    Cy = np.where(O - cloudiness_threshold > 0, O - cloudiness_threshold, 0)
    Cx = np.where(B - cloudiness_threshold > 0, B - cloudiness_threshold, 0)
    Ca = (Cy + Cx) / 2
    return Ca, Cx, Cy
def retrieve_stds_from_binned_Ca(X:np.array, df:pd.DataFrame, namelist:list):
    """
    Assuming you have a DataFrame called df containing the bins and corresponding std values
    df = pd.DataFrame({'bin': [(1, 2), (2, 3), ..., (39, 40)], 'std_value': [std_val_1, std_val_2, ..., std_val_39]})
    X 1D array of Ca values
    Y 1D array of stds that correspond to Ca value found in the df bins of Ca
    """
    
    # Initialize Y with the same length as X
    Y = [0] * len(X)

    # Iterate through the elements of X
    for i in range(len(X)):
        x = X[i]
        # Find the bin where X[i] lies
        bin_index = None
        for j in range(len(df)):
            if df[namelist[0]][j].left < x <= df[namelist[0]][j].right: # namelist[0] = predictor name
                bin_index = j
                break
        if bin_index is not None:
            # Save the corresponding std value
            Y[i] = df[namelist[1]][bin_index] # namelist[1] = FG depratures
        else:
            # If x does not fall into any bin, you can handle it accordingly
            Y[i] = None  # or any default value you prefer
    std_array = np.array(Y, dtype=np.float64)
    return std_array
def retrieve_stds_from_binned_Ca(X: np.array, df: pd.DataFrame, namelist: list, zero_Ca:np.nan):
    """
    Efficiently maps Ca values in X to corresponding standard deviations in df using pd.IntervalIndex.
    
    Parameters:
    X : np.array : 1D array of Ca values
    df : pd.DataFrame : DataFrame containing bin intervals and corresponding standard deviation values
    namelist : list : List containing two column names, 
                      namelist[0] = column with bins
                      namelist[1] = column with standard deviation values

    Returns:
    np.array : Array of standard deviation values corresponding to X
    """
    
    # Create an IntervalIndex for fast lookups
    interval_index = pd.IntervalIndex(df[namelist[0]])

    # Use pandas searchsorted-like approach to find the bin index for each X value
    bin_indices = interval_index.get_indexer(X)

    # Use bin_indices to map to the std values, setting out-of-bin values to NaN
    Y = np.where(bin_indices >= 0, df[namelist[1]].values[bin_indices], zero_Ca)

    return np.array(Y, dtype=np.float64)

def plot_2d_hist_grid(data, row_vars=None, col_vars=None, bins=50,ymin=0,ymax=1.2, cmap='viridis', norm=None, fontsize=18,figsize=(8, 6), save=None, out_path="/home/km4c/to_ucloud/", name=None):
    """
    Plots a grid of 2D histograms.

    Parameters:
        data (dict or DataFrame): Dictionary or DataFrame with keys/columns being variable names and values/columns being arrays.
        row_vars (list): Variables to use as rows.
        col_vars (list): Variables to use as columns.
        bins (int or [int, int]): Number of bins for the 2D histogram.
        cmap (str or Colormap): Matplotlib colormap.
        norm (Normalize): Optional normalization (e.g., LogNorm).
        figsize (tuple): Figure size.
    """
    import matplotlib.pyplot as plt
    row_vars = row_vars or []
    col_vars = col_vars or []

    nrows = len(row_vars)
    ncols = len(col_vars)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    plt.rcParams.update({'font.size': fontsize})

    for i, yvar in enumerate(row_vars):
        for j, xvar in enumerate(col_vars):
            ax = axes[i][j]
            x = np.asarray(data[xvar])
            y = np.asarray(data[yvar])

            # Clean NaNs
            mask = ~np.isnan(x) & ~np.isnan(y)
            hist, xedges, yedges = np.histogram2d(x[mask], y[mask], bins=bins)

            im = ax.imshow(
                hist.T,
                origin='lower',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap=cmap,
                norm=norm,
                aspect='auto'
            )
            ax.set_xlabel(xvar)
            ax.set_ylabel(yvar)
            if j !=0:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)
            ax.set_ylim(ymin,ymax)
            ax.set_xlim(ymin,ymax)
            #ax.tick_params(axis='both', labelsize=fontsize)
            #ax.legend(fontsize=fontsize)
    fig.tight_layout()
    #plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, 
             location='right', shrink=0.75)
    if save == True:
        fig.savefig(f"{out_path}/{name}.png", dpi=400)
    plt.show()
    return fig