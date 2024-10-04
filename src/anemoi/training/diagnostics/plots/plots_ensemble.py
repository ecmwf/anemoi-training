import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from anemoi.training.diagnostics.maps import Coastlines
from anemoi.training.diagnostics.maps import EquirectangularProjection
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from scipy.interpolate import griddata

from anemoi.training.diagnostics.plots.plots import _hide_axes_ticks
from anemoi.training.diagnostics.plots.plots import single_plot
from anemoi.training.diagnostics.plots.plots import compute_spectra
LOGGER = logging.getLogger(__name__)

continents = Coastlines()


def plot_rank_histograms(
    parameters: dict[int, str],
    rh: np.ndarray,
) -> Figure:
    """Plots one rank histogram per target variable.

    Parameters
    ----------
    parameters : Dict[int, str]
        Dictionary of target variables
    rh : np.ndarray
        Rank histogram data of shape (nens, nvar)

    Returns
    -------
    Figure
        The figure object handle.
    """
    fig, ax = plt.subplots(1, len(parameters), figsize=(len(parameters) * 4.5, 4))
    n_ens = rh.shape[0] - 1
    rh = rh.astype(float)

    # Ensure ax is iterable
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    for plot_idx, (_variable_idx, variable_name) in enumerate(parameters.items()):
        rh_ = rh[:, plot_idx]
        ax[plot_idx].bar(np.arange(0, n_ens + 1), rh_ / rh_.sum(), linewidth=1, color="blue", width=0.7)
        ax[plot_idx].hlines(rh_.mean() / rh_.sum(), xmin=-0.5, xmax=n_ens + 0.5, linestyles="--", colors="red")
        ax[plot_idx].set_title(f"{variable_name[0]} ranks")
        _hide_axes_ticks(ax[plot_idx])

    fig.tight_layout()
    return fig


def plot_predicted_ensemble(
    parameters: dict[int, str],
    n_plots_per_sample: int,
    latlons: np.ndarray,
    clevels: float,
    cmap_precip: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scatter: Optional[bool] = False,
    initial_condition: Optional[bool] = False,
) -> Figure:
    """Plots data for one ensemble member.

    Args:
        parameters : Dict[int, str]
            Dictionary of target variables
        n_plots_per_sample : int
            Number of plots per sample
        latlons : np.ndarray
            Latitudes and longitudes
        clevels : float
            Accumulation levels used for precipitation related plots
        cmap_precip: str
            Colours used for each precipitation accumulation level
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        scatter : bool, optional
            Scatter plot, by default False
        initial_condition : bool, optional
            Plotting initial condition, by default False

    Returns
    -------
        fig:
            The figure object handle.
    """
    n_plots_per_sample = 1 if initial_condition else 4

    nens = y_pred.shape[0] if len(y_pred.shape) == 3 else 1

    n_plots_x, n_plots_y = len(parameters), nens + n_plots_per_sample
    LOGGER.debug("n_plots_x = %d, n_plots_y = %d", n_plots_x, n_plots_y)

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize)

    lat, lon = latlons[:, 0], latlons[:, 1]
    projection = EquirectangularProjection()

    pc_lon, pc_lat = projection(lon, lat)

    for plot_idx, (variable_idx, variable_name) in enumerate(parameters.items()):
        yp = y_pred[..., variable_idx].squeeze()
        if initial_condition:
            ax_ = ax[plot_idx, :] if n_plots_x > 1 else ax
            plot_ensemble_sample(
                fig,
                ax_,
                pc_lon,
                pc_lat,
                yp,
                yp,
                variable_name,
                clevels,
                cmap_precip,
                scatter=scatter,
                initial_condition=True,
            )
        else:
            yt = y_true[..., variable_idx].squeeze()
            ax_ = ax[plot_idx, :] if n_plots_x > 1 else ax
            plot_ensemble_sample(fig, ax_, pc_lon, pc_lat, yt, yp, variable_name, clevels, cmap_precip, scatter=scatter)

    return fig


def plot_ensemble_sample(
    fig,
    ax,
    pc_lon: np.ndarray,
    pc_lat: np.ndarray,
    truth: np.ndarray,
    ens_arr: np.ndarray,
    vname: np.ndarray,
    clevels: float,
    cmap_precip: str,
    ens_dim: int = 0,
    scatter: Optional[bool] = False,
    initial_condition: Optional[bool] = False,
) -> None:
    """Use this when plotting ensembles.

    Each member is defined on "flat" (reduced Gaussian) grids.

    Parameters
    ----------
    fig: figure
        Figure object handle
    ax: matplotlib.axes
        Axis object handle
    pc_lon : np.ndarray
        Projected Longitude coordinates array
    pc_lat : np.ndarray
        Projected Latitude coordinates array
    truth : np.ndarray
        True values
    ens_arr : np.ndarray
        Ensemble array
    vname : np.ndarray
        Variable name
    clevels : float
        Accumulation levels used for precipitation related plots
    cmap_precip: str
        Colours used for each precipitation accumulation level
    ens_dim : int, optional
        Ensemble dimension, by default
    scatter : bool, optional
        Scatter plot, by default False
    initial_condition : bool, optional
        Plotting initial condition, by default False

    Returns
    -------
        None
    """
    if vname[0] == "tp" or vname[0] == "cp":
        nws_precip_colors = cmap_precip
        cmap_plt = ListedColormap(nws_precip_colors)

        # Defining the actual precipitation accumulation levels in mm
        cummulation_lvls = clevels
        norm = BoundaryNorm(cummulation_lvls, len(cummulation_lvls) + 1)

        # converting to mm from m
        truth = truth * 1000.0
        ens_arr = ens_arr * 1000.0

    else:
        cmap_plt = "viridis"
        norm = None

    if len(ens_arr.shape) == 2:
        nens = ens_arr.shape[ens_dim]
        ens_mean, ens_sd = ens_arr.mean(axis=ens_dim), ens_arr.std(axis=ens_dim)
    else:
        nens = 1
        ens_mean = ens_arr
        ens_sd = np.zeros(ens_arr.shape)

    if initial_condition:
        plot_index = 1

        # ensemble initial condition mean
        single_plot(
            fig,
            ax[0],
            pc_lon,
            pc_lat,
            ens_mean.squeeze(),
            norm=norm,
            cmap=cmap_plt,
            title=f"{vname[0]}_mean",
            scatter=scatter,
        )
    else:
        plot_index = 4

        # ensemble mean
        single_plot(fig, ax[0], pc_lon, pc_lat, truth, cmap=cmap_plt, norm=norm, title=f"{vname[0]} target", scatter=scatter)
        # ensemble mean
        single_plot(fig, ax[1], pc_lon, pc_lat, ens_mean, cmap=cmap_plt, norm=norm, title=f"{vname[0]} pred mean", scatter=scatter)
        # ensemble spread
        single_plot(
            fig,
            ax[2],
            pc_lon,
            pc_lat,
            ens_mean - truth,
            cmap="bwr",
            norm=TwoSlopeNorm(vcenter=0.0),
            title=f"{vname[0]} ens mean err",
            scatter=scatter,
        )
        # ensemble mean error
        single_plot(fig, ax[3], pc_lon, pc_lat, ens_sd, title=f"{vname[0]} ens sd", scatter=scatter)

    # ensemble members (difference from mean)
    for i_ens in range(nens):
        single_plot(
            fig,
            ax[i_ens + plot_index],
            pc_lon,
            pc_lat,
            np.take(ens_arr, i_ens, axis=ens_dim) - ens_mean,
            cmap="bwr",
            norm=TwoSlopeNorm(vcenter=0.0),
            title=f"{vname[0]}_{i_ens + 1} - mean",
            scatter=scatter,
        )


def plot_spread_skill(
    parameters: dict[int, str],
    ss_metric: tuple[np.ndarray, np.ndarray],
    time_step: int,
    rollout_idxs: list[int],
) -> Figure:
    """Plot the spread-skill metric.

    Parameters
    ----------
    parameters : Dict[int, str]
        Dictionary of target variables
    ss_metric : tuple[np.ndarray, np.ndarray]
        Spread-skill metric data
    time_step : int
        Time step

    Returns
    -------
    Figure
        Figure object handle
    """
    nplots = len(parameters)
    figsize = (nplots * 5, 4)
    fig, ax = plt.subplots(1, nplots, figsize=figsize)

    assert isinstance(ss_metric, tuple), f"Expected a tuple and got {type(ss_metric)}!"
    assert len(ss_metric) == 2, f"Expected a 2-tuple and got a {len(ss_metric)}-tuple!"
    assert (
        ss_metric[0].shape[1] == nplots
    ), f"Shape mismatch in the RMSE metric: expected (..., {nplots}) and got {ss_metric[0].shape}!"
    assert (
        ss_metric[0].shape == ss_metric[1].shape
    ), f"RMSE and spread metric shapes do not match! {ss_metric[0].shape} and {ss_metric[1].shape}"
    assert ss_metric[0].shape[0] == len(rollout_idxs), f"Rollout indices do not match the shape of the RMSE metric: {ss_metric[0].shape[0]} and {len(rollout_idxs)}"

    rmse, spread = ss_metric[0], ss_metric[1]

    x = np.arange(1, rollout_idxs[-1] + 1) * time_step

    for i, (_, pname) in enumerate(parameters.items()):
        ax_ = ax[i] if nplots > 1 else ax
        # Plot the full x-axis with no markers (optional)
        ax_.plot(x, np.zeros_like(x), color="lightgray", visible=False)

        ax_.plot(rollout_idxs, rmse[:, i], "-o", color="red", label="mean RMSE")
        ax_.plot(rollout_idxs, spread[:, i], "-o", color="blue", label="spread")

        ax_.legend()
        ax_.set_title(f"{pname[0]} spread-skill")
        ax_.set_xticks(x)
        ax_.set_xlabel("Lead time [hrs]")

    fig.tight_layout()
    return fig


def plot_spread_skill_bins(
    parameters: dict[int, str],
    ss_metric: tuple[np.ndarray, np.ndarray],
    time_step: int,
) -> Figure:
    """Plot the spread-skill metric in bins.

    Parameters
    ----------
    parameters : Dict[int, str]
        Dictionary of target variables
    ss_metric : tuple[np.ndarray, np.ndarray]
        Spread-skill metric data
    time_step : int
        Time step

    Returns
    -------
    Figure
        Figure object handle
    """
    nplots = len(parameters)
    figsize = (nplots * 5, 4)
    fig, ax = plt.subplots(1, nplots, figsize=figsize)

    assert isinstance(ss_metric, tuple), f"Expected a tuple and got {type(ss_metric)}!"
    assert len(ss_metric) == 2, f"Expected a 2-tuple and got a {len(ss_metric)}-tuple!"
    assert (
        ss_metric[0].shape[1] == nplots
    ), f"Shape mismatch in the RMSE metric: expected (..., {nplots}) and got {ss_metric[0].shape}!"
    assert (
        ss_metric[0].shape == ss_metric[1].shape
    ), f"RMSE and spread metric shapes do not match! {ss_metric[0].shape} and {ss_metric[1].shape}"

    bins_rmse, bins_spread = ss_metric[0], ss_metric[1]
    rollout = bins_rmse.shape[0]

    for i, (_, pname) in enumerate(parameters.items()):
        ax_ = ax[i] if nplots > 1 else ax

        if bins_spread.min() != 0:
            for j in range(rollout):
                ax_.plot(bins_spread[j, i, :], bins_rmse[j, i, :], "-", label=str((j + 1) * time_step) + " hr")
                ax_.plot(bins_rmse[j, i, :], bins_rmse[j, i, :], "--", color="black", label="__nolabel__")
            bins_max = max(np.nanmax(bins_rmse[:, i, :]), np.nanmax(bins_spread[:, i, :]))
        else:
            bins_max = 1
        ax_.set_xlim([0, bins_max])
        ax_.set_ylim([0, bins_max])
        ax_.legend()
        ax_.set_title(f"{pname[0]} spread-skill binned")
        ax_.set_xlabel("Spread")
        ax_.set_ylabel("Skill")

    fig.tight_layout()
    return fig


def plot_power_spectrum(
    parameters: dict[str, int],
    latlons: np.ndarray,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Figure:
    """Plots power spectrum.

    NB: this can be very slow for large data arrays
    call it as infrequently as possible!

    Parameters
    ----------
    parameters : dict[str, int]
        Dictionary of variable names and indices
    latlons : np.ndarray
        lat/lon coordinates array, shape (lat*lon, 2)
    x : np.ndarray
        Input data of shape (lat*lon, nvar*level)
    y_true : np.ndarray
        Expected data of shape (lat*lon, nvar*level)
    y_pred : np.ndarray
        Predicted data of shape (ensemble_size, lat*lon, nvar*level)

    Returns
    -------
    Figure
        The figure object handle.

    """
    n_plots_x, n_plots_y = len(parameters), 1

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize)

    pc = EquirectangularProjection()
    lat, lon = latlons[:, 0], latlons[:, 1]
    pc_lon, pc_lat = pc(lon, lat)
    pc_lon = np.array(pc_lon)
    pc_lat = np.array(pc_lat)
    # Calculate delta_lon and delta_lat on the projected grid
    delta_lon = abs(np.diff(pc_lon))
    non_zero_delta_lon = delta_lon[delta_lon != 0]
    delta_lat = abs(np.diff(pc_lat))
    non_zero_delta_lat = delta_lat[delta_lat != 0]

    # Define a regular grid for interpolation
    n_pix_lon = int(np.floor(abs(pc_lon.max() - pc_lon.min()) / abs(np.min(non_zero_delta_lon))))  # around 400 for O96
    n_pix_lat = int(np.floor(abs(pc_lat.max() - pc_lat.min()) / abs(np.min(non_zero_delta_lat))))  # around 192 for O96
    regular_pc_lon = np.linspace(pc_lon.min(), pc_lon.max(), n_pix_lon)
    regular_pc_lat = np.linspace(pc_lat.min(), pc_lat.max(), n_pix_lat)
    grid_pc_lon, grid_pc_lat = np.meshgrid(regular_pc_lon, regular_pc_lat)

    for plot_idx, (variable_idx, (variable_name, output_only)) in enumerate(parameters.items()):
        yt = y_true[..., variable_idx].squeeze()
        yp = y_pred[..., variable_idx].squeeze()

        # check for any nan in yt
        nan_flag = np.isnan(yt).any()

        method = "linear" if nan_flag else "cubic"
        if output_only:
            xt = x[..., variable_idx].squeeze()
            yt_i = griddata((pc_lon, pc_lat), (yt - xt), (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)

            # Handle ensemble predictions
            yp_i_ensemble = np.array([griddata((pc_lon, pc_lat), (yp_member - xt), (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)
                                      for yp_member in yp])
        else:
            yt_i = griddata((pc_lon, pc_lat), yt, (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)

            # Handle ensemble predictions
            yp_i_ensemble = np.array([griddata((pc_lon, pc_lat), yp_member, (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)
                                      for yp_member in yp])

        # Masking NaN values
        if nan_flag:
            mask = np.isnan(yt_i)
            if mask.any():
                yt_i = np.where(mask, 0.0, yt_i)
                yp_i_ensemble = np.where(mask, 0.0, yp_i_ensemble)

        amplitude_t = np.array(compute_spectra(yt_i))

        # Compute spectra for each ensemble member
        amplitude_p_ensemble = np.array([compute_spectra(yp_i) for yp_i in yp_i_ensemble])

        ax_ = ax[plot_idx] if len(parameters) > 1 else ax
        ax_.loglog(
            np.arange(1, amplitude_t.shape[0]),
            amplitude_t[1:],
            label="Truth (data)",
            color="black",
            linewidth=2,
        )

        # Plot each ensemble member
        for i, amplitude_p in enumerate(amplitude_p_ensemble):
            ax_.loglog(
                np.arange(1, amplitude_p.shape[0]),
                amplitude_p[1:],
                label=f"Ensemble member {i + 1}" if i == 0 else "_nolegend_",
                alpha=0.5,
            )

        ax_.legend()
        ax_.set_title(variable_name)

        ax_.set_xlabel("$k$")
        ax_.set_ylabel("$P(k)$")
        ax_.set_aspect("auto", adjustable=None)
    fig.tight_layout()
    return fig
