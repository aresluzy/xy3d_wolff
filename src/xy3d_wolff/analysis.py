from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence

import numpy as np

from . import core


@dataclass
class CriticalFitResults:
    """
    Container for critical exponent fits.
    """
    params: Dict[str, float]
    errors: Dict[str, float]


class XYAnalysis:
    """
    Collection of analysis and fitting methods wrapping core functions.
    """

    @staticmethod
    def autocorrelation(series, max_lag: int | None = None):
        """
        Compute autocorrelation and estimate autocorrelation time.

        Parameters
        ----------
        series : array_like
            Time series of an observable.
        max_lag : int or None
            Maximum lag; if None, use length of series - 1.

        Returns
        -------
        tuple
            (autocorr, tau_int) from core.compute_autocorrelation and
            estimate_autocorrelation_time.
        """
        series = np.asarray(series)
        if max_lag is None:
            max_lag = len(series) - 1
        ac = core.compute_autocorrelation(series, max_lag)
        tau_int = core.estimate_autocorrelation_time(ac)
        return ac, tau_int

    @staticmethod
    def correlation_length_from_Sq(Sq, q_vectors, L: int) -> float:
        """
        Wrapper around compute_correlation_length_from_Sq.

        Parameters
        ----------
        Sq : ndarray
        q_vectors : ndarray
        L : int

        Returns
        -------
        float
            Correlation length.
        """
        return core.compute_correlation_length_from_Sq(Sq, q_vectors, L)

    @staticmethod
    def correlation_length_from_clusters(cluster_sizes, L: int) -> float:
        """
        Wrapper around compute_correlation_length_from_cluster_sizes.
        """
        return core.compute_correlation_length_from_cluster_sizes(
            cluster_sizes, L
        )

    @staticmethod
    def structure_factor_from_spins(spins):
        """
        Compute spin correlation function and structure factor S(q).
        """
        G_r = core.compute_spin_correlation(spins)
        S_q = core.compute_structure_factor(G_r)
        return G_r, S_q

    @staticmethod
    def fit_correlation_length(simulation_results, L_list, T_list, Tc_guess):
        """
        Call fit_correlation_length from core.
        """
        return core.fit_correlation_length(
            simulation_results, L_list, T_list, Tc_guess
        )

    @staticmethod
    def fit_susceptibility(simulation_results, L_list, T_list, Tc_guess):
        """
        Call fit_susceptibility from core.
        """
        return core.fit_susceptibility(
            simulation_results, L_list, T_list, Tc_guess
        )

    @staticmethod
    def fit_specific_heat_per_L(simulation_results, L_list, T_list, Tc_guess):
        """
        Call fit_specific_heat_per_L from core.
        """
        return core.fit_specific_heat_per_L(
            simulation_results, L_list, T_list, Tc_guess
        )

    @staticmethod
    def fit_binder_crossings(simulation_results, L_list, T_list):
        """
        Call fit_binder_crossings from core.
        """
        return core.fit_binder_crossings(simulation_results, L_list, T_list)

    @staticmethod
    def fit_magnetization_per_L(simulation_results, L_list, T_list, Tc_guess):
        """
        Call fit_magnetization_per_L from core.
        """
        return core.fit_magnetization_per_L(
            simulation_results, L_list, T_list, Tc_guess
        )

    @staticmethod
    def fit_specific_heat(simulation_results, L_list, T_list):
        """
        Call fit_specific_heat from core.
        """
        return core.fit_specific_heat(simulation_results, L_list, T_list)

    @staticmethod
    def fit_magnetization(simulation_results, L_list, T_list):
        """
        Call fit_magnetization from core.
        """
        return core.fit_magnetization(simulation_results, L_list, T_list)

    @staticmethod
    def fit_susceptibility_global(simulation_results, L_list, T_list):
        """
        Call fit_susceptibility (global version) from core.
        """
        return core.fit_susceptibility(simulation_results, L_list, T_list)

    @staticmethod
    def fit_nu_from_binder_cumulant(simulation_results, L_list, T_list):
        """
        Call fit_nu_from_binder_cumulant from core.
        """
        return core.fit_nu_from_binder_cumulant(
            simulation_results, L_list, T_list
        )

    @staticmethod
    def data_collapse_specific_heat(
        simulation_results,
        L_list: Sequence[int],
        T_list,
        Tc: float,
        alpha: float,
        nu: float,
        plot: bool = True,
    ):
        """
        Wrapper for data_collapse_specific_heat.
        """
        return core.data_collapse_specific_heat(
            simulation_results, L_list, T_list, Tc, alpha, nu, plot=plot
        )

    @staticmethod
    def data_collapse_susceptibility(
        simulation_results,
        L_list: Sequence[int],
        T_list,
        Tc: float,
        gamma: float,
        nu: float,
        plot: bool = True,
    ):
        """
        Wrapper for data_collapse_susceptibility.
        """
        return core.data_collapse_susceptibility(
            simulation_results, L_list, T_list, Tc, gamma, nu, plot=plot
        )

    @staticmethod
    def data_collapse_magnetization(
        simulation_results,
        L_list: Sequence[int],
        T_list,
        Tc: float,
        beta: float,
        nu: float,
        plot: bool = True,
    ):
        """
        Wrapper for data_collapse_magnetization.
        """
        return core.data_collapse_magnetization(
            simulation_results, L_list, T_list, Tc, beta, nu, plot=plot
        )
