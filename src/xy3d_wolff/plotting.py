from __future__ import annotations

from typing import Dict, Any, Sequence

from . import core


class XYPlotter:
    """
    Plotting utilities that wrap the plotting functions in core.py.
    """

    @staticmethod
    def plot_spins(spins) -> None:
        """
        Plot spin orientations as a 3D quiver plot.

        Parameters
        ----------
        spins : ndarray
            Spin configuration array of shape (L, L, L, 2).
        """
        core.plot_spin_orientations(spins)

    @staticmethod
    def create_spin_gif(
        spins_sequence,
        filename: str = "spins.gif",
        interval: int = 100,
    ) -> None:
        """
        Create an animated GIF of spin configurations.

        Parameters
        ----------
        spins_sequence : sequence of ndarray
            Sequence of spin configurations.
        filename : str
            Output GIF filename.
        interval : int
            Frame interval in milliseconds.
        """
        core.create_spin_orientation_gif(
            spins_sequence, filename=filename, interval=interval
        )

    @staticmethod
    def plot_observables(
        simulation_results: Dict[int, Dict[float, Any]],
        L_list: Sequence[int],
        T_list,
    ) -> None:
        """
        Plot magnetization, energy, susceptibility, specific heat, and
        Binder cumulant vs temperature.

        This calls plot_simulation_results from core.
        """
        core.plot_simulation_results(simulation_results, L_list, T_list)

    @staticmethod
    def plot_data_collapse_specific_heat(
        simulation_results,
        L_list,
        T_list,
        Tc: float,
        alpha: float,
        nu: float,
    ):
        """
        Produce specific-heat data-collapse plot.
        """
        return core.data_collapse_specific_heat(
            simulation_results, L_list, T_list, Tc, alpha, nu, plot=True
        )

    @staticmethod
    def plot_data_collapse_susceptibility(
        simulation_results,
        L_list,
        T_list,
        Tc: float,
        gamma: float,
        nu: float,
    ):
        """
        Produce susceptibility data-collapse plot.
        """
        return core.data_collapse_susceptibility(
            simulation_results, L_list, T_list, Tc, gamma, nu, plot=True
        )

    @staticmethod
    def plot_data_collapse_magnetization(
        simulation_results,
        L_list,
        T_list,
        Tc: float,
        beta: float,
        nu: float,
    ):
        """
        Produce magnetization data-collapse plot.
        """
        return core.data_collapse_magnetization(
            simulation_results, L_list, T_list, Tc, beta, nu, plot=True
        )

