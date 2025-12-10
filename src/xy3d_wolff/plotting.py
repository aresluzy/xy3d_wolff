from __future__ import annotations

from typing import Dict, Any, Sequence

from .core import (
    plot_spin_orientations,
    plot_simulation_results,
    data_collapse_specific_heat,
    data_collapse_susceptibility,
    data_collapse_magnetization,
)


def plot_spins(spins) -> None:
    plot_spin_orientations(spins)


def plot_results(
    simulation_results: Dict[int, Dict[float, Any]],
    L_list: Sequence[int],
    T_list,
) -> None:
    plot_simulation_results(simulation_results, L_list, T_list)


def collapse_all(
    simulation_results,
    L_list,
    T_list,
    Tc: float,
    alpha: float,
    beta: float,
    gamma: float,
    nu: float,
):
    c = data_collapse_specific_heat(simulation_results, L_list, T_list, Tc, alpha, nu)
    s = data_collapse_susceptibility(
        simulation_results, L_list, T_list, Tc, gamma, nu
    )
    m = data_collapse_magnetization(
        simulation_results, L_list, T_list, Tc, beta, nu
    )
    return c, s, m
