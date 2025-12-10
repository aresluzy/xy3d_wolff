from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict, Any

import numpy as np

from .core import (
    simulate_all_data,
    improved_simulate_all_data,
    plot_simulation_results,
    data_collapse_specific_heat,
    data_collapse_susceptibility,
    data_collapse_magnetization,
)


@dataclass
class SimulationConfig:
    J: float
    n_steps: int
    n_equil: int
    L_list: Sequence[int]
    T_list: Sequence[float]


class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config

    @property
    def J(self) -> float:
        return self.config.J

    @property
    def n_steps(self) -> int:
        return self.config.n_steps

    @property
    def n_equil(self) -> int:
        return self.config.n_equil

    @property
    def L_list(self) -> Sequence[int]:
        return self.config.L_list

    @property
    def T_list(self) -> np.ndarray:
        return np.array(self.config.T_list, dtype=float)

    def run_basic(self) -> Dict[int, Dict[float, Any]]:
        return simulate_all_data(
            self.L_list,
            self.T_list,
            self.J,
            self.n_steps,
            self.n_equil,
        )

    def run_with_estimator(self) -> Dict[int, Dict[float, Any]]:
        return improved_simulate_all_data(
            self.L_list,
            self.T_list,
            self.J,
            self.n_steps,
            self.n_equil,
        )

    def plot_basic(self, simulation_results: Dict[int, Dict[float, Any]]) -> None:
        plot_simulation_results(simulation_results, self.L_list, self.T_list)

    def collapse_specific_heat(
        self,
        simulation_results: Dict[int, Dict[float, Any]],
        Tc: float,
        alpha: float,
        nu: float,
        plot: bool = True,
    ):
        return data_collapse_specific_heat(
            simulation_results,
            self.L_list,
            self.T_list,
            Tc,
            alpha,
            nu,
            plot=plot,
        )

    def collapse_susceptibility(
        self,
        simulation_results: Dict[int, Dict[float, Any]],
        Tc: float,
        gamma: float,
        nu: float,
        plot: bool = True,
    ):
        return data_collapse_susceptibility(
            simulation_results,
            self.L_list,
            self.T_list,
            Tc,
            gamma,
            nu,
            plot=plot,
        )

    def collapse_magnetization(
        self,
        simulation_results: Dict[int, Dict[float, Any]],
        Tc: float,
        beta: float,
        nu: float,
        plot: bool = True,
    ):
        return data_collapse_magnetization(
            simulation_results,
            self.L_list,
            self.T_list,
            Tc,
            beta,
            nu,
            plot=plot,
        )
