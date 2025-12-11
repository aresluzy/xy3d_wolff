from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict, Any

import numpy as np

from . import core


@dataclass
class SimulationConfig:
    """
    Configuration for XY Monte Carlo simulations.

    Attributes
    ----------
    J : float
        Coupling constant.
    n_steps : int
        Total Wolff updates per temperature.
    n_equil : int
        Number of equilibration steps to discard.
    L_list : sequence of int
        Set of linear system sizes.
    T_list : sequence of float
        Temperatures to simulate.
    """
    J: float
    n_steps: int
    n_equil: int
    L_list: Sequence[int]
    T_list: Sequence[float]


class XYSimulation:
    """
    Represents a simulation of a single (L, T) pair.
    """

    def __init__(self, L: int, T: float, J: float, n_steps: int, n_equil: int):
        """
        Parameters
        ----------
        L : int
            Linear system size.
        T : float
            Temperature.
        J : float
            Coupling constant.
        n_steps : int
            Number of Wolff steps.
        n_equil : int
            Number of equilibration steps.
        """
        self.L = L
        self.T = T
        self.J = J
        self.n_steps = n_steps
        self.n_equil = n_equil

    def run_basic(self) -> Dict[str, Any]:
        """
        Run the basic simulation using run_simulation from core.

        Returns
        -------
        dict
            Result dictionary with observables and time series.
        """
        return core.run_simulation(
            self.L, self.J, self.T, self.n_steps, self.n_equil
        )

    def run_improved(self) -> Dict[str, Any]:
        """
        Run the improved-estimator simulation using run_improved_simulation.

        Returns
        -------
        dict
            Result dictionary with cluster-based estimators.
        """
        return core.run_improved_simulation(
            self.L, self.J, self.T, self.n_steps, self.n_equil
        )


class XYStudy:
    """
    Study object that runs simulations over sets of (L, T).
    """

    def __init__(self, config: SimulationConfig):
        """
        Parameters
        ----------
        config : SimulationConfig
            Global simulation configuration.
        """
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

    def run_basic_all(self) -> Dict[int, Dict[float, Any]]:
        """
        Run run_simulation over all L and T using simulate_all_data.

        Returns
        -------
        dict
            Nested dictionary results[L][T].
        """
        return core.simulate_all_data(
            self.L_list,
            self.T_list,
            self.J,
            self.n_steps,
            self.n_equil,
        )

    def run_improved_all(self) -> Dict[int, Dict[float, Any]]:
        """
        Run improved simulations over all L and T using improved_simulate_all_data.

        Returns
        -------
        dict
            Nested dictionary results[L][T].
        """
        return core.improved_simulate_all_data(
            self.L_list,
            self.T_list,
            self.J,
            self.n_steps,
            self.n_equil,
        )

    def run_single_basic(self, L: int, T: float) -> Dict[str, Any]:
        """
        Run a single (L, T) basic simulation.

        Returns
        -------
        dict
            Result from run_simulation.
        """
        sim = XYSimulation(L, T, self.J, self.n_steps, self.n_equil)
        return sim.run_basic()

    def run_single_improved(self, L: int, T: float) -> Dict[str, Any]:
        """
        Run a single (L, T) improved simulation.

        Returns
        -------
        dict
            Result from run_improved_simulation.
        """
        sim = XYSimulation(L, T, self.J, self.n_steps, self.n_equil)
        return sim.run_improved()
