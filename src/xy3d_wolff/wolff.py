from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from . import core


@dataclass
class WolffParameters:
    """
    Parameters controlling the Wolff cluster updates.

    Attributes
    ----------
    J : float
        Coupling constant.
    T : float
        Temperature.
    """
    J: float
    T: float


class XYLattice:
    """
    3D XY spin lattice with helper methods for initialization and copying.
    """

    def __init__(self, L: int):
        """
        Parameters
        ----------
        L : int
            Linear system size.
        """
        self.L = L
        self.spins = core.initialize_lattice(L)

    def reset_random(self) -> None:
        """
        Reinitialize spins to a new random XY configuration.
        """
        self.spins = core.initialize_lattice(self.L)

    def copy(self) -> "XYLattice":
        """
        Returns
        -------
        XYLattice
            Deep copy of the lattice.
        """
        new_lat = XYLattice(self.L)
        new_lat.spins = np.copy(self.spins)
        return new_lat


class WolffClusterUpdater:
    """
    Object-oriented wrapper for Wolff cluster update functions.
    """

    def __init__(self, params: WolffParameters):
        """
        Parameters
        ----------
        params : WolffParameters
            Wolff algorithm parameters.
        """
        self.params = params

    @property
    def J(self) -> float:
        return self.params.J

    @property
    def T(self) -> float:
        return self.params.T

    def step(self, spins: np.ndarray) -> int:
        """
        Perform one Wolff cluster update using the standard update.

        Parameters
        ----------
        spins : ndarray
            Spin configuration array of shape (L, L, L, 2).

        Returns
        -------
        int
            Size of the cluster that was flipped.
        """
        return core.wolff_update(spins, self.J, self.T)

    def step_new(self, spins: np.ndarray) -> int:
        """
        Perform one Wolff update using wolff_update_new.

        Parameters
        ----------
        spins : ndarray
            Spin configuration array.

        Returns
        -------
        int
            Size of the updated cluster.
        """
        return core.wolff_update_new(spins, self.J, self.T)

    def step_with_estimator(
        self, spins: np.ndarray
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Perform one Wolff update and return improved-estimator data.

        Parameters
        ----------
        spins : ndarray
            Spin configuration array.

        Returns
        -------
        tuple
            (cluster_size, cluster_Sq, q_vectors) as in
            wolff_update_with_estimator.
        """
        return core.wolff_update_with_estimator(spins, self.J, self.T)


