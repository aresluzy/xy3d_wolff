from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
from src.xy3d_wolff import core


@dataclass
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
        self.spins = self.initialize_lattice(L)

    def initialize_lattice(self, L: int) -> np.ndarray:
        """
        Create a random XY spin configuration.

        Each spin is (cosθ, sinθ) with θ ∈ [0, 2π).

        Parameters
        ----------
        L : int
        Linear system size.

        Returns
        -------
        ndarray
        Random spins with shape (L, L, L, 2)
        """
        theta = np.random.uniform(0, 2 * np.pi, (L, L, L))
        spins = np.stack((np.cos(theta), np.sin(theta)), axis=-1)  # Shape: (L, L, L, 2)
        return spins


class WolffClusterUpdater:
    """
    Wolff cluster updates for the 3D XY model.

    Parameters
    ----------
    J : float
        Coupling constant.
    T : float
        Temperature.
    """

    def __init__(self, J: float, T: float):
        self.J = J
        self.T = T

    def wolff_update(self, spins: np.ndarray) -> int:
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
        beta = 1.0 / self.T
        L = spins.shape[0]

        # Choose a random reflection axis
        phi = np.random.uniform(0, 2 * np.pi)
        r = np.array([np.cos(phi), np.sin(phi)])  # Unit vector in x-y plane

        # Choose a random seed spin
        i0, j0, k0 = np.random.randint(0, L, 3)
        S_i0 = spins[i0, j0, k0]

        # Reflect the seed spin
        S_i0_new = S_i0 - 2 * np.dot(S_i0, r) * r
        spins[i0, j0, k0] = S_i0_new

        # Initialize cluster
        cluster = set()
        cluster.add((i0, j0, k0))

        # Use a stack for depth-first search
        stack = deque()
        stack.append((i0, j0, k0))

        # Keep track of flipped spins
        flipped = np.zeros(spins.shape[:3], dtype=bool)
        flipped[i0, j0, k0] = True

        while stack:
            i, j, k = stack.pop()
            S_i = spins[i, j, k]

            # Neighbor indices with periodic boundary conditions
            neighbors = [
                ((i + 1) % L, j, k),
                ((i - 1) % L, j, k),
                (i, (j + 1) % L, k),
                (i, (j - 1) % L, k),
                (i, j, (k + 1) % L),
                (i, j, (k - 1) % L)
            ]

            for ni, nj, nk in neighbors:
                if not flipped[ni, nj, nk]:
                    S_j = spins[ni, nj, nk]
                    delta = 2 * beta * self.J * np.dot(S_i, r) * np.dot(S_j, r)
                    p_add = 1 - np.exp(min(0, delta))

                    if np.random.rand() < p_add:
                        # Reflect spin S_j
                        S_j_new = S_j - 2 * np.dot(S_j, r) * r
                        spins[ni, nj, nk] = S_j_new
                        flipped[ni, nj, nk] = True
                        cluster.add((ni, nj, nk))
                        stack.append((ni, nj, nk))

        return len(cluster)

    def wolff_update_new(self, spins: np.ndarray) -> int:
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
        beta = 1.0 / self.T
        L = spins.shape[0]

        # Choose a random reflection axis
        phi = np.random.uniform(0, 2 * np.pi)
        r = np.array([np.cos(phi), np.sin(phi)])  # Unit vector in x-y plane

        # Choose a random seed spin
        i0, j0, k0 = np.random.randint(0, L, 3)
        S_i0 = spins[i0, j0, k0]

        # Reflect the seed spin
        S_i0_new = S_i0 - 2 * np.dot(S_i0, r) * r
        spins[i0, j0, k0] = S_i0_new

        # Initialize cluster
        cluster = set()
        cluster.add((i0, j0, k0))

        # Use a stack for depth-first search
        stack = deque()
        stack.append((i0, j0, k0))

        # Keep track of flipped spins
        flipped = np.zeros(spins.shape[:3], dtype=bool)
        flipped[i0, j0, k0] = True

        while stack:
            i, j, k = stack.pop()
            S_i = spins[i, j, k]

            # Neighbor indices with periodic boundary conditions
            neighbors = [
                ((i + 1) % L, j, k),
                ((i - 1) % L, j, k),
                (i, (j + 1) % L, k),
                (i, (j - 1) % L, k),
                (i, j, (k + 1) % L),
                (i, j, (k - 1) % L)
            ]

            for ni, nj, nk in neighbors:
                if not flipped[ni, nj, nk]:
                    S_j = spins[ni, nj, nk]
                    # Corrected delta with negative sign
                    delta = -2 * beta * self.J * np.dot(S_i, r) * np.dot(S_j, r)
                    p_add = 1 - np.exp(min(0, delta))

                    if np.random.rand() < p_add:
                        # Reflect spin S_j
                        S_j_new = S_j - 2 * np.dot(S_j, r) * r
                        spins[ni, nj, nk] = S_j_new
                        flipped[ni, nj, nk] = True
                        cluster.add((ni, nj, nk))
                        stack.append((ni, nj, nk))

        return len(cluster)

    def wolff_update_with_estimator(self, spins: np.ndarray):
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
        beta = 1.0 / self.T
        L = spins.shape[0]

        # Choose a random reflection axis (unit vector in x-y plane)
        phi = np.random.uniform(0, 2 * np.pi)
        r = np.array([np.cos(phi), np.sin(phi)])  # Reflection axis

        # Choose a random seed spin
        i0, j0, k0 = np.random.randint(0, L, 3)
        S_i0 = spins[i0, j0, k0]

        # Reflect the seed spin
        S_i0_new = S_i0 - 2 * np.dot(S_i0, r) * r
        spins[i0, j0, k0] = S_i0_new

        # Initialize cluster
        cluster = [(i0, j0, k0)]
        flipped = np.zeros((L, L, L), dtype=bool)
        flipped[i0, j0, k0] = True

        # Store positions of spins in the cluster
        cluster_positions = [(i0, j0, k0)]

        # Cluster growth
        while cluster:
            i, j, k = cluster.pop()
            S_i = spins[i, j, k]

            # Neighbor indices with periodic boundary conditions
            neighbors = [
                ((i + 1) % L, j, k),
                ((i - 1) % L, j, k),
                (i, (j + 1) % L, k),
                (i, (j - 1) % L, k),
                (i, j, (k + 1) % L),
                (i, j, (k - 1) % L)
            ]

            for ni, nj, nk in neighbors:
                if not flipped[ni, nj, nk]:
                    S_j = spins[ni, nj, nk]
                    delta = 2 * beta * self.J * np.dot(S_i, r) * np.dot(S_j, r)
                    p_add = 1 - np.exp(min(0, delta))

                    if np.random.rand() < p_add:
                        # Reflect spin S_j
                        S_j_new = S_j - 2 * np.dot(S_j, r) * r
                        spins[ni, nj, nk] = S_j_new
                        flipped[ni, nj, nk] = True
                        cluster.append((ni, nj, nk))
                        cluster_positions.append((ni, nj, nk))
        cluster_size = np.sum(flipped)

        cluster_Sq, q_vectors = core.compute_improved_structure_factor(cluster_positions, L)

        return cluster_size, cluster_Sq, q_vectors


