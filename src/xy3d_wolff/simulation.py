from __future__ import annotations
from typing import Sequence, Dict, Any
import numpy as np
from src.xy3d_wolff import core
from src.xy3d_wolff.wolff import XYLattice, WolffClusterUpdater


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

    def run_simulation(self) -> Dict[str, Any]:
        """
        Run the basic simulation (logic from run_simulation).

        Returns
        -------
        dict
            Result dictionary with observables and time series.
        """
        L = self.L
        J = self.J
        T = self.T
        n_steps = self.n_steps
        n_equil = self.n_equil

        lat = XYLattice(L)
        spins = lat.initialize_lattice(L)
        V = L ** 3
        energies = []
        magnetizations = []
        cluster_sizes = []

        # Equilibration
        for _ in range(n_equil):
            WolffClusterUpdater.wolff_update(spins, J, T)

        # Measurement
        for _ in range(n_steps):
            cluster_size = WolffClusterUpdater.wolff_update(spins, J, T)
            cluster_sizes.append(cluster_size / V)

            E = core.compute_energy(spins, J)
            energies.append(E / V)

            m = core.compute_magnetization(spins)
            magnetizations.append(m)

        magnetizations = np.array(magnetizations)
        energies = np.array(energies)

        chi, error_chi = core.compute_susceptibility(magnetizations, V, T)
        C, error_C = core.compute_specific_heat(energies, V, T)
        U, error_U = core.compute_binder_cumulant(magnetizations, V)

        return {
            "magnetizations": magnetizations,
            "energies": energies,
            "cluster_sizes": cluster_sizes,
            "susceptibility": (chi, error_chi),
            "specific_heat": (C, error_C),
            "binder_cumulant": (U, error_U),
            # 'structure_factor': (S_q_mean, S_q_std),
            # 'correlation_length': (xi_mean, xi_std),
            # 'correlation_length': (correlation_lengths),
            "spin": spins,
        }

    def run_improved(self) -> Dict[str, Any]:
        """
        Run the improved-estimator simulation (logic from run_improved_simulation).

        Returns
        -------
        dict
            Result dictionary with cluster-based estimators.
        """
        L = self.L
        J = self.J
        T = self.T
        n_steps = self.n_steps
        n_equil = self.n_equil

        # now from wolff.py
        lat = XYLattice(L)
        spins = lat.initialize_lattice(L)
        V = L ** 3
        energies = []
        magnetizations = []
        cluster_sizes = []
        m2_improved = []

        # Equilibration
        for _ in range(n_equil):
            WolffClusterUpdater.wolff_update_with_estimator(spins, J, T)

        # Measurement
        for _ in range(n_steps):
            cluster_size, cluster_Sq, q_vectors = WolffClusterUpdater.wolff_update_with_estimator(
                spins, J, T
            )
            cluster_sizes.append(cluster_size / V)

            E = core.compute_energy(spins, J)
            energies.append(E / V)

            m = core.compute_magnetization(spins)
            magnetizations.append(m)

            # Improved estimator for m^2 via cluster size
            m2 = cluster_size / V**2
            m2_improved.append(m2)

        magnetizations = np.array(magnetizations)
        energies = np.array(energies)
        m2_improved = np.array(m2_improved)

        autocorr_m2 = core.compute_autocorrelation(m2_improved)
        tau_int_m2 = core.estimate_autocorrelation_time(autocorr_m2)
        print(f"Integrated Autocorrelation Time for m^2: {tau_int_m2:.2f}")

        chi, error_chi = core.compute_susceptibility_with_estimator(
            magnetizations, V, T
        )
        C, error_C = core.compute_specific_heat(energies, V, T)
        U, error_U = core.compute_binder_cumulant(magnetizations, V)

        return {
            "magnetizations": magnetizations,
            "energies": energies,
            "cluster_sizes": cluster_sizes,
            "m2_improved": m2_improved,
            "autocorrelation_time": tau_int_m2,
            "susceptibility": (chi, error_chi),
            "specific_heat": (C, error_C),
            "binder_cumulant": (U, error_U),
            # 'structure_factor': (S_q_mean, S_q_std),
            # 'correlation_length': (xi_mean, xi_std),
            "spins": spins,
        }


class XYStudy:
    """
    Study object that runs simulations over sets of (L, T).
    """

    def __init__(self, J: float, n_steps: int, n_equil: int, L_list: Sequence[int], T_list: Sequence[float]):
        """
        Parameters
        ----------
        L_list : Sequence[int]
            List of linear system size.
        T_list : Sequence[float]
            List of temperature.
        J : float
            Coupling constant.
        n_steps : int
            Number of Wolff steps.
        n_equil : int
            Number of equilibration steps.
        """
        self.J = J
        self.n_steps = n_steps
        self.n_equil = n_equil
        self.L_list = list(L_list)
        self.T_list = np.array(T_list, dtype=float)

    def run_simulation_all(self) -> Dict[int, Dict[float, Any]]:
        """
        Run run_simulation over all L and T using simulate_all_data.

        Returns
        -------
        dict
            Nested dictionary results[L][T].
        """
        all_data: Dict[int, Dict[float, Any]] = {}
        print(self.T_list)

        for L in self.L_list:
            all_data[L] = {}
            for T in self.T_list:
                sim = XYSimulation(L, float(T), self.J, self.n_steps, self.n_equil)
                res = sim.run_simulation()
                all_data[L][float(T)] = res

                U, error_U = res["binder_cumulant"]
                chi, error_chi = res["susceptibility"]
                C, error_C = res["specific_heat"]
                # xi, error_xi = all_data[L][T]['correlation_length']
                # xi = all_data[L][T]['correlation_length']

                print(
                    f"L={L}, T={T:.2f}, Binder Cumulant U={U:.4f}, "
                    f"Susceptibility={chi:.4f}, Specific Heat={C:.4f}"
                )

        return all_data

    def run_improved_all(self) -> Dict[int, Dict[float, Any]]:
        """
        Run improved simulations over all L and T using improved_simulate_all_data.

        Returns
        -------
        dict
            Nested dictionary results[L][T].
        """
        all_data: Dict[int, Dict[float, Any]] = {}

        for L in self.L_list:
            all_data[L] = {}
            for T in self.T_list:
                sim = XYSimulation(L, float(T), self.J, self.n_steps, self.n_equil)
                res = sim.run_improved()
                all_data[L][float(T)] = res

                U, error_U = res["binder_cumulant"]
                chi, error_chi = res["susceptibility"]
                C, error_C = res["specific_heat"]
                # xi, error_xi = all_data[L][T]['correlation_length']

                print(
                    f"L={L}, T={T:.2f}, Binder Cumulant U={U:.4f}, "
                    f"Susceptibility={chi:.4f}, Specific Heat={C:.4f}"
                )

        return all_data

    def run_single_basic(self, L: int, T: float) -> Dict[str, Any]:
        """
        Run a single (L, T) basic simulation.

        Returns
        -------
        dict
            Result from run_simulation.
        """
        sim = XYSimulation(L, T, self.J, self.n_steps, self.n_equil)
        return sim.run_simulation()

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
