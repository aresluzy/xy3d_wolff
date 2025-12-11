import numpy as np

from xy3d_wolff.simulation import SimulationConfig, XYStudy
from xy3d_wolff.plotting import XYPlotter
from xy3d_wolff.wolff import XYLattice
from xy3d_wolff import core


def main():
    # -------- basic simulation (no estimator) --------
    J = 1.0
    n_steps = 10_000
    n_equil = 3_000
    L_list = [6, 8, 10, 12]
    T_list = np.linspace(1.5, 3.0, 20)

    basic_config = SimulationConfig(
        J=J,
        n_steps=n_steps,
        n_equil=n_equil,
        L_list=L_list,
        T_list=T_list,
    )
    basic_study = XYStudy(basic_config)
    basic_results = basic_study.run_basic_all()

    XYPlotter.plot_observables(basic_results, L_list, T_list)

    # -------- improved-estimator simulation --------
    J2 = 1.0
    n_steps2 = 10_000
    n_equil2 = 5_000
    L_list2 = [16]
    T_list2 = np.linspace(2.18, 2.25, 10)

    improved_config = SimulationConfig(
        J=J2,
        n_steps=n_steps2,
        n_equil=n_equil2,
        L_list=L_list2,
        T_list=T_list2,
    )
    improved_study = XYStudy(improved_config)
    improved_results = improved_study.run_improved_all()

    # keep improved_results around so user can inspect/use it
    _ = improved_results

    # -------- finite-size fits at Tc (using your core fit functions) --------
    Tc_guess = 2.2
    L_list_fit = L_list

    a_fit, alpha_over_nu_fit, alpha_over_nu_err = core.fit_specific_heat_at_Tc(
        basic_results, L_list_fit, T_list, Tc_guess
    )
    b_fit, gamma_over_nu_fit, gamma_over_nu_err = core.fit_susceptibility_at_Tc(
        basic_results, L_list_fit, T_list, Tc_guess
    )
    c_fit, minus_beta_over_nu_fit, minus_beta_over_nu_err = (
        core.fit_magnetization_at_Tc(
            basic_results, L_list_fit, T_list, Tc_guess
        )
    )

    _ = (
        a_fit,
        alpha_over_nu_fit,
        alpha_over_nu_err,
        b_fit,
        gamma_over_nu_fit,
        gamma_over_nu_err,
        c_fit,
        minus_beta_over_nu_fit,
        minus_beta_over_nu_err,
    )

    # -------- data collapse using your core collapse routines --------
    # plug in whatever exponents you like (these can be your fitted values)
    alpha = 0.0
    beta = 0.35
    gamma = 1.3
    nu = 0.67

    core.data_collapse_specific_heat(
        basic_results, L_list, T_list, Tc_guess, alpha, nu, plot=True
    )
    core.data_collapse_susceptibility(
        basic_results, L_list, T_list, Tc_guess, gamma, nu, plot=True
    )
    core.data_collapse_magnetization(
        basic_results, L_list, T_list, Tc_guess, beta, nu, plot=True
    )

    # -------- spin orientation demo --------
    L_spin = 6
    lattice = XYLattice(L_spin)
    XYPlotter.plot_spins(lattice.spins)


if __name__ == "__main__":
    main()
