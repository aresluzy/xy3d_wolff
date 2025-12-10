import numpy as np

from xy3d_wolff.simulation import SimulationConfig, SimulationRunner


def main():
    J = 1.0
    n_steps = 10000
    n_equil = 3000
    L_list = [6, 8, 10, 12]
    T_list = np.linspace(1.5, 3.0, 20)

    config = SimulationConfig(
        J=J,
        n_steps=n_steps,
        n_equil=n_equil,
        L_list=L_list,
        T_list=T_list,
    )
    runner = SimulationRunner(config)
    simulation_results = runner.run_basic()
    runner.plot_basic(simulation_results)

    J2 = 1.0
    n_steps2 = 10000
    n_equil2 = 5000
    L_list2 = [16]
    T_list2 = np.linspace(2.18, 2.25, 10)

    config2 = SimulationConfig(
        J=J2,
        n_steps=n_steps2,
        n_equil=n_equil2,
        L_list=L_list2,
        T_list=T_list2,
    )
    runner2 = SimulationRunner(config2)
    simulation_results_with_estimator = runner2.run_with_estimator()
    _ = simulation_results_with_estimator


if __name__ == "__main__":
    main()
