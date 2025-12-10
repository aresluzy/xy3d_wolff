import numpy as np

from xy3d_wolff.simulation import SimulationConfig, SimulationRunner


def test_simulation():
    J = 1.0
    n_steps = 20
    n_equil = 5
    L_list = [4]
    T_list = np.linspace(2.0, 2.4, 3)

    config = SimulationConfig(
        J=J,
        n_steps=n_steps,
        n_equil=n_equil,
        L_list=L_list,
        T_list=T_list,
    )
    runner = SimulationRunner(config)
    results = runner.run_basic()

    assert isinstance(results, dict)
    assert L_list[0] in results

    T0 = float(T_list[0])
    assert T0 in results[L_list[0]]

    sample = results[L_list[0]][T0]
    assert "energies" in sample
    assert "magnetizations" in sample
    assert isinstance(sample["energies"], list)
    assert isinstance(sample["magnetizations"], list)
    assert len(sample["energies"]) > 0
    assert len(sample["magnetizations"]) > 0
