## 3D XY Model Simulation

### Overview

This project implements a 3D XY spin model on cubic lattices of various sizes ($L$ = 8 – 20) to study continuous U(1) symmetry breaking and critical phenomena.
The model is simulated using two Monte Carlo update schemes:
*	Metropolis algorithm — a local update method that becomes inefficient near $T_{\langle C\rangle}$.
* 8 Wolff cluster algorithm — a non-local cluster update that mitigates critical slowing down.

### Physics Background

Each lattice site hosts a 2D unit vector spin $\mathbf S_i = (\cos\theta_i,\sin\theta_i)$, with Hamiltonian
$$\mathcal H=-J\sum_{\langle i,j\rangle}\cos(\theta_i-\theta_j).$$
At low temperature the system exhibits spontaneous U(1) symmetry breaking; at $T=T_c$ it undergoes a second-order phase transition belonging to the 3D XY universality class ($\nu ≈ 0.67,\ \eta ≈ 0.038$).

### Implementation Plan
* Generate XY spin configurations on periodic 3D lattices.
* Perform sweeps at multiple temperatures around $T_{\langle C\rangle}$.
* Measure observables: magnetization, susceptibility, Binder cumulant, specific heat, and correlation length.
* Apply finite-size scaling to extract $T_{\langle C\rangle}$ and critical exponents.
* Compare algorithmic efficiency via autocorrelation time and average cluster size.

###  Expected Results
* Precise estimate of the critical temperature $T_{\langle C\rangle}$.
* Scaling collapse plots and critical exponents consistent with 3D XY universality.
* Demonstration that Wolff updates reduce critical slowing down relative to Metropolis.

---

### Basic Usage Example

Below is a minimal working example showing how to:

1. Create a lattice  
2. Run XY simulations using the Wolff updater  
3. Measure observables  
4. Plot and fit observables  
5. Animated visualization of spin orientations

These features can be achieved in examples/wolff_basic_demo.py by choosing initial parameters
you want and start the simulation. If you want to perform fittings, examples/Advanced_fitting_demo.py shows 
the ways to call several critical fitting functions in src/core.py.

This assumes you installed the project via:

```bash
uv pip install -e .
```
Example: Run a Wolff-Based Simulation
```bash
from xy3d_wolff.simulation import XYSimulation
from xy3d_wolff.wolff import XYLattice
from xy3d_wolff.plotter import XYPlotter

# --- Simulation parameters ---
L = 8           # Lattice size
T = 2.2         # Temperature
J = 1.0         # Coupling
n_steps = 2000  # Number of Monte Carlo sweeps

# --- Initialize lattice ---
lat = XYLattice(L)

# --- Create simulation object ---
sim = XYSimulation(lattice=lat, J=J, T=T, n_steps=n_steps)

# --- Run simulation ---
results = sim.run_wolff()

print("Magnetization =", results["magnetization"])
print("Susceptibility =", results["susceptibility"])
print("Binder cumulant =", results["binder_cumulant"])
print("Correlation length =", results["correlation_length"])
```
Example: Sweep Over Multiple Temperatures
```bash
from xy3d_wolff.simulation import XYStudy

L_list = [8, 10, 12]
T_list = [1.6, 1.8, 2.0, 2.2, 2.3]

study = XYStudy(L_list=L_list, T_list=T_list, n_steps=1500)
all_results = study.simulate_all()

# Access results:
xi_10_T22 = all_results[10][2.2]["correlation_length"]
print("Correlation length for L=10, T=2.2:", xi_10_T22)
```
Directory structure reminder
```bash
xy3d_wolff/
│
├── src/xy3d_wolff/
│   ├── core.py
│   ├── wolff.py
│   ├── simulation.py
│   ├── plotter.py
│   └── __init__.py
│
├── tests/
├── examples/
├── pyproject.toml
└── README.md
```


