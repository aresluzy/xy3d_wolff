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
