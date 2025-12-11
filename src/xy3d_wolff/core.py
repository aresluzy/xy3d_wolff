import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData
from scipy.interpolate import UnivariateSpline

# Compute critical parameters
def compute_improved_structure_factor(cluster_positions, L):
    """
    Compute the improved estimator for the structure factor S(q)
    using cluster positions from a Wolff update.

    This uses the cluster-improved estimator for the structure factor
    at the three smallest non-zero momenta along the Cartesian axes:
    q = (2π/L, 0, 0), (0, 2π/L, 0), (0, 0, 2π/L).

    Parameters
    ----------
    cluster_positions : list of tuple[int, int, int] or ndarray
        List of lattice coordinates (i, j, k) belonging to the cluster.
    L : int
    Linear system size. The lattice is assumed to be of size L^3.

    Returns
    -------
    cluster_Sq : ndarray, shape (3,)
        Improved-estimator structure factor S(q) for the three q-vectors.
    q_vectors : list of ndarray
        List of three q-vectors (each shape (3,)) corresponding to the
        smallest non-zero momenta along x, y, and z.
    """
    cluster_size = len(cluster_positions)

    q_vectors = [
        np.array([2 * np.pi / L, 0, 0]),  # Small q in x-direction
        np.array([0, 2 * np.pi / L, 0]),  # Small q in y-direction
        np.array([0, 0, 2 * np.pi / L])  # Small q in z-direction
    ]

    cluster_Sq = np.zeros(len(q_vectors))

    # Convert positions to numpy array
    positions = np.array(cluster_positions)  # Shape: (cluster_size, 3)

    # Compute sum over cluster positions
    for idx, q in enumerate(q_vectors):
        qr = np.dot(positions, q)  # Shape: (cluster_size,)
        sum_exp = np.sum(np.exp(1j * qr))
        cluster_Sq[idx] = (np.abs(sum_exp) ** 2) / (3 * cluster_size)

    return cluster_Sq, q_vectors

def compute_correlation_length_from_Sq(structure_factors, q_vectors, L):
    """
    Estimate a correlation length from structure factors S(q)
    using a second-moment-like estimator.

    Parameters
    ----------
    structure_factors : array-like
        Values of S(q) at a set of small momentum vectors.
    q_vectors : array-like
        Momentum vectors corresponding to `structure_factors`.
        Typically a list/array of shape (N_q, 3).
    L : int
        Linear system size. (Currently not used explicitly.)

    Returns
    -------
    xi : float
        Estimated correlation length. Returns np.inf if the moments
        are not positive (mu_0 <= 0 or mu_2 <= 0).
    """
    d = 3  # Spatial dimensionality
    S_q = np.array(structure_factors)  # Structure factors
    q_values = np.array(q_vectors)  # Magnitudes of q vectors

    # Zeroth moment (mu_0): Sum of S(q)
    mu_0 = np.sum(S_q)

    # Second moment (mu_2): Sum of |q|^2 * S(q)
    mu_2 = np.sum((sum(np.abs(q_values)) ** 2) * S_q)

    # Avoid division by zero or invalid results
    if mu_0 <= 0 or mu_2 <= 0:
        return np.inf  # Infinite correlation length if moments are invalid

    # Compute correlation length
    xi = np.sqrt(mu_2 / (2 * d * mu_0))
    return xi

def compute_autocorrelation(data):
    """
    Compute the autocorrelation function of a 1D array.

    Parameters:
        data (array-like): Time series data.

    Returns:
        autocorr (ndarray): Autocorrelation function.
    """
    data = np.asarray(data)
    n = len(data)
    data_mean = np.mean(data)
    data_var = np.var(data)

    autocorr = np.correlate(data - data_mean, data - data_mean, mode='full')
    autocorr = autocorr[n - 1:] / (data_var * n)

    return autocorr

def estimate_autocorrelation_time(autocorr):
    """
    Estimate the integrated autocorrelation time.

    Parameters:
        autocorr (ndarray): Autocorrelation function.

    Returns:
        tau_int (float): Integrated autocorrelation time.
    """
    # Integrated autocorrelation time
    # Sum until the autocorrelation function drops below zero
    positive_autocorr = autocorr[autocorr > 0]
    tau_int = 0.5 + np.sum(positive_autocorr[1:])  # Skip the first term (t=0)

    return tau_int

def compute_susceptibility_with_estimator(magnetizations, V, T):
    """
    Compute the magnetic susceptibility and its statistical error estimate
    using improved-estimator magnetization data.

    Parameters
    ----------
    magnetizations : array-like
        Array of magnetization per site (or per volume) measurements.
    V : int or float
        System volume (e.g. L^3).
    T : float
        Temperature.

    Returns
    -------
    chi : float
        Estimated susceptibility.
    error_chi : float
        Standard error estimate of the susceptibility.
    """
    M_abs = np.array(magnetizations) * V  # Total magnetization
    M_abs_mean = np.mean(M_abs)
    M2_mean = np.mean(M_abs ** 2)
    chi = (M2_mean - M_abs_mean ** 2) / (V * T)

    N = len(M_abs)
    numerator = M_abs ** 2 - M_abs_mean ** 2
    sigma_numerator = np.std(numerator, ddof=1)
    error_chi = sigma_numerator / (np.sqrt(N) * V * T)

    return chi, error_chi

def compute_F(spins, L):
    """
    Compute F = Ĝ(k) at |k| = 2π/L for the 3D XY model.

    Here Ĝ(k) is the Fourier transform of the spin field at the
    smallest non-zero momentum magnitude along the three axes, and
    this function returns the average over those directions.

    Parameters
    ----------
    spins : ndarray, shape (L, L, L, 2)
        Spin configuration, where the last dimension holds (Sx, Sy).
    L : int
        Linear system size.

    Returns
    -------
    F_mean : float
        Direction-averaged value of |∑_r S(r) e^{-i k·r}|^2 / V
        for |k| = 2π/L.
    """
    V = L ** 3
    # Smallest non-zero momentum vectors
    q_vectors = [
        np.array([2 * np.pi / L, 0, 0]),
        np.array([0, 2 * np.pi / L, 0]),
        np.array([0, 0, 2 * np.pi / L])
    ]

    # Initialize F
    F_values = []

    # Convert spins to complex numbers
    spin_complex = spins[..., 0] + 1j * spins[..., 1]  # Shape: (L, L, L)

    # Coordinates
    x = np.arange(L)
    y = np.arange(L)
    z = np.arange(L)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    positions = np.stack([X, Y, Z], axis=-1)  # Shape: (L, L, L, 3)

    for q in q_vectors:
        # Compute phase factors
        r_dot_q = np.tensordot(positions, q, axes=([3], [0]))  # Shape: (L, L, L)
        phase = np.exp(-1j * r_dot_q)  # Shape: (L, L, L)

        # Compute sum over spins
        sum_sq = np.sum(spin_complex * phase)
        F = np.abs(sum_sq) ** 2 / V
        F_values.append(F)

    # Average over directions
    F_mean = np.mean(F_values)
    return F_mean

def compute_xi_second_moment(chi, F, L):
    """
    Compute the second-moment correlation length from susceptibility
    and Fourier amplitude F at |k| = 2π/L.

    The estimator is

        xi_2nd = sqrt( |chi/F - 1| / (4 sin^2(pi/L)) ).

    Parameters
    ----------
    chi : float
        Susceptibility (typically χ = ⟨M^2⟩ / V T or similar).
    F : float
        Fourier amplitude F = Ĝ(k) at the smallest non-zero |k|.
    L : int
        Linear system size.

    Returns
    -------
    xi_2nd : float
        Second-moment correlation length estimate.
    """
    pi_over_L = np.pi / L
    sin_term = np.sin(pi_over_L)
    denominator = 4 * sin_term ** 2
    ratio = chi / F
    xi_2nd = np.sqrt(np.abs(ratio - 1) / denominator)
    return xi_2nd

def compute_magnetization(spins):
    """
    Compute the magnetization per site for a 3D XY spin configuration.

    Parameters
    ----------
    spins : ndarray, shape (L, L, L, 2)
        Spin configuration, where the last axis is (Sx, Sy).

    Returns
    -------
    m : float
        Magnetization per site, m = |∑_r S(r)| / V, with V = L^3.
    """
    total_spin = np.sum(spins, axis=(0, 1, 2))
    M = np.linalg.norm(total_spin)
    V = spins.shape[0] ** 3
    m = M / V
    return m

def compute_susceptibility(magnetizations, V, T):
    """
    Compute the magnetic susceptibility and its statistical error
    from magnetization-per-site measurements.

    Parameters
    ----------
    magnetizations : array-like
        Measurements of magnetization per site m.
    V : int or float
        System volume (e.g. L^3).
    T : float
        Temperature.

    Returns
    -------
    chi : float
        Susceptibility estimate.
    error_chi : float
        Standard error of the susceptibility.
    """
    M_abs = np.array(magnetizations) * V  # Total magnetization
    M_abs_mean = np.mean(M_abs)
    M2_mean = np.mean(M_abs ** 2)
    chi = (M2_mean - M_abs_mean ** 2) / (V * T)

    N = len(M_abs)
    numerator = M_abs ** 2 - M_abs_mean ** 2
    sigma_numerator = np.std(numerator, ddof=1)
    error_chi = sigma_numerator / (np.sqrt(N) * V * T)

    return chi, error_chi

def compute_binder_cumulant(magnetizations, V):
    """
    Compute the Binder cumulant U and its error from magnetization data.

    Parameters
    ----------
    magnetizations : array-like
        Magnetization per site m for each measurement.
    V : int or float
        System volume (e.g. L^3).

    Returns
    -------
    U : float
        Binder cumulant U = 1 - ⟨M^4⟩ / (3 ⟨M^2⟩^2),
        where M is the total magnetization.
    error_U : float
        Error estimate for the Binder cumulant.
    """
    M_abs = np.array(magnetizations) * V
    M2 = M_abs ** 2
    M4 = M_abs ** 4
    U = 1 - np.mean(M4) / (3 * np.mean(M2) ** 2)

    N = len(M_abs)
    sigma_M2 = np.std(M2, ddof=1) / np.sqrt(N)
    sigma_M4 = np.std(M4, ddof=1) / np.sqrt(N)
    dU_dM4 = -1 / (3 * np.mean(M2) ** 2)
    dU_dM2 = (2 * np.mean(M4)) / (3 * np.mean(M2) ** 3)

    # Standard error
    error_U = np.sqrt((dU_dM4 * sigma_M4) ** 2 + (dU_dM2 * sigma_M2) ** 2)
    return U, error_U

def compute_energy(spins, J):
    """
    Compute the total energy of a 3D XY configuration
    with nearest-neighbor interactions and coupling J.

    Parameters
    ----------
    spins : ndarray, shape (L, L, L, 2)
        Spin configuration, last axis is (Sx, Sy).
    J : float
        Coupling constant.

    Returns
    -------
    E : float
        Total energy. Each bond is counted twice.
    """
    L = spins.shape[0]
    E = 0.0
    for shift in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        shifted_spins = np.roll(spins, shift=shift, axis=(0, 1, 2))
        dot_product = 2 * np.sum(spins * shifted_spins, axis=-1)
        E -= J * np.sum(dot_product)
    return E  # Each bond is counted twice

def compute_specific_heat(energies, V, T):
    """
    Compute the specific heat and its error using block variance analysis.

    Parameters
    ----------
    energies : array-like
        Energy measurements E for different Monte Carlo samples.
    V : int or float
        System volume (e.g. L^3).
    T : float
        Temperature.

    Returns
    -------
    C : float
        Specific heat (per ??? – note C is not divided by V in this code).
    sigma_C : float
        Error estimate for the specific heat, including block analysis.
    """
    E = np.array(energies)
    N = len(E)
    E_mean = np.mean(E)
    E2_mean = np.mean(E ** 2)
    variance_E = E2_mean - E_mean ** 2
    C = variance_E / (3 * T ** 2)

    # Number of blocks
    M = 10
    m = N // M  # Ensure m is an integer

    # Split energies into M blocks
    variances = []
    for k in range(M):
        E_block = E[k * m:(k + 1) * m]
        E_mean_block = np.mean(E_block)
        E2_mean_block = np.mean(E_block ** 2)
        variance_block = E2_mean_block - E_mean_block ** 2
        variances.append(variance_block)

    variances = np.array(variances)
    variance_mean = np.mean(variances)
    sigma_variance = np.sqrt(np.sum((variances - variance_mean) ** 2) / (M - 1)) / np.sqrt(M)

    # Error in specific heat
    sigma_C = sigma_variance / (V * T ** 2)

    return C, sigma_C

def compute_structure_factor(spins, n_max=5):
    """
    Compute the structure factor S(q) for multiple momenta
    along the Cartesian axes.

    Parameters
    ----------
    spins : ndarray, shape (L, L, L, 2)
        Spin configuration, last axis is (Sx, Sy).
    n_max : int, optional
        Maximum integer multiple n for q = 2π n / L.

    Returns
    -------
    S_q : ndarray
        Structure factor values for all considered q-vectors.
    q_values : ndarray
        Magnitudes |q| corresponding to each S(q).
    """
    L = spins.shape[0]
    N = L ** 3
    S_q = []
    q_values = []
    spin_complex = spins[..., 0] + 1j * spins[..., 1]

    x = np.arange(L)[:, None, None]
    y = np.arange(L)[None, :, None]
    z = np.arange(L)[None, None, :]

    for n in range(1, n_max + 1):
        q = 2 * np.pi * n / L
        for q_vector in [
            np.array([q, 0, 0]),
            np.array([0, q, 0]),
            np.array([0, 0, q])
        ]:
            r_dot_q = q_vector[0] * x + q_vector[1] * y + q_vector[2] * z
            phase = np.exp(-1j * r_dot_q)
            S_q_value = np.abs(np.sum(spin_complex * phase)) ** 2 / N
            S_q.append(S_q_value)
            q_values.append(np.linalg.norm(q_vector))
    return np.array(S_q), np.array(q_values)

def compute_S0(spins):
    """
    Compute S(0), the zero-momentum structure factor.

    Parameters
    ----------
    spins : ndarray, shape (L, L, L, 2)
        Spin configuration, last axis is (Sx, Sy).

    Returns
    -------
    S0 : float
        Zero-momentum structure factor S(0) = |∑_r S(r)|^2 / N.
    """
    # Convert spins to complex numbers
    spin_complex = spins[..., 0] + 1j * spins[..., 1]  # Shape: (L, L, L)
    N = spins.shape[0] ** 3
    S0 = np.abs(np.sum(spin_complex)) ** 2 / N
    return S0

def estimate_correlation_length(S0, S_q, L):
    """
    Estimate a correlation length from S(0) and S(q) at |q| = 2π/L.

    Parameters
    ----------
    S0 : float
        Zero-momentum structure factor.
    S_q : float
        Structure factor at the smallest non-zero momentum |q| = 2π/L.
    L : int
        Linear system size.

    Returns
    -------
    xi : float
        Estimated correlation length. Returns np.inf if S_q <= 0 or S0 <= S_q.
    """
    q = 2 * np.pi / L  # Smallest non-zero momentum

    # Ensure S_q is not zero to avoid division by zero
    if S_q == 0 or S0 <= S_q:
        return np.inf

    xi = 1 / q * np.sqrt(np.abs(S0 / S_q - 1))

    return xi

def compute_correlation_length_from_cluster_sizes(cluster_sizes, V, num_bins=100):
    """
    Estimate a correlation length from the distribution of cluster sizes.

    Parameters
    ----------
    cluster_sizes : array-like
        Cluster sizes normalized by volume (e.g. s / V).
    V : int or float
        System volume used to un-normalize cluster sizes.
    num_bins : int, optional
        Number of histogram bins for building n_s.

    Returns
    -------
    xi : float
        Correlation length estimate derived from the ratio of
        moments of the cluster-size distribution.

    Raises
    ------
    ValueError
        If the denominator in the correlation length expression is zero.
    """
    # Convert cluster sizes to actual sizes (since they are normalized by V)
    actual_cluster_sizes = np.array(cluster_sizes) * V  # Now cluster sizes are integers

    # Remove zero or negligible cluster sizes (if any)
    actual_cluster_sizes = actual_cluster_sizes[actual_cluster_sizes > 0.5]

    # Build histogram: n_s is the number of clusters of size s
    counts, bin_edges = np.histogram(actual_cluster_sizes, bins=num_bins)

    # Compute bin centers (representative s values)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Avoid zeros in bin centers (to prevent division by zero)
    valid_indices = bin_centers > 0
    bin_centers = bin_centers[valid_indices]
    counts = counts[valid_indices]

    # Compute s^2 * n_s and s^{8/3} * n_s
    s2_ns = bin_centers ** 2 * counts
    s8_3_ns = bin_centers ** (8 / 3) * counts

    # Compute numerator and denominator
    numerator = np.sum(s8_3_ns)
    denominator = np.sum(s2_ns)

    # Check for zero denominator
    if denominator == 0:
        raise ValueError("Denominator in correlation length calculation is zero.")

    # Estimate xi^2 (proportional to numerator/denominator)
    xi_squared = numerator / denominator

    # Compute xi
    xi = np.sqrt(xi_squared)

    return xi

def compute_spin_correlation(spins, max_r):
    """
    Compute the spin–spin correlation function G(r) up to distance max_r.

    For each integer separation r = 1, ..., max_r, this computes

        G(r) = (1/N) ∑_i S_i · S_{i+r},

    where the shift (i → i+r) is applied along all three directions with
    periodic boundary conditions, and N = L^3 is the number of lattice sites.

    Parameters
    ----------
    spins : ndarray, shape (L, L, L, 2)
        Spin configuration, where the last axis is (Sx, Sy).
    max_r : int
        Maximum separation r for which to compute G(r).

    Returns
    -------
    G_r : ndarray, shape (max_r,)
        Spin–spin correlation function values G(r) for r = 1, ..., max_r.
    """
    L = spins.shape[0]
    G_r = np.zeros(max_r)
    N = L**3

    for r in range(1, max_r + 1):
        corr = 0
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    # Periodic boundary conditions
                    ni = (i + r) % L
                    nj = (j + r) % L
                    nk = (k + r) % L
                    corr += np.dot(spins[i, j, k], spins[ni, nj, nk])
        G_r[r - 1] = corr / N

    return G_r

def compute_structure_factor(spins, q_vals):
    """
    Compute the structure factor S(q) using a simple Fourier transform
    along one lattice direction.

    Parameters
    ----------
    spins : ndarray, shape (L, L, L, 2)
        Spin configuration, where the last axis is (Sx, Sy) or some
        spin components.
    q_vals : array-like
        List or array of momentum values q (typically scalars) for which
        S(q) should be computed.

    Returns
    -------
    S_q : ndarray, shape (len(q_vals),)
        Structure factor values S(q) for each q in q_vals.
    """
    L = spins.shape[0]
    N = L**3
    S_q = []

    for q in q_vals:
        # Fourier transform
        spin_sum = np.sum(spins * np.exp(-1j * q * np.arange(L)), axis=(0, 1, 2))
        S_q.append(np.abs(spin_sum)**2 / N)

    return np.array(S_q)

def fit_correlation_length(G_r):
    """
    Fit the real-space correlation function G(r) to an exponential decay
    to extract the correlation length ξ.

    The fitting model is

        G(r) ≈ A * exp(-r / ξ),

    where ξ and A are fit parameters.

    Parameters
    ----------
    G_r : ndarray
        Values of the correlation function G(r) for r = 1, 2, ..., len(G_r).

    Returns
    -------
    xi : float
        Fitted correlation length ξ.

    Notes
    -----
    This function uses `scipy.optimize.curve_fit` under the hood, so
    make sure you have::

        from scipy.optimize import curve_fit

    imported in the module where this function is defined.
    """
    def exponential_decay(r, xi, A):
        return A * np.exp(-r / xi)

    r_vals = np.arange(1, len(G_r) + 1)
    popt, _ = curve_fit(exponential_decay, r_vals, G_r, p0=[1.0, 1.0])
    return popt[0]  # Return ξ

# Plotting functions
def plot_simulation_results(simulation_results, L_list, T_list):
    T_c = 2.2  # Critical temperature

    # Plot Magnetization
    plt.figure()
    for L in L_list:
        temperatures = []
        avg_magnetizations = []

        for T in T_list:
            results = simulation_results[L][T]
            temperatures.append(T)
            avg_magnetizations.append(np.mean(results['magnetizations']))

        plt.plot(temperatures, avg_magnetizations, 'o-', label=f"L={L}")
    plt.axvline(x=T_c, color='r', linestyle='--', label=f"T_c = {T_c}")
    plt.title("Magnetization vs Temperature")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Energy
    plt.figure()
    for L in L_list:
        temperatures = []
        avg_energies = []

        for T in T_list:
            results = simulation_results[L][T]
            temperatures.append(T)
            avg_energies.append(np.mean(results['energies']))

        plt.plot(temperatures, avg_energies, 'o-', label=f"L={L}")
    plt.axvline(x=T_c, color='r', linestyle='--', label=f"T_c = {T_c}")
    plt.title("Energy vs Temperature")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Cluster Size
    plt.figure()
    for L in L_list:
        temperatures = []
        avg_cluster_sizes = []

        for T in T_list:
            results = simulation_results[L][T]
            temperatures.append(T)
            avg_cluster_sizes.append(np.mean(results['cluster_sizes']))

        plt.plot(temperatures, avg_cluster_sizes, 'o-', label=f"L={L}")
    plt.axvline(x=T_c, color='r', linestyle='--', label=f"T_c = {T_c}")
    plt.title("Cluster Size vs Temperature")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Cluster Size")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Susceptibility
    plt.figure()
    for L in L_list:
        temperatures = []
        susceptibilities = []
        susceptibility_errors = []

        for T in T_list:
            results = simulation_results[L][T]
            temperatures.append(T)
            susceptibilities.append(results['susceptibility'][0])
            susceptibility_errors.append(results['susceptibility'][1])

        plt.errorbar(temperatures, susceptibilities, yerr=susceptibility_errors, fmt='o-', label=f"L={L}")
    plt.axvline(x=T_c, color='r', linestyle='--', label=f"T_c = {T_c}")
    plt.title("Susceptibility vs Temperature")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Susceptibility")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Specific Heat
    plt.figure()
    for L in L_list:
        temperatures = []
        specific_heats = []
        specific_heat_errors = []

        for T in T_list:
            results = simulation_results[L][T]
            temperatures.append(T)
            specific_heats.append(results['specific_heat'][0])
            log_c = np.log(specific_heats)
            specific_heat_errors.append(results['specific_heat'][1])

        plt.errorbar(temperatures, specific_heats, yerr=specific_heat_errors, fmt='o-', label=f"L={L}")
    plt.axvline(x=T_c, color='r', linestyle='--', label=f"T_c = {T_c}")
    plt.title("Specific Heat vs Temperature")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Specific Heat")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Binder Cumulant
    plt.figure()
    for L in L_list:
        temperatures = []
        binder_cumulants = []
        binder_cumulant_errors = []

        for T in T_list:
            results = simulation_results[L][T]
            temperatures.append(T)
            binder_cumulants.append(results['binder_cumulant'][0])
            binder_cumulant_errors.append(results['binder_cumulant'][1])

        plt.errorbar(temperatures, binder_cumulants, yerr=binder_cumulant_errors, fmt='o-', label=f"L={L}")
    plt.axvline(x=T_c, color='r', linestyle='--', label=f"T_c = {T_c}")
    plt.title("Binder Cumulant vs Temperature")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Binder Cumulant")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Correlation Length
    # plt.figure()
    # for L in L_list:
    #     temperatures = []
    #     Correlation_length = []
    #     Correlation_length_errors = []

    #     for T in T_list:
    #         results = simulation_results[L][T]
    #         temperatures.append(T)
    #         Correlation_length.append(results['correlation_length'][0])
    #         Correlation_length_errors.append(results['correlation_length'][1])

    #     plt.errorbar(temperatures, Correlation_length, yerr=Correlation_length_errors, fmt='o-', label=f"L={L}")
    # plt.axvline(x=T_c, color='r', linestyle='--', label=f"T_c = {T_c}")
    # plt.title("Correlation Length vs Temperature")
    # plt.xlabel("Temperature (T)")
    # plt.ylabel("Correlation Length")
    # plt.legend()
    # plt.grid()
    # plt.show()

def plot_spin_orientations(simulation_results, L, T):
    """
    Plots the 3D spin orientations for a specific lattice size L at a specific temperature T.

    Parameters:
        simulation_results (dict): Simulation results containing spin data.
                                   Format: {L: {T: {'spins': ndarray of shape (L, L, L, 2)}, ...}, ...}.
        L (int): Lattice size.
        T (float): Temperature.
    """
    # Extract spin data
    if L not in simulation_results or T not in simulation_results[L]:
        print(f"No data available for L={L} and T={T}.")
        return

    spins = simulation_results[L][T]['spin']  # Expected shape: (L, L, L, 2)

    # Validate spins shape
    if spins.shape[:3] != (L, L, L) or spins.shape[-1] != 2:
        print(f"Invalid spins data shape: {spins.shape}. Expected (L, L, L, 2).")
        return

    # Generate grid points for lattice
    x, y, z = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing='ij')

    # Compute spin components
    u = spins[..., 0]  # x-component of spin
    v = spins[..., 1]  # y-component of spin
    w = np.zeros_like(u)  # z-component is zero for XY model

    # Create the 3D quiver plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x, y, z, u, v, w, length=0.5, normalize=True, color='blue', alpha=0.8)

    # Label axes
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'Spin Orientations (L={L}, T={T})')

    # Set limits
    ax.set_xlim([0, L - 1])
    ax.set_ylim([0, L - 1])
    ax.set_zlim([0, L - 1])

    plt.show()

import pickle



import imageio
import os

def create_spin_orientation_gif(simulation_results, L, T_list, interval, output_file="spin_orientations.gif"):
    """
    Creates a GIF showing 3D spin orientations for all temperatures in T_list.

    Parameters:
        simulation_results (dict): Simulation results containing spin data.
                                   Format: {L: {T: {'spins': ndarray of shape (L, L, L, 2)}, ...}, ...}.
        L (int): Lattice size.
        T_list (list): List of temperatures.
        output_file (str): Filename for the output GIF.
        interval (int): Time interval (ms) between frames in the GIF.
    """
    temp_dir = "frames"
    os.makedirs(temp_dir, exist_ok=True)
    frames = []

    for T in T_list:
        # Check if data for L and T exists
        if L not in simulation_results or T not in simulation_results[L]:
            print(f"No data available for L={L} and T={T}. Skipping...")
            continue

        spins = simulation_results[L][T].get('spin', None)
        if spins is None:
            print(f"Spin data not found for L={L} and T={T}. Skipping...")
            continue

        # Validate spins shape
        if spins.shape[:3] != (L, L, L) or spins.shape[-1] != 2:
            print(f"Invalid spins data shape for L={L} and T={T}: {spins.shape}. Expected (L, L, L, 2).")
            continue

        # Generate grid points for lattice
        x, y, z = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing='ij')

        # Compute spin components
        u = spins[..., 0]  # x-component of spin
        v = spins[..., 1]  # y-component of spin
        w = np.zeros_like(u)  # z-component is zero for XY model

        # Create the 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(x, y, z, u, v, w, length=0.5, normalize=True, color='blue', alpha=0.8)

        # Label axes and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(f'Spin Orientations (L={L}, T={T})')

        # Set limits
        ax.set_xlim([0, L - 1])
        ax.set_ylim([0, L - 1])
        ax.set_zlim([0, L - 1])

        # Save the current frame
        frame_path = os.path.join(temp_dir, f"frame_{T}.png")
        plt.savefig(frame_path)
        frames.append(frame_path)
        plt.close(fig)

    # Create GIF
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(output_file, images, duration=interval)

    # Cleanup temporary files
    for frame in frames:
        os.remove(frame)
    os.rmdir(temp_dir)

    print(f"GIF saved as {output_file}")


# Fitting functions
def xi_fit(T, A, nu, Tc=None):
    if Tc is not None:
        return A * np.abs(T - Tc) ** (-nu)
    else:
        return A * np.abs(T) ** (-nu)

def C_fit(T, A, Tc, alpha):
    return A * np.abs(T - Tc) ** (-alpha)

def chi_fit(T, A, Tc, gamma):
    return A * np.abs(T - Tc) ** (-gamma)

def fit_correlation_length(simulation_results, L_list, T_list, Tc=None, plot=True):
    """
    Fits the correlation length data to extract the critical exponent ν.

    Parameters:
        simulation_results (dict): Output from simulate_all_data.
        L_list (list): List of system sizes.
        T_list (list or ndarray): List of temperatures.
        Tc (float, optional): Critical temperature. If None, Tc is fitted.
        plot (bool): If True, plots the data and the fit.

    Returns:
        dict: Fitted parameters {'A': value, 'Tc': value, 'nu': value} and their errors.
    """
    # Collect data across all system sizes
    T_all = []
    xi_all = []
    xi_err_all = []

    for L in L_list:
        for T in T_list:
            results = simulation_results[L][T]
            xi_mean, xi_std = results['correlation_length']
            T_all.append(T)
            xi_all.append(xi_mean)
            xi_err_all.append(xi_std)

    T_all = np.array(T_all)
    xi_all = np.array(xi_all)
    xi_err_all = np.array(xi_err_all)

    # Sort data by temperature
    sort_indices = np.argsort(T_all)
    T_all = T_all[sort_indices]
    xi_all = xi_all[sort_indices]
    xi_err_all = xi_err_all[sort_indices]

    # Define fitting function
    if Tc is None:
        # Fit Tc as a parameter
        def xi_fit(T, A, Tc_fit, nu):
            return A * np.abs(T - Tc_fit) ** (-nu)

        p0 = [np.max(xi_all), np.mean(T_all), 0.67]
        bounds = ([0, np.min(T_all), 0], [np.inf, np.max(T_all), np.inf])
    else:
        # Fix Tc
        def xi_fit(T, A, nu):
            return A * np.abs(T - Tc) ** (-nu)

        p0 = [np.max(xi_all), 0.67]
        bounds = ([0, 0], [np.inf, np.inf])

    # Perform the fit
    try:
        if Tc is None:
            params, cov = curve_fit(
                xi_fit, T_all, xi_all, p0=p0, sigma=xi_err_all,
                absolute_sigma=True, bounds=bounds
            )
            A_fit, Tc_fit, nu_fit = params
            errors = np.sqrt(np.diag(cov))
            A_err, Tc_err, nu_err = errors
        else:
            params, cov = curve_fit(
                xi_fit, T_all, xi_all, p0=p0, sigma=xi_err_all,
                absolute_sigma=True, bounds=bounds
            )
            A_fit, nu_fit = params
            errors = np.sqrt(np.diag(cov))
            A_err, nu_err = errors
            Tc_fit = Tc
            Tc_err = 0
    except Exception as e:
        print(f"Fit failed: {e}")
        return None

    # Plotting
    if plot:
        T_fit = np.linspace(np.min(T_all), np.max(T_all), 500)
        if Tc is None:
            xi_fit_values = xi_fit(T_fit, A_fit, Tc_fit, nu_fit)
        else:
            xi_fit_values = xi_fit(T_fit, A_fit, nu_fit)

        plt.figure(figsize=(8, 6))
        plt.errorbar(T_all, xi_all, yerr=xi_err_all, fmt='o', label='Simulation Data')
        plt.plot(T_fit, xi_fit_values, '-', label=f'Fit: ν = {nu_fit:.4f}')
        plt.xlabel('Temperature T')
        plt.ylabel('Correlation Length ξ(T)')
        plt.title('Correlation Length vs Temperature')
        plt.legend()
        plt.show()

    # Return fitted parameters
    fit_results = {
        'A': (A_fit, A_err),
        'Tc': (Tc_fit, Tc_err),
        'nu': (nu_fit, nu_err)
    }
    return fit_results

def fit_susceptibility(simulation_results, L_list, T_list, Tc=None, plot=True):
    """
    Fits the susceptibility data to extract the critical exponent γ.

    Parameters:
        simulation_results (dict): Output from simulate_all_data.
        L_list (list): List of system sizes.
        T_list (list or ndarray): List of temperatures.
        Tc (float, optional): Critical temperature. If None, Tc is fitted.
        plot (bool): If True, plots the data and the fit.

    Returns:
        dict: Fitted parameters {'A': value, 'Tc': value, 'gamma': value} and their errors.
    """
    # Collect data across all system sizes
    T_all = []
    chi_all = []
    chi_err_all = []

    for L in L_list:
        for T in T_list:
            results = simulation_results[L][T]
            (chi_mean, chi_err) = results['susceptibility']
            T_all.append(T)
            chi_all.append(chi_mean)
            chi_err_all.append(chi_err)

    T_all = np.array(T_all)
    chi_all = np.array(chi_all)
    chi_err_all = np.array(chi_err_all)

    # Sort data by temperature
    sort_indices = np.argsort(T_all)
    T_all = T_all[sort_indices]
    chi_all = chi_all[sort_indices]
    chi_err_all = chi_err_all[sort_indices]

    # Define fitting function
    if Tc is None:
        # Fit Tc as a parameter
        def chi_fit(T, A, Tc_fit, gamma):
            return A * np.abs(T - Tc_fit) ** (-gamma)

        p0 = [np.max(chi_all), np.mean(T_all), 1.3]
        bounds = ([0, np.min(T_all), 0], [np.inf, np.max(T_all), np.inf])
    else:
        # Fix Tc
        def chi_fit(T, A, gamma):
            return A * np.abs(T - Tc) ** (-gamma)

        p0 = [np.max(chi_all), 1.3]
        bounds = ([0, 0], [np.inf, np.inf])

    # Perform the fit
    try:
        if Tc is None:
            params, cov = curve_fit(
                chi_fit, T_all, chi_all, p0=p0, sigma=chi_err_all,
                absolute_sigma=True, bounds=bounds
            )
            A_fit, Tc_fit, gamma_fit = params
            errors = np.sqrt(np.diag(cov))
            A_err, Tc_err, gamma_err = errors
        else:
            params, cov = curve_fit(
                chi_fit, T_all, chi_all, p0=p0, sigma=chi_err_all,
                absolute_sigma=True, bounds=bounds
            )
            A_fit, gamma_fit = params
            errors = np.sqrt(np.diag(cov))
            A_err, gamma_err = errors
            Tc_fit = Tc
            Tc_err = 0
    except Exception as e:
        print(f"Fit failed: {e}")
        return None

    # Plotting
    if plot:
        T_fit = np.linspace(np.min(T_all), np.max(T_all), 500)
        if Tc is None:
            chi_fit_values = chi_fit(T_fit, A_fit, Tc_fit, gamma_fit)
        else:
            chi_fit_values = chi_fit(T_fit, A_fit, gamma_fit)

        plt.figure(figsize=(8, 6))
        plt.errorbar(T_all, chi_all, yerr=chi_err_all, fmt='o', label='Simulation Data')
        plt.plot(T_fit, chi_fit_values, '-', label=f'Fit: γ = {gamma_fit:.4f}')
        plt.xlabel('Temperature T')
        plt.ylabel('Susceptibility χ(T)')
        plt.title('Susceptibility vs Temperature')
        plt.legend()
        plt.show()

    # Return fitted parameters
    fit_results = {
        'A': (A_fit, A_err),
        'Tc': (Tc_fit, Tc_err),
        'gamma': (gamma_fit, gamma_err)
    }
    return fit_results

def fit_specific_heat_per_L(simulation_results, L_list, T_list, Tc=None, plot=True):
    fit_results = {}

    for L in L_list:
        # Collect data for this lattice size
        T_all = []
        C_all = []
        C_err_all = []

        for T in T_list:
            results = simulation_results[L][T]
            (C_mean, C_err) = results['specific_heat']
            T_all.append(T)
            C_all.append(C_mean)
            C_err_all.append(C_err)

        T_all = np.array(T_all)
        C_all = np.array(C_all)
        C_err_all = np.array(C_err_all)

        # Sort data by temperature
        sort_indices = np.argsort(T_all)
        T_all = T_all[sort_indices]
        C_all = C_all[sort_indices]
        C_err_all = C_err_all[sort_indices]

        # Define fitting function
        if Tc is None:
            # Fit Tc as a parameter
            def C_fit(T, A, Tc_fit, alpha):
                return A * np.abs(T - Tc_fit) ** (-alpha)

            p0 = [np.max(C_all), np.mean(T_all), 0.0]
            bounds = ([0, np.min(T_all), -1], [np.inf, np.max(T_all), 2])
        else:
            # Fix Tc
            def C_fit(T, A, alpha):
                return A * np.abs(T - Tc) ** (-alpha)

            p0 = [np.max(C_all), 0.0]
            bounds = ([0, -1], [np.inf, 2])

        # Perform the fit
        try:
            if Tc is None:
                params, cov = curve_fit(
                    C_fit, T_all, C_all, p0=p0, sigma=C_err_all,
                    absolute_sigma=True, bounds=bounds
                )
                A_fit, Tc_fit, alpha_fit = params
                errors = np.sqrt(np.diag(cov))
                A_err, Tc_err, alpha_err = errors
            else:
                params, cov = curve_fit(
                    C_fit, T_all, C_all, p0=p0, sigma=C_err_all,
                    absolute_sigma=True, bounds=bounds
                )
                A_fit, alpha_fit = params
                errors = np.sqrt(np.diag(cov))
                A_err, alpha_err = errors
                Tc_fit = Tc
                Tc_err = 0
        except Exception as e:
            print(f"Fit failed for L={L}: {e}")
            fit_results[L] = None
            continue

        # Store results
        fit_results[L] = {
            'A': (A_fit, A_err),
            'Tc': (Tc_fit, Tc_err),
            'alpha': (alpha_fit, alpha_err)
        }

        # Plotting
        if plot:
            T_fit = np.linspace(np.min(T_all), np.max(T_all), 500)
            if Tc is None:
                C_fit_values = C_fit(T_fit, A_fit, Tc_fit, alpha_fit)
            else:
                C_fit_values = C_fit(T_fit, A_fit, alpha_fit)

            plt.figure(figsize=(8, 6))
            plt.errorbar(T_all, C_all, yerr=C_err_all, fmt='o', label=f'Simulation Data (L={L})')
            plt.plot(T_fit, C_fit_values, '-', label=f'Fit: α = {alpha_fit:.4f}, Tc = {Tc_fit:.4f}')
            plt.xlabel('Temperature T')
            plt.ylabel('Specific Heat C(T)')
            plt.title(f'Specific Heat vs Temperature for L={L}')
            plt.legend()
            plt.show()

    return fit_results

def fit_binder_crossings(simulation_results, L_list, T_list):
    L_vals = sorted(L_list)  # Sort lattice sizes
    T_crossings = []
    L_pairs = []

    for i in range(len(L_vals) - 1):  # Compare pairs of L
        L1, L2 = L_vals[i], L_vals[i + 1]

        # Extract Binder cumulants for both lattice sizes
        U_L1 = []
        U_L2 = []
        for T in T_list:
            U_L1.append(simulation_results[L1][T]['binder_cumulant'][0])  # Mean Binder cumulant for L1
            U_L2.append(simulation_results[L2][T]['binder_cumulant'][0])  # Mean Binder cumulant for L2

        # Find crossings between Binder cumulants
        crossing_found = False
        for idx in range(len(T_list) - 1):
            if (U_L1[idx] - U_L2[idx]) * (U_L1[idx + 1] - U_L2[idx + 1]) < 0:
                # Linear interpolation for crossing point
                t_cross = T_list[idx] + (T_list[idx + 1] - T_list[idx]) * (
                        (U_L2[idx] - U_L1[idx]) /
                        ((U_L1[idx + 1] - U_L1[idx]) - (U_L2[idx + 1] - U_L2[idx]))
                )
                T_crossings.append(t_cross)
                L_pairs.append((L1, L2))
                crossing_found = True
                break  # Stop after finding one crossing for this pair

        if not crossing_found:
            print(f"No crossing found between L={L1} and L={L2}")

    return T_crossings, L_pairs

def fit_magnetization_per_L(simulation_results, L_list, T_list, Tc=None, plot=True):
    fit_results = {}
    for L in L_list:
        # Collect data for this lattice size
        T_all = []
        m_all = []

        for T in T_list:
            results = simulation_results[L][T]
            m_mean = np.mean(np.abs(results['magnetizations']))  # Magnetization is the average of |M|/V
            T_all.append(T)
            m_all.append(m_mean)

        T_all = np.array(T_all)
        m_all = np.array(m_all)

        # Keep only T < Tc for fitting
        if Tc:
            T_fit = T_all[T_all < Tc]
            m_fit = m_all[T_all < Tc]
        else:
            T_fit = T_all
            m_fit = m_all

        # Define fitting function
        if Tc is None:
            # Fit Tc as a parameter
            def m_fit_func(T, A, Tc_fit, beta):
                return A * (Tc_fit - T) ** (beta)

            p0 = [0.1, 2.2, 0.6]
            bounds = ([0, np.min(T_fit), 0], [np.inf, np.max(T_fit), 2])
        else:
            # Fix Tc
            def m_fit_func(T, A, beta):
                return A * (Tc - T) ** (beta)

            p0 = [0.1, 0.6]
            bounds = ([0, 0], [np.inf, 2])

        # Perform the fit
        try:
            if Tc is None:
                params, cov = curve_fit(
                    m_fit_func, T_fit, m_fit, p0=p0,
                    absolute_sigma=True, bounds=bounds
                )
                A_fit, Tc_fit, beta_fit = params
                errors = np.sqrt(np.diag(cov))
                A_err, Tc_err, beta_err = errors
            else:
                params, cov = curve_fit(
                    m_fit_func, T_fit, m_fit, p0=p0,
                    absolute_sigma=True, bounds=bounds
                )
                A_fit, beta_fit = params
                errors = np.sqrt(np.diag(cov))
                A_err, beta_err = errors
                Tc_fit = Tc
                Tc_err = 0
        except Exception as e:
            print(f"Fit failed for L={L}: {e}")
            fit_results[L] = None
            continue

        # Store results
        fit_results[L] = {
            'A': (A_fit, A_err),
            'Tc': (Tc_fit, Tc_err),
            'beta': (beta_fit, beta_err)
        }

        # Plotting
        if plot:
            T_plot = np.linspace(np.min(T_fit), np.max(T_fit), 500)
            if Tc is None:
                m_fit_values = m_fit_func(T_plot, A_fit, Tc_fit, beta_fit)
            else:
                m_fit_values = m_fit_func(T_plot, A_fit, beta_fit)

            plt.figure(figsize=(8, 6))
            plt.scatter(T_fit, m_fit, label=f'Simulation Data (L={L})')
            plt.plot(T_plot, m_fit_values, '-', label=f'Fit: β = {beta_fit:.4f}, Tc = {Tc_fit:.4f}')
            plt.xlabel('Temperature T')
            plt.ylabel('Magnetization m(T)')
            plt.title(f'Magnetization vs Temperature for L={L}')
            plt.legend()
            plt.show()

    return fit_results

def scaling_fit(L_list):
    # Define the scaling function
    Tc = 2.2

    def scaling_func(L, T_c, k, const):
        return T_c + const * L ** (-k)

    # Perform the curve fit
    popt, pcov = curve_fit(scaling_func, L_list, Tc, p0=[2.2, 0.65, 5])
    Tc, k, _ = popt
    c, kerror, _ = pcov
    nu = 1 / k
    nu_error = 1 / kerror
    return Tc, nu, kerror, c

def scaling_fit2(L_list):
    # Define the scaling function
    Tc = 2.2

    def scaling_func(L, T_c, nu, const):
        return T_c + const * L ** (-1 / nu)

    # Perform the curve fit
    popt, pcov = curve_fit(scaling_func, L_list, Tc, p0=[2.19, 1.41, 7])
    Tc, nu, _ = popt
    _, nu_error, _ = pcov
    return Tc, nu, nu_error






def fit_specific_heat(results_dict, T_list, L_list, nu):
    # Collect maximum specific heat data for each L
    C_max_values = []
    C_max_errors = []
    L_values = []

    for L in L_list:
        C_list = []
        C_errors = []
        T_list_L = []

        # For each T in T_list, get C(L,T)
        for T in T_list:
                C, C_error = results_dict[L][T]['specific_heat']
                C_list.append(C)
                C_errors.append(C_error)
                T_list_L.append(T)

        if C_list:
            # Convert to arrays
            C_array = np.array(C_list)
            T_array = np.array(T_list_L)
            C_error_array = np.array(C_errors)

            # Find maximum C and corresponding T
            max_idx = np.argmax(C_array)
            C_max = C_array[max_idx]
            C_max_error = C_error_array[max_idx]
            T_max = T_array[max_idx]

            C_max_values.append(C_max)
            C_max_errors.append(C_max_error)
            L_values.append(L)
        else:
            print(f"No specific heat data for L={L}")
            continue

    # Convert lists to arrays
    L_values = np.array(L_values)
    C_max_values = np.array(C_max_values)
    C_max_errors = np.array(C_max_errors)

    # Take logarithms of L and C_max
    log_L = np.log(L_values)
    log_C_max = np.log(C_max_values)
    log_C_max_errors = C_max_errors / C_max_values  # Δ(ln C) = ΔC / C

    # Define linear function for fitting
    def linear_fit(x, slope, intercept):
        return -slope * x + intercept

    # Perform linear regression with weighting
    popt, pcov = curve_fit(
        linear_fit,
        log_L,
        log_C_max,
        sigma=log_C_max_errors,
        absolute_sigma=True
    )
    slope = popt[0]
    intercept = popt[1]
    slope_std = np.sqrt(np.diag(pcov))[0]

    alpha_over_nu = slope
    alpha_over_nu_error = slope_std
    alpha = slope*nu

    # Plotting the results
    plt.figure(figsize=(8,6))
    plt.errorbar(log_L, log_C_max, yerr=log_C_max_errors, fmt='o', label='Data with errors')
    plt.plot(log_L, linear_fit(log_L, *popt), 'r-', label=f'Fit: slope={slope:.3f}±{slope_std:.3f}')
    plt.xlabel('ln(L)')
    plt.ylabel('ln(C_max)')
    plt.title('Log-Log Plot of Maximum Specific Heat vs System Size')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prepare fit details
    fit_details = {
        'alpha_over_nu': alpha_over_nu,
        'alpha_over_nu_error': alpha_over_nu_error,
        'slope': slope,
        'intercept': intercept,
        'slope_error': slope_std,
        'covariance_matrix': pcov,
    }

    return alpha, alpha_over_nu, fit_details

def fit_magnetization(results_dict, T_list, L_list, nu):
    # Collect magnetization data for each L
    m_values = []
    L_values = []

    for L in L_list:
        m_list = []
        T_list_L = []

        # For each T in T_list, get m(L,T)
        for T in T_list:
            try:
                # Access the magnetization data from results_dict
                m = results_dict[L][T]['magnetizations']
                # If m is a list or array, compute its mean
                if isinstance(m, (list, np.ndarray)):
                    m = np.mean(np.abs(m))
                else:
                    m = float(np.abs(m))  # Ensure m is a float
                m_list.append(m)
                T_list_L.append(T)
            except KeyError:
                # Data not available for this L and T
                continue

        if m_list:
            # Convert to arrays
            m_array = np.array(m_list)
            T_array = np.array(T_list_L)

            # Find the index of T closest to Tc
            T_c = 2.2018  # Example critical temperature
            idx = np.argmin(np.abs(T_array - T_c))
            m_at_Tc = m_array[idx]
            T_at_Tc = T_array[idx]

            # Check if m_at_Tc is positive
            if m_at_Tc > 0:
                m_values.append(m_at_Tc)
                L_values.append(L)
                # print(f"L={L}, T={T_at_Tc}, m={m_at_Tc}")
            else:
                print(f"Non-positive magnetization at L={L}, T={T_at_Tc}, m={m_at_Tc}")
        else:
            print(f"No magnetization data for L={L}")
            continue

    # Convert lists to arrays
    L_values = np.array(L_values)
    m_values = np.array(m_values)

    # Check that m_values and L_values have the same length
    if len(L_values) != len(m_values):
        print("Mismatch in data lengths after collection.")
        return None, None, None

    # Ensure there are enough data points
    if len(L_values) < 2:
        print("Not enough data points to perform the fit.")
        return None, None, None

    # Take logarithms
    log_L = np.log(L_values)
    log_m = np.log(m_values)

    # Define linear model for fitting
    def linear_model(B, x):
        return B[0] * x + B[1]

    # Prepare data for ODR without uncertainties
    data = RealData(log_L, log_m)
    model = Model(linear_model)
    odr_instance = ODR(data, model, beta0=[-0.5, 0.0])
    output = odr_instance.run()

    # Extract fitting parameters
    slope = output.beta[0]
    intercept = output.beta[1]
    slope_std = output.sd_beta[0]

    beta_over_nu = -slope  # Negative sign due to the scaling relation
    beta_over_nu_error = slope_std
    beta = beta_over_nu * nu

    # Plotting the results
    plt.figure(figsize=(8,6))
    plt.scatter(log_L, log_m, label='Data')
    plt.plot(log_L, linear_model(output.beta, log_L), 'r-', label=f'Fit: slope={slope:.3f}±{slope_std:.3f}')
    plt.xlabel('ln(L)')
    plt.ylabel('ln(m)')
    plt.title('Log-Log Plot of Magnetization vs System Size at $T_c$')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prepare fit details
    fit_details = {
        'beta_over_nu': beta_over_nu,
        'beta_over_nu_error': beta_over_nu_error,
        'slope': slope,
        'intercept': intercept,
        'slope_error': slope_std,
        'covariance_matrix': output.cov_beta,
    }

    return beta, beta_over_nu, fit_details

def fit_susceptibility(results_dict, T_list, L_list, nu):
    # Collect maximum susceptibility data for each L
    chi_max_values = []
    chi_max_errors = []
    L_values = []

    for L in L_list:
        chi_list = []
        chi_errors = []
        T_list_L = []

        # For each T in T_list, get chi(L,T)
        for T in T_list:
            key = (L, T)
            chi, chi_error = results_dict[L][T]['susceptibility']
            chi_list.append(chi)
            chi_errors.append(chi_error)
            T_list_L.append(T)


        if chi_list:
            # Convert to arrays
            chi_array = np.array(chi_list)
            T_array = np.array(T_list_L)
            chi_error_array = np.array(chi_errors)

            # Find maximum chi and corresponding T
            max_idx = np.argmax(chi_array)
            chi_max = chi_array[max_idx]
            chi_max_error = chi_error_array[max_idx]
            T_max = T_array[max_idx]

            chi_max_values.append(chi_max)
            chi_max_errors.append(chi_max_error)
            L_values.append(L)
        else:
            print(f"No susceptibility data for L={L}")
            continue

    # Convert lists to arrays
    L_values = np.array(L_values)
    chi_max_values = np.array(chi_max_values)
    chi_max_errors = np.array(chi_max_errors)

    # Take logarithms
    log_L = np.log(L_values)
    log_chi_max = np.log(chi_max_values)
    log_chi_max_errors = chi_max_errors / chi_max_values  # Δ(ln χ) = Δχ / χ

    # Define linear model for fitting
    def linear_model(B, x):
        return B[0] * x + B[1]

    # Prepare data for Orthogonal Distance Regression (ODR)
    data = RealData(log_L, log_chi_max, sy=log_chi_max_errors)
    model = Model(linear_model)
    odr_instance = ODR(data, model, beta0=[2.0, 0.0])
    output = odr_instance.run()

    # Extract fitting parameters
    slope = output.beta[0]
    intercept = output.beta[1]
    slope_std = output.sd_beta[0]

    gamma_over_nu = slope
    gamma_over_nu_error = slope_std
    gamma = slope*nu

    # Plotting the results
    plt.figure(figsize=(8,6))
    plt.errorbar(log_L, log_chi_max, yerr=log_chi_max_errors, fmt='o', label='Data with errors')
    plt.plot(log_L, linear_model(output.beta, log_L), 'r-', label=f'Fit: slope={slope:.3f}±{slope_std:.3f}')
    plt.xlabel('ln(L)')
    plt.ylabel('ln(χ_max)')
    plt.title('Log-Log Plot of Maximum Susceptibility vs System Size')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prepare fit details
    fit_details = {
        'gamma_over_nu': gamma_over_nu,
        'gamma_over_nu_error': gamma_over_nu_error,
        'slope': slope,
        'intercept': intercept,
        'slope_error': slope_std,
        'covariance_matrix': output.cov_beta,
    }

    return gamma, gamma_over_nu, fit_details

def fit_nu_from_binder_cumulant(results_dict, L_list, T_list, T_critical, delta_T=0.15, initial_guess=(-5, 1.2)):
    log_L = []
    log_dU_dT = []
    log_dU_dT_errors = []
    valid_L = []

    for L in L_list:
        if L not in results_dict:
            print(f"L={L} not found in results_dict. Skipping.")
            continue

        U_list = []
        U_errors_list = []

        # Extract U and U_error for all T at this L within [T_critical - delta_T, T_critical + delta_T]
        for T in T_list:
            if 'binder_cumulant' in results_dict[L][T]:
                U, U_error = results_dict[L][T]['binder_cumulant']
                if T_critical - delta_T <= T <= T_critical + delta_T:
                    U_list.append(U)
                    U_errors_list.append(U_error)

        U_array = np.array(U_list)
        T_array = np.array([T for T in T_list if
                            T_critical - delta_T <= T <= T_critical + delta_T and 'binder_cumulant' in results_dict[L][
                                T]])
        U_error_array = np.array(U_errors_list)

        # Estimate U(L, T_critical) as the median of U within the specified range
        U_median = np.median(U_array)
        U_median_error = np.median(U_error_array)  # Using median as a robust estimator

        # Calculate dU/dT using the central difference method
        # Find the two closest temperatures below and above T_critical
        lower_T = T_array[T_array < T_critical]
        upper_T = T_array[T_array > T_critical]

        T1 = max(lower_T)
        T2 = min(upper_T)

        # Corresponding U and U_error
        idx_T1 = np.where(T_array == T1)[0][0]
        idx_T2 = np.where(T_array == T2)[0][0]

        U1 = U_array[idx_T1]
        U2 = U_array[idx_T2]
        U1_error = U_error_array[idx_T1]
        U2_error = U_error_array[idx_T2]

        # Compute dU/dT and its uncertainty
        dU_dT = (U2 - U1) / (T1 - T2)
        dU_dT_error = np.sqrt(U1_error ** 2 + U2_error ** 2) / (T2 - T1)

        # Take logarithms
        log_L.append(np.log2(L))
        log_dU_dT.append(np.log2(dU_dT))
        log_dU_dT_errors.append(dU_dT_error / dU_dT)  # Δ(log y) = Δy / y
        valid_L.append(L)

    if not log_L:
        raise ValueError("No valid data points found for fitting.")

    log_L = np.array(log_L)
    log_dU_dT = np.array(log_dU_dT)
    log_dU_dT_errors = np.array(log_dU_dT_errors)

    # Define the linear model: log(dU/dT) = A + B * log(L), where B = 1/nu
    def linear_model(params, x):
        return params[0] + params[1] * x  # params[0] = A, params[1] = B = 1/nu

    # Create a Model for ODR
    model = Model(linear_model)

    # Prepare data for ODR
    # Independent variable x is log_L
    # Dependent variable y is log_dU_dT
    # Uncertainties in y are log_dU_dT_errors
    data = RealData(x=log_L, y=log_dU_dT, sx=None, sy=log_dU_dT_errors)

    # Initial parameter estimates: [A_initial, B_initial]
    beta0 = list(initial_guess)

    # Set up ODR
    odr_instance = ODR(data, model, beta0=beta0)

    # Run ODR
    output = odr_instance.run()

    # Extract parameters
    A, B = output.beta
    A_error, B_error = output.sd_beta

    # Calculate nu and its uncertainty

    nu = 1.0 / B
    # Propagate error: Δnu = ΔB / B^2
    nu_error = B_error / (B ** 2)

    # Prepare fit details
    fit_details = {
        'A': A,
        'A_error': A_error,
        'B': B,
        'B_error': B_error,
        'nu': nu,
        'nu_error': nu_error,
        'covariance_matrix': output.cov_beta,
        'odr_output': output,
    }

    # Generate data for the fit line
    log_L_fit = np.linspace(min(log_L) * 0.95, max(log_L) * 1.05, 500)
    log_dU_dT_fit = linear_model(output.beta, log_L_fit)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.errorbar(log_L, log_dU_dT, yerr=log_dU_dT_errors, fmt='o', label='Data with error bars', capsize=5)
    plt.plot(log_L_fit, log_dU_dT_fit, 'r-', label=f'Fit: A={A:.3f}±{A_error:.3f}, B=1/nu={B:.3f}±{B_error:.3f}')
    plt.xlabel(r'$\log(L)$')
    plt.ylabel(r'$\log\left(\frac{dU}{dT}\right)$')
    plt.title('Scaling of $dU/dT$ with System Size $L$ at $T_{\text{critical}}$')
    plt.legend()
    plt.grid(True)
    plt.show()

    return nu, nu_error, fit_details



# Collapse code
def data_collapse_specific_heat(simulation_results, L_list, T_list, Tc, alpha, nu, plot=True):
    collapsed_data = {}
    step = -1

    for L in L_list:
        # Collect data for this lattice size
        T_all = []
        C_all = []
        step += 1
        for T in T_list:
            results = simulation_results[L][T]
            C_mean, _ = results['specific_heat']
            T_all.append(T)
            C_all.append(C_mean)

        T_all = np.array(T_all)
        C_all = np.array(C_all)
        # Rescale data
        rescaled_T = (T_all - Tc) * L ** (1 / nu)
        rescaled_C = C_all / L ** (alpha / nu)

        collapsed_data[L] = (rescaled_T, rescaled_C)

        # Plot collapsed data
        if plot:
            plt.scatter(rescaled_T, rescaled_C, label=f'L={L}', s=10)

    if plot:
        plt.axvline(0, color='red', linestyle='--', label=f'T_c={Tc}')
        plt.xlabel(r'Rescaled Temperature $(T - T_c) L^{1/\nu}$')
        plt.ylabel(r'Rescaled Specific Heat $C(T, L) / L^{\alpha/\nu}$')
        plt.title('Specific Heat Data Collapse')
        plt.legend()
        plt.grid(True)
        plt.show()

    return collapsed_data


def data_collapse_susceptibility(simulation_results, L_list, T_list, Tc, gamma, nu, plot=True):
    collapsed_data = {}
    step = -1
    for L in L_list:
        # Collect susceptibility data for this lattice size
        T_all = []
        chi_all = []
        step += 1
        for T in T_list:
            results = simulation_results[L][T]
            chi_mean, _ = results['susceptibility']  # Extract susceptibility
            T_all.append(T)
            chi_all.append(chi_mean)

        T_all = np.array(T_all)
        chi_all = np.array(chi_all)

        # Rescale data
        rescaled_T = (T_all - Tc) * L ** (1 / nu)
        rescaled_chi = chi_all / L ** (gamma / nu)

        collapsed_data[L] = (rescaled_T, rescaled_chi)

        # Plot collapsed data
        if plot:
            plt.scatter(rescaled_T, rescaled_chi, label=f'L={L}', s=10)

    if plot:
        plt.axvline(0, color='red', linestyle='--', label=f'T_c={Tc}')
        plt.xlabel(r'Rescaled Temperature $(T - T_c) L^{1/\nu}$')
        plt.ylabel(r'Rescaled Susceptibility $\chi(T, L) / L^{\gamma/\nu}$')
        plt.title('Magnetic Susceptibility Data Collapse')
        plt.legend()
        plt.grid(True)
        plt.show()

    return collapsed_data


def data_collapse_magnetization(simulation_results, L_list, T_list, Tc, beta, nu, plot=True):
    collapsed_data = {}
    step = 0
    for L in L_list:
        # Collect magnetization data for this lattice size
        T_all = []
        M_all = []
        for T in T_list:
            results = simulation_results[L][T]
            M_mean = np.mean(np.abs(results['magnetizations']))  # Use absolute magnetization
            T_all.append(T)
            M_all.append(M_mean)

        T_all = np.array(T_all)
        M_all = np.array(M_all)

        # Rescale data
        rescaled_T = (Tc - T_all) * L ** (1 / nu)
        rescaled_M = M_all * L ** (beta / nu)

        collapsed_data[L] = (rescaled_T, rescaled_M)
        step += 1
        # Plot collapsed data
        if plot:
            plt.scatter(rescaled_T, rescaled_M, label=f'L={L}', s=10)

    if plot:
        plt.axvline(0, color='red', linestyle='--', label=f'T_c={Tc}')
        plt.xlabel(r'Rescaled Temperature $(T_c - T) L^{1/\nu}$')
        plt.ylabel(r'Rescaled Magnetization $M(T, L) \cdot L^{\beta/\nu}$')
        plt.title('Magnetization Data Collapse')
        plt.legend()
        plt.grid(True)
        plt.show()

    return collapsed_data
