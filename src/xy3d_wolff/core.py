import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData
from scipy.interpolate import UnivariateSpline


def initialize_lattice(L):
    theta = np.random.uniform(0, 2 * np.pi, (L, L, L))
    spins = np.stack((np.cos(theta), np.sin(theta)), axis=-1)  # Shape: (L, L, L, 2)
    return spins


def wolff_update(spins, J, T):
    beta = 1.0 / T
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

    flipped = np.zeros((L, L, L), dtype=bool)
    flipped[i0, j0, k0] = True

    while stack:
        i, j, k = stack.pop()
        S_i = spins[i, j, k]

        # Neighbor offsets in 3D
        for di, dj, dk in [(-1, 0, 0), (1, 0, 0),
                           (0, -1, 0), (0, 1, 0),
                           (0, 0, -1), (0, 0, 1)]:
            ni, nj, nk = (i + di) % L, (j + dj) % L, (k + dk) % L

            if not flipped[ni, nj, nk]:
                S_j = spins[ni, nj, nk]
                # Corrected delta with negative sign
                delta = -2 * beta * J * np.dot(S_i, r) * np.dot(S_j, r)
                p_add = 1 - np.exp(min(0, delta))

                if np.random.rand() < p_add:
                    # Reflect spin S_j
                    S_j_new = S_j - 2 * np.dot(S_j, r) * r
                    spins[ni, nj, nk] = S_j_new
                    flipped[ni, nj, nk] = True
                    cluster.add((ni, nj, nk))
                    stack.append((ni, nj, nk))

    return len(cluster)


def compute_energy(spins, J):
    L = spins.shape[0]
    energy = 0.0

    # Interactions in x-direction
    energy += np.sum(spins * np.roll(spins, shift=-1, axis=0))

    # Interactions in y-direction
    energy += np.sum(spins * np.roll(spins, shift=-1, axis=1))

    # Interactions in z-direction
    energy += np.sum(spins * np.roll(spins, shift=-1, axis=2))

    return -J * energy


def compute_magnetization(spins):
    L = spins.shape[0]
    total_spin = np.sum(spins, axis=(0, 1, 2))  # Sum over all spins
    magnetization = np.linalg.norm(total_spin)  # Magnitude of total spin
    return magnetization / (L ** 3), total_spin  # Return average magnetization per spin and total vector


def compute_susceptibility(magnetizations, T, V):
    mag_sq = np.mean(np.array(magnetizations) ** 2)
    susceptibility = V / T * mag_sq
    return susceptibility


def compute_specific_heat(energies, T, V):
    energy_sq = np.mean(np.array(energies) ** 2)
    mean_energy = np.mean(energies)
    specific_heat = (energy_sq - mean_energy ** 2) / (T ** 2) / V
    return specific_heat


def compute_binder_cumulant(magnetizations, V):
    mag_sq = np.mean(np.array(magnetizations) ** 2)
    mag_fourth = np.mean(np.array(magnetizations) ** 4)
    binder_cumulant = 1 - mag_fourth / (3 * mag_sq ** 2)
    return binder_cumulant


def run_simulation(L, J, T, n_steps, n_equil):
    spins = initialize_lattice(L)
    energies = []
    magnetizations = []

    for step in range(n_steps):
        cluster_size = wolff_update(spins, J, T)
        # print(f"Step {step}, cluster size: {cluster_size}")

        energy = compute_energy(spins, J)
        mag, mag_vector = compute_magnetization(spins)

        if step >= n_equil:
            energies.append(energy)
            magnetizations.append(mag)

    # Calculate observables
    V = L ** 3
    energy_density = np.mean(energies) / V
    energy_error = np.std(energies) / (np.sqrt(len(energies)) * V)

    avg_magnetization = np.mean(magnetizations)
    magnetization_error = np.std(magnetizations) / np.sqrt(len(magnetizations))

    susceptibility = compute_susceptibility(magnetizations, T, V)
    susceptibility_error = np.std(np.array(magnetizations) ** 2) / (np.sqrt(len(magnetizations)) * T)

    specific_heat = compute_specific_heat(energies, T, V)
    specific_heat_error = np.std(np.array(energies) ** 2) / (np.sqrt(len(energies)) * T ** 2 * V)

    binder_cumulant = compute_binder_cumulant(magnetizations, V)
    binder_cumulant_error = np.std(
        1 - np.array(magnetizations) ** 4 / (3 * (np.array(magnetizations) ** 2) ** 2)) / np.sqrt(len(magnetizations))

    results = {
        'energy_density': (energy_density, energy_error),
        'magnetization': (avg_magnetization, magnetization_error),
        'susceptibility': (susceptibility, susceptibility_error),
        'specific_heat': (specific_heat, specific_heat_error),
        'binder_cumulant': (binder_cumulant, binder_cumulant_error),
        'energies': energies,
        'magnetizations': magnetizations
    }

    return results


def plot_spin_orientations(spins):
    L = spins.shape[0]
    X, Y, Z = np.indices((L, L, L))
    U = spins[..., 0]  # x-component
    V = spins[..., 1]  # y-component
    W = np.zeros_like(U)  # z-component is 0 since spins are in x-y plane

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, length=0.5, normalize=True)
    ax.set_title("3D XY Model Spins")
    plt.show()


# L = 8
# spins = initialize_lattice(L)
# plot_spin_orientations(spins)

def simulate_all_data(L_list, T_list, J, n_steps, n_equil):
    all_data = {}
    print(T_list)
    for L in L_list:
        all_data[L] = {}
        V = L ** 3

        for T in T_list:
            all_data[L][T] = run_simulation(L, J, T, n_steps, n_equil)
            U, error_U = all_data[L][T]['binder_cumulant']
            chi, error_chi = all_data[L][T]['susceptibility']
            C, error_C = all_data[L][T]['specific_heat']
            # xi, error_xi = all_data[L][T]['correlation_length']
            # xi = all_data[L][T]['correlation_length']

            print(f"L={L}, T={T:.2f}, Binder Cumulant U={U:.4f}, "
                  f"Susceptibility={chi:.4f}, Specific Heat={C:.4f}")  # Correlation Length={xi:.4f}")

    return all_data


def run_improved_simulation(L, J, T, n_steps, n_equil):
    spins = initialize_lattice(L)
    energies = []
    magnetizations = []

    chi_estimators = []
    C_estimators = []

    for step in range(n_steps):
        cluster_size, chi_estimator, C_estimator = wolff_update_with_estimator(spins, J, T)

        energy = compute_energy(spins, J)
        mag, mag_vector = compute_magnetization(spins)

        if step >= n_equil:
            energies.append(energy)
            magnetizations.append(mag)
            chi_estimators.append(chi_estimator)
            C_estimators.append(C_estimator)

    # Calculate observables
    V = L ** 3
    energy_density = np.mean(energies) / V
    energy_error = np.std(energies) / (np.sqrt(len(energies)) * V)

    avg_magnetization = np.mean(magnetizations)
    magnetization_error = np.std(magnetizations) / np.sqrt(len(magnetizations))

    # Improved susceptibility calculation using cluster sizes
    chi = np.mean(chi_estimators)
    chi_error = np.std(chi_estimators) / np.sqrt(len(chi_estimators))

    # Improved specific heat calculation using cluster sizes
    C = np.mean(C_estimators)
    C_error = np.std(C_estimators) / np.sqrt(len(C_estimators))

    binder_cumulant = compute_binder_cumulant(magnetizations, V)
    binder_cumulant_error = np.std(
        1 - np.array(magnetizations) ** 4 / (3 * (np.array(magnetizations) ** 2) ** 2)) / np.sqrt(len(magnetizations))

    results = {
        'energy_density': (energy_density, energy_error),
        'magnetization': (avg_magnetization, magnetization_error),
        'susceptibility': (chi, chi_error),
        'specific_heat': (C, C_error),
        'binder_cumulant': (binder_cumulant, binder_cumulant_error),
        'energies': energies,
        'magnetizations': magnetizations
    }

    return results


def improved_simulate_all_data(L_list, T_list, J, n_steps, n_equil):
    all_data = {}
    for L in L_list:
        all_data[L] = {}
        V = L ** 3

        for T in T_list:
            all_data[L][T] = run_improved_simulation(L, J, T, n_steps, n_equil)
            U, error_U = all_data[L][T]['binder_cumulant']
            chi, error_chi = all_data[L][T]['susceptibility']
            C, error_C = all_data[L][T]['specific_heat']
            # xi, error_xi = all_data[L][T]['correlation_length']

            print(f"L={L}, T={T:.2f}, Binder Cumulant U={U:.4f}, "
                  f"Susceptibility={chi:.4f}, Specific Heat={C:.4f}")
            #   f"Correlation Length={xi:.4f}")

    return all_data


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

        plt.errorbar(temperatures, avg_magnetizations,
                     yerr=np.std(simulation_results[L][T_list[0]]['magnetizations']) / np.sqrt(
                         len(simulation_results[L][T_list[0]]['magnetizations'])),
                     label=f'L={L}', capsize=5, marker='o')

    plt.axvline(T_c, color='r', linestyle='--', label=f'T_c ≈ {T_c}')
    plt.xlabel('Temperature T')
    plt.ylabel('Magnetization M')
    plt.title('Magnetization vs Temperature')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Energy Density
    plt.figure()
    for L in L_list:
        temperatures = []
        energy_densities = []
        energy_errors = []

        for T in T_list:
            energy_density, energy_error = simulation_results[L][T]['energy_density']
            temperatures.append(T)
            energy_densities.append(energy_density)
            energy_errors.append(energy_error)

        plt.errorbar(temperatures, energy_densities, yerr=energy_errors, label=f'L={L}', capsize=5, marker='o')

    plt.axvline(T_c, color='r', linestyle='--', label=f'T_c ≈ {T_c}')
    plt.xlabel('Temperature T')
    plt.ylabel('Energy Density E/V')
    plt.title('Energy Density vs Temperature')
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
            chi, chi_error = simulation_results[L][T]['susceptibility']
            temperatures.append(T)
            susceptibilities.append(chi)
            susceptibility_errors.append(chi_error)

        plt.errorbar(temperatures, susceptibilities, yerr=susceptibility_errors, label=f'L={L}', capsize=5, marker='o')

    plt.axvline(T_c, color='r', linestyle='--', label=f'T_c ≈ {T_c}')
    plt.xlabel('Temperature T')
    plt.ylabel('Susceptibility χ')
    plt.title('Susceptibility vs Temperature')
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
            C, C_error = simulation_results[L][T]['specific_heat']
            temperatures.append(T)
            specific_heats.append(C)
            specific_heat_errors.append(C_error)

        plt.errorbar(temperatures, specific_heats, yerr=specific_heat_errors, label=f'L={L}', capsize=5, marker='o')

    plt.axvline(T_c, color='r', linestyle='--', label=f'T_c ≈ {T_c}')
    plt.xlabel('Temperature T')
    plt.ylabel('Specific Heat C')
    plt.title('Specific Heat vs Temperature')
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
            U, U_error = simulation_results[L][T]['binder_cumulant']
            temperatures.append(T)
            binder_cumulants.append(U)
            binder_cumulant_errors.append(U_error)

        plt.errorbar(temperatures, binder_cumulants, yerr=binder_cumulant_errors, label=f'L={L}', capsize=5, marker='o')

    plt.axvline(T_c, color='r', linestyle='--', label=f'T_c ≈ {T_c}')
    plt.xlabel('Temperature T')
    plt.ylabel('Binder Cumulant U')
    plt.title('Binder Cumulant vs Temperature')
    plt.legend()
    plt.grid()
    plt.show()


# Parameters
J = 1.0
n_steps = 10000
n_equil = 3000
L_list = [6, 8, 10, 12]

T_list = np.linspace(1.5, 3.0, 20)  # Temperatures around expected T_c ~2.2
print(T_list)
# Run the simulations and store results
simulation_results = simulate_all_data(L_list, T_list, J, n_steps, n_equil)

# test_result
plot_simulation_results(simulation_results, L_list, T_list)


# Here for each L and T, we have the full Monte Carlo time series of energy and magnetization.

def scaling_function(x, a, b, c):
    return a * x ** 2 + b * x + c


# Define the function C_fit for fitting specific heat
def C_fit(L, T, a, alpha, nu, Tc):
    return a * (L ** (alpha / nu)) * scaling_function((T - Tc) * L ** (1 / nu), 1, 1, 1)


# Extract specific heat data at Tc for multiple system sizes
def fit_specific_heat_at_Tc(simulation_results, L_list, T_list, Tc_initial_guess):
    L_values = []
    C_values = []
    C_errors = []

    # Find the closest temperature to Tc_initial_guess for each L
    for L in L_list:
        T_closest = min(T_list, key=lambda T: abs(T - Tc_initial_guess))
        results = simulation_results[L][T_closest]
        C, C_error = results['specific_heat']

        L_values.append(L)
        C_values.append(C)
        C_errors.append(C_error)

    # Convert to numpy arrays
    L_values = np.array(L_values)
    C_values = np.array(C_values)
    C_errors = np.array(C_errors)

    # Initial guess for a, alpha, and nu
    initial_guess = [1.0, 0.0, 1.0]

    # Use curve_fit to fit C(L, Tc) = a * L^{alpha / nu}
    def fit_function(L, a, alpha_over_nu):
        return a * (L ** alpha_over_nu)

    popt, pcov = curve_fit(fit_function, L_values, C_values, sigma=C_errors, p0=[1.0, 0.1], absolute_sigma=True)

    a_fit, alpha_over_nu_fit = popt
    alpha_over_nu_error = np.sqrt(np.diag(pcov))[1]

    print(f"Fitted a: {a_fit:.4f}")
    print(f"Fitted alpha/nu: {alpha_over_nu_fit:.4f} ± {alpha_over_nu_error:.4f}")

    # Plot the fit
    plt.figure()
    plt.errorbar(L_values, C_values, yerr=C_errors, fmt='o', label='Data')
    L_fit = np.linspace(min(L_values), max(L_values), 100)
    C_fit_values = fit_function(L_fit, a_fit, alpha_over_nu_fit)
    plt.plot(L_fit, C_fit_values, label=f'Fit: C ~ L^(alpha/nu), alpha/nu = {alpha_over_nu_fit:.4f}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('L')
    plt.ylabel('C(L, Tc)')
    plt.title('Finite-Size Scaling of Specific Heat at Tc')
    plt.legend()
    plt.grid()
    plt.show()

    return a_fit, alpha_over_nu_fit, alpha_over_nu_error


# Example usage
Tc_guess = 2.2  # Initial guess for Tc
L_list_fit = [6, 8, 10, 12]  # Use the same list of system sizes as before

a_fit, alpha_over_nu_fit, alpha_over_nu_error = fit_specific_heat_at_Tc(
    simulation_results, L_list_fit, T_list, Tc_guess
)


def chi_fit(L, T, b, gamma, nu, Tc):
    return b * (L ** (gamma / nu)) * scaling_function((T - Tc) * L ** (1 / nu), 1, 1, 1)


def fit_susceptibility_at_Tc(simulation_results, L_list, T_list, Tc_initial_guess):
    L_values = []
    chi_values = []
    chi_errors = []

    for L in L_list:
        T_closest = min(T_list, key=lambda T: abs(T - Tc_initial_guess))
        results = simulation_results[L][T_closest]
        chi, chi_error = results['susceptibility']

        L_values.append(L)
        chi_values.append(chi)
        chi_errors.append(chi_error)

    L_values = np.array(L_values)
    chi_values = np.array(chi_values)
    chi_errors = np.array(chi_errors)

    def fit_function(L, b, gamma_over_nu):
        return b * (L ** gamma_over_nu)

    popt, pcov = curve_fit(fit_function, L_values, chi_values, sigma=chi_errors, p0=[1.0, 1.0], absolute_sigma=True)

    b_fit, gamma_over_nu_fit = popt
    gamma_over_nu_error = np.sqrt(np.diag(pcov))[1]

    print(f"Fitted b: {b_fit:.4f}")
    print(f"Fitted gamma/nu: {gamma_over_nu_fit:.4f} ± {gamma_over_nu_error:.4f}")

    plt.figure()
    plt.errorbar(L_values, chi_values, yerr=chi_errors, fmt='o', label='Data')
    L_fit = np.linspace(min(L_values), max(L_values), 100)
    chi_fit_values = fit_function(L_fit, b_fit, gamma_over_nu_fit)
    plt.plot(L_fit, chi_fit_values, label=f'Fit: χ ~ L^(γ/ν), γ/ν = {gamma_over_nu_fit:.4f}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('L')
    plt.ylabel('χ(L, Tc)')
    plt.title('Finite-Size Scaling of Susceptibility at Tc')
    plt.legend()
    plt.grid()
    plt.show()

    return b_fit, gamma_over_nu_fit, gamma_over_nu_error


b_fit, gamma_over_nu_fit, gamma_over_nu_error = fit_susceptibility_at_Tc(
    simulation_results, L_list_fit, T_list, Tc_guess
)


def M_fit(L, T, c, beta, nu, Tc):
    return c * (L ** (-beta / nu)) * scaling_function((T - Tc) * L ** (1 / nu), 1, 1, 1)


def fit_magnetization_at_Tc(simulation_results, L_list, T_list, Tc_initial_guess):
    L_values = []
    M_values = []
    M_errors = []

    for L in L_list:
        T_closest = min(T_list, key=lambda T: abs(T - Tc_initial_guess))
        results = simulation_results[L][T_closest]
        magnetizations = results['magnetizations']
        M_mean = np.mean(np.abs(magnetizations))
        M_error = np.std(np.abs(magnetizations)) / np.sqrt(len(magnetizations))

        L_values.append(L)
        M_values.append(M_mean)
        M_errors.append(M_error)

    L_values = np.array(L_values)
    M_values = np.array(M_values)
    M_errors = np.array(M_errors)

    def fit_function(L, c, minus_beta_over_nu):
        return c * (L ** minus_beta_over_nu)

    popt, pcov = curve_fit(fit_function, L_values, M_values, sigma=M_errors, p0=[1.0, -0.1], absolute_sigma=True)

    c_fit, minus_beta_over_nu_fit = popt
    minus_beta_over_nu_error = np.sqrt(np.diag(pcov))[1]

    print(f"Fitted c: {c_fit:.4f}")
    print(f"Fitted -beta/nu: {minus_beta_over_nu_fit:.4f} ± {minus_beta_over_nu_error:.4f}")

    plt.figure()
    plt.errorbar(L_values, M_values, yerr=M_errors, fmt='o', label='Data')
    L_fit = np.linspace(min(L_values), max(L_values), 100)
    M_fit_values = fit_function(L_fit, c_fit, minus_beta_over_nu_fit)
    plt.plot(L_fit, M_fit_values, label=f'Fit: M ~ L^(-β/ν), -β/ν = {minus_beta_over_nu_fit:.4f}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('L')
    plt.ylabel('M(L, Tc)')
    plt.title('Finite-Size Scaling of Magnetization at Tc')
    plt.legend()
    plt.grid()
    plt.show()

    return c_fit, minus_beta_over_nu_fit, minus_beta_over_nu_error


c_fit, minus_beta_over_nu_fit, minus_beta_over_nu_error = fit_magnetization_at_Tc(
    simulation_results, L_list_fit, T_list, Tc_guess
)


def data_collapse_specific_heat(simulation_results, L_list, T_list, Tc, alpha, nu, plot=True):
    collapsed_data = {}
    step = 0
    for L in L_list:
        T_scaled = []
        C_scaled = []
        for T in T_list:
            results = simulation_results[L][T]
            C, _ = results['specific_heat']
            T_scaled.append((T - Tc) * L ** (1 / nu))  # x-axis: scaled temperature
            C_scaled.append(C / (L ** (alpha / nu)))  # y-axis: scaled specific heat

        collapsed_data[L] = {
            'T_scaled': np.array(T_scaled),
            'C_scaled': np.array(C_scaled)
        }
        if plot:
            plt.figure(4)
            plt.scatter(T_scaled, C_scaled, marker='o', c='r', s=20)
            # Plot collapsed data
            plt.xlabel(r'$(T - T_c) L^{1/\nu}$')
            plt.ylabel(r'$C(L, T)/L^{\alpha/\nu}$')
            plt.title('Data Collapse of Specific Heat')
            plt.grid(False)
        # plt.savefig(str(step)+'.png')
        step += 1

    plt.show()
    return collapsed_data


def data_collapse_susceptibility(simulation_results, L_list, T_list, Tc, gamma, nu, plot=True):
    collapsed_data = {}
    step = 0
    for L in L_list:
        # Collect susceptibility data for this lattice size
        T_all = []
        chi_all = []
        for T in T_list:
            results = simulation_results[L][T]
            chi_mean, _ = results['susceptibility']  # Extract mean susceptibility
            T_all.append(T)
            chi_all.append(chi_mean)

        T_all = np.array(T_all, dtype=float)
        chi_all = np.array(chi_all, dtype=float)

        # Scale the data to match scaling form: χ ~ L^(γ/ν) * f[(T-Tc)*L^(1/ν)]
        T_scaled = (T_all - Tc) * (L ** (1 / nu))
        chi_scaled = chi_all / (L ** (gamma / nu))

        collapsed_data[L] = {
            'T_scaled': T_scaled,
            'chi_scaled': chi_scaled,
        }
        if plot:
            # Optionally plot for verification
            plt.figure(5)
            plt.scatter(T_scaled, chi_scaled, label=f'L={L}')
            # plt.legend()
            plt.xlabel(r'$(T - T_c) L^{1/\nu}$')
            plt.ylabel(r'$\chi(L, T)/L^{\gamma/\nu}$')
            plt.title('Data Collapse of Susceptibility')
            plt.grid(False)
        step += 1
        for T in T_list:
            results = simulation_results[L][T]
            chi_mean, _ = results['susceptibility']  # Extract mean susceptibility
            T_all.append(T)
            chi_all.append(chi_mean)

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

        T_all = np.array(T_all, dtype=float)
        M_all = np.array(M_all, dtype=float)

        # Scale the data according to the scaling form:
        # M(L, T) ~ L^(-β/ν) * f[(T - Tc)L^(1/ν)]
        T_scaled = (T_all - Tc) * (L ** (1 / nu))
        M_scaled = M_all * (L ** (beta / nu))

        collapsed_data[L] = {
            'T_scaled': T_scaled,
            'M_scaled': M_scaled,
        }

        if plot:
            # Plot for verification
            plt.figure(6)
            plt.scatter(T_scaled, M_scaled, label=f'L={L}')
            plt.xlabel(r'$(T - T_c) L^{1/\nu}$')
            plt.ylabel(r'$M(L, T) L^{\beta/\nu}$')
            plt.title('Data Collapse of Magnetization')
            plt.grid(False)

        step += 1

    if plot:
        plt.show()

    return collapsed_data


def wolff_update_with_estimator(spins, J, T):
    beta = 1.0 / T
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

    # Track the total cluster size for estimators
    cluster_size = 1

    while cluster:
        i, j, k = cluster.pop()
        S_i = spins[i, j, k]

        # Neighbor offsets in 3D
        for di, dj, dk in [(-1, 0, 0), (1, 0, 0),
                           (0, -1, 0), (0, 1, 0),
                           (0, 0, -1), (0, 0, 1)]:
            ni, nj, nk = (i + di) % L, (j + dj) % L, (k + dk) % L

            if not flipped[ni, nj, nk]:
                S_j = spins[ni, nj, nk]
                # Calculate bond probability using reflected part
                delta = -2 * beta * J * np.dot(S_i, r) * np.dot(S_j, r)
                p_add = 1 - np.exp(min(0, delta))

                if np.random.rand() < p_add:
                    # Reflect neighboring spin
                    S_j_new = S_j - 2 * np.dot(S_j, r) * r
                    spins[ni, nj, nk] = S_j_new
                    flipped[ni, nj, nk] = True
                    cluster.append((ni, nj, nk))
                    cluster_size += 1

    # Total system volume
    V = L ** 3

    # Improved estimators:
    # Susceptibility estimator scales with cluster size squared
    chi_estimator = (cluster_size ** 2) / V

    # Specific heat estimator (using a naive scaling with cluster size)
    C_estimator = cluster_size / V

    return cluster_size, chi_estimator, C_estimator


# Parameters
J = 1.0
n_steps = 10000
n_equil = 5000
L_list = [16]

T_list = np.linspace(2.18, 2.25, 10)  # Temperatures around expected T_c ~2.2
print(T_list)
# Run the simulations and store results
simulation_results_with_estimator = improved_simulate_all_data(L_list, T_list, J, n_steps, n_equil)
