from collections import deque
from typing import Optional
import numpy as np

def wolff_update_xy(
    spins: np.ndarray,
    J: float,
    T: float,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    One Wolff cluster update for the 3D XY model on an LxLxL lattice.

    Parameters
    ----------
    spins : np.ndarray
        Shape (L, L, L, 2). Each spin is a 2D unit vector (cos θ, sin θ).
    J : float
        Ferromagnetic coupling (>0).
    T : float
        Temperature (k_B = 1).
    rng : np.random.Generator, optional
        Numpy RNG for reproducibility. If None, uses default_rng().

    Returns
    -------
    int
        Size of the flipped cluster.
    """
    if rng is None:
        rng = np.random.default_rng()

    if T <= 0:
        raise ValueError("Temperature T must be > 0.")
    if spins.ndim != 4 or spins.shape[-1] != 2:
        raise ValueError("spins must have shape (L, L, L, 2) for XY spins.")

    beta = 1.0 / T
    L = spins.shape[0]

    # 1) Choose a random reflection axis r (unit vector in the plane)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    r = np.array([np.cos(phi), np.sin(phi)], dtype=spins.dtype)

    # Precompute fixed projections from the ORIGINAL configuration
    # proj[i,j,k] = s_{ijk} · r
    proj = spins[..., 0] * r[0] + spins[..., 1] * r[1]

    # 2) Random seed site
    i0, j0, k0 = rng.integers(0, L, size=3)

    # 3) Grow cluster using fixed projections (embedded Ising)
    in_cluster = np.zeros(spins.shape[:3], dtype=bool)
    in_cluster[i0, j0, k0] = True
    stack = deque([(i0, j0, k0)])

    # Helper for periodic neighbors
    def neighbors(i, j, k):
        return (
            ((i + 1) % L, j, k),
            ((i - 1) % L, j, k),
            (i, (j + 1) % L, k),
            (i, (j - 1) % L, k),
            (i, j, (k + 1) % L),
            (i, j, (k - 1) % L),
        )

    # Bond-add probability uses POSITIVE projection product only
    # p_add = 1 - exp(-2 * beta * J * max(0, (s_i·r)(s_j·r)))
    # Efficiently: only try to add when proj_i * proj_j > 0.
    two_beta_J = 2.0 * beta * J

    while stack:
        i, j, k = stack.pop()
        proj_i = proj[i, j, k]

        for ni, nj, nk in neighbors(i, j, k):
            if in_cluster[ni, nj, nk]:
                continue

            prod = proj_i * proj[ni, nj, nk]
            if prod <= 0.0:
                continue  # cannot bond if projections have opposite signs or zero

            p_add = 1.0 - np.exp(-two_beta_J * prod)
            if rng.random() < p_add:
                in_cluster[ni, nj, nk] = True
                stack.append((ni, nj, nk))

    # 4) Reflect the entire cluster at once: s -> s - 2 (s·r) r
    # Vectorized reflection for all cluster spins
    idx = np.where(in_cluster)
    # current projections for those sites (still original spins)
    proj_cluster = proj[idx]  # shape (Nc,)
    # subtract 2*(proj)*r from the 2D vectors
    spins[idx + (slice(None),)] -= (2.0 * proj_cluster)[:, None] * r[None, :]

    return int(in_cluster.sum())
