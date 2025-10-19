import numpy as np

def init_lattice(L, seed=None):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2*np.pi, size=(L,L,L))
    spins = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    return spins
