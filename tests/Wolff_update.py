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

    # Keep track of flipped spins
    flipped = np.zeros(spins.shape[:3], dtype=bool)
    flipped[i0, j0, k0] = True

    while stack:
        i, j, k = stack.pop()
        S_i = spins[i, j, k]

        # Neighbor indices with periodic boundary conditions
        neighbors = [
            ((i + 1) % L, j, k),
            ((i - 1) % L, j, k),
            (i, (j + 1) % L, k),
            (i, (j - 1) % L, k),
            (i, j, (k + 1) % L),
            (i, j, (k - 1) % L)
        ]

        for ni, nj, nk in neighbors:
            if not flipped[ni, nj, nk]:
                S_j = spins[ni, nj, nk]
                delta = 2 * beta * J * np.dot(S_i, r) * np.dot(S_j, r)
                p_add = 1 - np.exp(min(0, delta))

                if np.random.rand() < p_add:
                    # Reflect spin S_j
                    S_j_new = S_j - 2 * np.dot(S_j, r) * r
                    spins[ni, nj, nk] = S_j_new
                    flipped[ni, nj, nk] = True
                    cluster.add((ni, nj, nk))
                    stack.append((ni, nj, nk))

    return len(cluster)
