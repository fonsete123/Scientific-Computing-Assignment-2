"""
Monte Carlo Diffusion-Limited Aggregation (DLA) using random walkers.

Exercise C: Basic Monte Carlo DLA.
Exercise D: Effect of sticking probability on cluster morphology.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

FIGURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)


def mc_dla(N=100, n_particles=500, p_stick=1.0, seed=42, verbose=True):
    """
    Grow a DLA cluster using Monte Carlo random walkers.

    Parameters
    ----------
    N : int
        Grid size (N x N).
    n_particles : int
        Number of particles to attempt to add.
    p_stick : float
        Sticking probability when walker reaches a neighbor of the cluster.
    seed : int
        Random seed.

    Returns
    -------
    cluster : ndarray (N, N)
        Binary mask of the cluster (1 = occupied).
    """
    rng = np.random.default_rng(seed)
    cluster = np.zeros((N, N), dtype=int)

    # Seed at bottom center
    cluster[1, N // 2] = 1

    # Directions: right, left, up, down
    dj = np.array([0, 0, 1, -1])
    di = np.array([1, -1, 0, 0])

    particles_added = 0
    max_walks = n_particles * 500  # safety limit

    for attempt in range(max_walks):
        if particles_added >= n_particles:
            break

        # Release walker from random x on top boundary
        x = rng.integers(0, N)
        y = N - 1   # top row

        # If spawn position is already cluster, skip
        if cluster[y, x] == 1:
            continue

        max_steps = 5 * N * N  # prevent infinite walks
        for _ in range(max_steps):
            # Choose random direction
            direction = rng.integers(0, 4)
            ny = y + dj[direction]
            nx = (x + di[direction]) % N   # periodic in x

            # Check if walker exits top or bottom → remove
            if ny < 0 or ny >= N:
                break

            # Check if destination is cluster → can't enter
            if cluster[ny, nx] == 1:
                # Walker is adjacent to cluster → try to stick
                if rng.random() < p_stick:
                    cluster[y, x] = 1
                    particles_added += 1
                    if verbose and particles_added % 100 == 0:
                        print(f"  Particles added: {particles_added}/{n_particles}")
                    break
                else:
                    # Don't move, stay in place (rejected sticking)
                    continue

            # Move walker
            y, x = ny, nx

            # Check adjacency to cluster at new position
            adjacent = False
            for d in range(4):
                cy = y + dj[d]
                cx = (x + di[d]) % N
                if 0 <= cy < N and cluster[cy, cx] == 1:
                    adjacent = True
                    break

            if adjacent:
                if rng.random() < p_stick:
                    cluster[y, x] = 1
                    particles_added += 1
                    if verbose and particles_added % 100 == 0:
                        print(f"  Particles added: {particles_added}/{n_particles}")
                    break

    if verbose:
        print(f"  Final cluster size: {np.sum(cluster)}")
    return cluster


def exercise_c(N=100, n_particles=300):
    """Exercise C: Basic Monte Carlo DLA."""
    print("=" * 60)
    print("Exercise C: Monte Carlo DLA")
    print("=" * 60)

    t0 = time.time()
    cluster = mc_dla(N=N, n_particles=n_particles, p_stick=1.0)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cluster, origin='lower', cmap='Greys', extent=[0, N, 0, N])
    ax.set_title(f'Monte Carlo DLA (N={N}, {n_particles} particles, p_s=1.0)',
                 fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'exercise_c_mc_dla.png'), dpi=150)
    plt.close()
    print(f"Figure saved: exercise_c_mc_dla.png")


def exercise_d(N=100, n_particles=300):
    """Exercise D: Effect of sticking probability."""
    print("\n" + "=" * 60)
    print("Exercise D: Sticking Probability Effect")
    print("=" * 60)

    p_sticks = [0.1, 0.3, 0.5, 0.7, 1.0]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    for idx, ps in enumerate(p_sticks):
        print(f"\n  Running p_s = {ps}...")
        t0 = time.time()
        cluster = mc_dla(N=N, n_particles=n_particles, p_stick=ps,
                         verbose=False)
        elapsed = time.time() - t0
        cluster_size = np.sum(cluster)
        print(f"    Done in {elapsed:.1f}s, cluster size={cluster_size}")

        ax = axes[idx]
        ax.imshow(cluster, origin='lower', cmap='Greys', extent=[0, N, 0, N])
        ax.set_title(f'p_s = {ps}', fontsize=13)
        ax.set_xlabel('x')
        if idx == 0:
            ax.set_ylabel('y')

    plt.suptitle(f'Monte Carlo DLA with Different Sticking Probabilities '
                 f'(N={N}, {n_particles} particles)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'exercise_d_sticking_prob.png'), dpi=150)
    plt.close()
    print(f"\nFigure saved: exercise_d_sticking_prob.png")


if __name__ == "__main__":
    exercise_c(N=100, n_particles=300)
    exercise_d(N=100, n_particles=300)
