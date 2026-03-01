"""
Diffusion-Limited Aggregation (DLA) using the diffusion equation.

Exercise A: DLA growth with different eta values.
Exercise B: SOR optimization — naive vs vectorized timing comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

from sor_solver import initialize_concentration, sor_solve_vectorized, sor_solve_naive

FIGURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)


def get_growth_candidates(object_mask):
    """Find free sites that are 4-connected neighbors of the object."""
    N = object_mask.shape[0]
    candidates = []
    for j in range(1, N - 1):  # avoid top/bottom boundaries
        for i in range(N):
            if object_mask[j, i] == 1:
                continue
            # Check 4 neighbors (periodic in x)
            neighbors = [
                (j, (i + 1) % N),
                (j, (i - 1) % N),
                (j + 1, i),
                (j - 1, i),
            ]
            for nj, ni in neighbors:
                if 0 <= nj < N and object_mask[nj, ni] == 1:
                    candidates.append((j, i))
                    break
    return candidates


def get_growth_candidates_vectorized(object_mask):
    """Vectorized version of growth candidate finding."""
    N = object_mask.shape[0]
    free = (object_mask == 0)
    # Check if any neighbor is object
    has_obj_neighbor = (
        np.roll(object_mask, 1, axis=1) |  # left neighbor is object
        np.roll(object_mask, -1, axis=1) |  # right neighbor is object
        np.roll(object_mask, 1, axis=0) |   # down neighbor is object
        np.roll(object_mask, -1, axis=0)    # up neighbor is object
    ).astype(bool)
    # Interior only (not boundaries)
    interior = np.zeros((N, N), dtype=bool)
    interior[1:N-1, :] = True
    candidate_mask = free & has_obj_neighbor & interior
    coords = np.argwhere(candidate_mask)
    return [(j, i) for j, i in coords]


def dla_growth(N=100, eta=1.0, n_steps=200, omega=1.8, solver='vectorized',
               tol=1e-5, verbose=True):
    """
    Grow a DLA cluster using the diffusion equation method.

    Parameters
    ----------
    N : int
        Grid size.
    eta : float
        Growth exponent. eta=0 → Eden, eta=1 → DLA, eta>1 → thin branches.
    n_steps : int
        Number of growth steps.
    omega : float
        SOR relaxation parameter.
    solver : str
        'vectorized' or 'naive'.
    tol : float
        SOR convergence tolerance.

    Returns
    -------
    object_mask : ndarray
        Final cluster mask.
    c : ndarray
        Final concentration field.
    """
    solve = sor_solve_vectorized if solver == 'vectorized' else sor_solve_naive

    object_mask = np.zeros((N, N), dtype=int)
    # Seed at bottom center
    object_mask[1, N // 2] = 1

    c = initialize_concentration(N, object_mask)
    c, _ = solve(c, object_mask, omega=omega, tol=tol)

    rng = np.random.default_rng(42)

    for step in range(n_steps):
        candidates = get_growth_candidates_vectorized(object_mask)
        if len(candidates) == 0:
            if verbose:
                print(f"No candidates at step {step}")
            break

        # Compute growth probabilities
        concentrations = np.array([c[j, i] for j, i in candidates])
        concentrations = np.maximum(concentrations, 0)  # clip negatives

        if eta == 0:
            # Uniform growth probability
            probs = np.ones(len(candidates))
        else:
            probs = concentrations ** eta

        total = probs.sum()
        if total == 0:
            # Fallback to uniform if all zero
            probs = np.ones(len(candidates))
            total = probs.sum()
        probs /= total

        # Select growth site
        idx = rng.choice(len(candidates), p=probs)
        gj, gi = candidates[idx]
        object_mask[gj, gi] = 1

        # Re-solve with warm start
        c[gj, gi] = 0.0
        c, iters = solve(c, object_mask, omega=omega, tol=tol)

        if verbose and (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{n_steps}, cluster size={np.sum(object_mask)}, "
                  f"SOR iters={iters}")

    return object_mask, c


def exercise_a(N=100, n_steps=200):
    """Exercise A: DLA with different eta values."""
    print("=" * 60)
    print("Exercise A: DLA with Diffusion Equation")
    print("=" * 60)

    etas = [0, 0.5, 1, 2, 3]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, eta in enumerate(etas):
        print(f"\nRunning eta={eta}...")
        t0 = time.time()
        mask, c = dla_growth(N=N, eta=eta, n_steps=n_steps, omega=1.8)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s, cluster size={np.sum(mask)}")

        ax = axes[idx]
        # Plot concentration field
        ax.imshow(c, origin='lower', cmap='hot', extent=[0, N, 0, N])
        # Overlay cluster
        cluster_vis = np.ma.masked_where(mask == 0, mask)
        ax.imshow(cluster_vis, origin='lower', cmap='Blues', alpha=0.8,
                  extent=[0, N, 0, N])
        ax.set_title(f'η = {eta}', fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Remove extra subplot
    axes[-1].axis('off')

    plt.suptitle(f'DLA Clusters (Diffusion Method, N={N}, {n_steps} steps)',
                 fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'exercise_a_dla_clusters.png'), dpi=150)
    plt.close()
    print(f"\nFigure saved: exercise_a_dla_clusters.png")

    # Also plot just the clusters without concentration field
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, eta in enumerate(etas):
        print(f"\nRunning eta={eta} (cluster only plot)...")
        mask, c = dla_growth(N=N, eta=eta, n_steps=n_steps, omega=1.8,
                             verbose=False)
        ax = axes[idx]
        ax.imshow(mask, origin='lower', cmap='Greys', extent=[0, N, 0, N])
        ax.set_title(f'η = {eta}', fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    axes[-1].axis('off')
    plt.suptitle(f'DLA Cluster Shapes (N={N}, {n_steps} steps)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'exercise_a_cluster_shapes.png'), dpi=150)
    plt.close()
    print(f"Figure saved: exercise_a_cluster_shapes.png")


def exercise_a_omega_test(N=100, n_steps=50):
    """Test different omega values for SOR convergence with DLA."""
    print("\n" + "=" * 60)
    print("Exercise A: Omega Optimization Test")
    print("=" * 60)

    omegas = [1.0, 1.2, 1.4, 1.6, 1.7, 1.8, 1.85, 1.9, 1.95]
    times = []

    for omega in omegas:
        print(f"\n  omega={omega:.2f}...")
        t0 = time.time()
        mask, c = dla_growth(N=N, eta=1.0, n_steps=n_steps, omega=omega,
                             verbose=False)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"    Time: {elapsed:.2f}s")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(omegas, times, 'bo-')
    ax.set_xlabel('ω (SOR parameter)')
    ax.set_ylabel('Total time (s)')
    ax.set_title(f'SOR Performance vs ω (N={N}, {n_steps} growth steps, η=1)')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'exercise_a_omega_test.png'), dpi=150)
    plt.close()
    print(f"\nFigure saved: exercise_a_omega_test.png")


def exercise_b(n_steps=30):
    """Exercise B: Compare naive vs vectorized SOR performance."""
    print("\n" + "=" * 60)
    print("Exercise B: SOR Optimization Comparison")
    print("=" * 60)

    grid_sizes = [50, 100, 150]
    naive_times = []
    vec_times = []

    for N in grid_sizes:
        print(f"\n  Grid {N}x{N}...")

        # Vectorized
        t0 = time.time()
        dla_growth(N=N, eta=1.0, n_steps=n_steps, omega=1.8, solver='vectorized',
                   verbose=False)
        t_vec = time.time() - t0
        vec_times.append(t_vec)
        print(f"    Vectorized: {t_vec:.2f}s")

        # Naive (only for small grids)
        if N <= 100:
            t0 = time.time()
            dla_growth(N=N, eta=1.0, n_steps=n_steps, omega=1.8, solver='naive',
                       verbose=False)
            t_naive = time.time() - t0
            naive_times.append(t_naive)
            print(f"    Naive:      {t_naive:.2f}s")
            print(f"    Speedup:    {t_naive / t_vec:.1f}x")
        else:
            naive_times.append(None)
            print(f"    Naive: skipped (too slow)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid_sizes, vec_times, 'bo-', label='Vectorized (red-black)')
    valid_naive = [(n, t) for n, t in zip(grid_sizes, naive_times) if t is not None]
    if valid_naive:
        ns, ts = zip(*valid_naive)
        ax.plot(ns, ts, 'rs-', label='Naive (loop-based)')
    ax.set_xlabel('Grid size N')
    ax.set_ylabel('Total time (s)')
    ax.set_title(f'SOR Performance: Naive vs Vectorized ({n_steps} DLA steps)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'exercise_b_optimization.png'), dpi=150)
    plt.close()
    print(f"\nFigure saved: exercise_b_optimization.png")


if __name__ == "__main__":
    exercise_a(N=100, n_steps=200)
    exercise_a_omega_test(N=100, n_steps=50)
    exercise_b(n_steps=30)
