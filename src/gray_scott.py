"""
Gray-Scott Reaction-Diffusion Model.

Exercise E: Simulate Turing-like pattern formation with different parameter sets.

Equations:
    du/dt = D_u * laplacian(u) - u*v^2 + f*(1 - u)
    dv/dt = D_v * laplacian(v) + u*v^2 - (f + k)*v

Uses explicit Euler time stepping and periodic boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

FIGURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)


def laplacian_periodic(field):
    """Compute 2D Laplacian with periodic BC using 5-point stencil (dx=1)."""
    return (
        np.roll(field, 1, axis=0) +
        np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) +
        np.roll(field, -1, axis=1) -
        4 * field
    )


def gray_scott_simulate(N=256, Du=0.16, Dv=0.08, f=0.035, k=0.060,
                         dt=1.0, n_steps=10000, snapshot_steps=None,
                         seed=42):
    """
    Simulate the Gray-Scott reaction-diffusion system.

    Parameters
    ----------
    N : int
        Grid size.
    Du, Dv : float
        Diffusion coefficients for u and v.
    f : float
        Feed rate.
    k : float
        Kill rate.
    dt : float
        Time step.
    n_steps : int
        Number of time steps.
    snapshot_steps : list of int
        Steps at which to save snapshots.

    Returns
    -------
    u, v : ndarray
        Final concentration fields.
    snapshots : list of (step, v_copy)
        Snapshots of v at requested steps.
    """
    if snapshot_steps is None:
        snapshot_steps = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]

    rng = np.random.default_rng(seed)

    # Initial conditions: u=1 everywhere, v=0 everywhere
    u = np.ones((N, N))
    v = np.zeros((N, N))

    # Perturbed center square: u=0.5, v=0.25
    r = 10  # half-width of center square
    cx, cy = N // 2, N // 2
    u[cy - r:cy + r, cx - r:cx + r] = 0.5
    v[cy - r:cy + r, cx - r:cx + r] = 0.25

    # Add small random noise to break symmetry
    u += rng.uniform(-0.01, 0.01, (N, N))
    v += rng.uniform(-0.01, 0.01, (N, N))
    v = np.clip(v, 0, 1)
    u = np.clip(u, 0, 1)

    snapshots = []
    if 0 in snapshot_steps:
        snapshots.append((0, v.copy()))

    for step in range(1, n_steps + 1):
        lap_u = laplacian_periodic(u)
        lap_v = laplacian_periodic(v)

        uvv = u * v * v

        u_new = u + dt * (Du * lap_u - uvv + f * (1 - u))
        v_new = v + dt * (Dv * lap_v + uvv - (f + k) * v)

        u = u_new
        v = v_new

        if step in snapshot_steps:
            snapshots.append((step, v.copy()))

    return u, v, snapshots


def exercise_e():
    """Exercise E: Gray-Scott reaction-diffusion patterns."""
    print("=" * 60)
    print("Exercise E: Gray-Scott Reaction-Diffusion")
    print("=" * 60)

    N = 256
    n_steps = 10000

    # Parameter sets (well-known Gray-Scott regimes)
    param_sets = [
        {'f': 0.035, 'k': 0.060, 'label': 'Labyrinth (f=0.035, k=0.060)',
         'steps': 10000},
        {'f': 0.030, 'k': 0.057, 'label': 'Spots (f=0.030, k=0.057)',
         'steps': 15000},
        {'f': 0.025, 'k': 0.055, 'label': 'Stripes (f=0.025, k=0.055)',
         'steps': 15000},
        {'f': 0.040, 'k': 0.060, 'label': 'Waves (f=0.040, k=0.060)',
         'steps': 10000},
    ]

    # First: time evolution of default parameters
    print("\nRunning default parameters with snapshots...")
    snapshot_steps = [0, 1000, 2500, 5000, 7500, 10000]
    t0 = time.time()
    u, v, snapshots = gray_scott_simulate(
        N=N, n_steps=n_steps, snapshot_steps=snapshot_steps
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (step, v_snap) in enumerate(snapshots):
        ax = axes[idx]
        im = ax.imshow(v_snap, origin='lower', cmap='inferno',
                       vmin=0, vmax=0.35)
        ax.set_title(f't = {step}', fontsize=13)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    plt.suptitle('Gray-Scott: Time Evolution (f=0.035, k=0.060)', fontsize=15)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label='v concentration')
    plt.savefig(os.path.join(FIGURE_DIR, 'exercise_e_time_evolution.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"Figure saved: exercise_e_time_evolution.png")

    # Second: compare different parameter sets (final state)
    print("\nComparing different parameter sets...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, params in enumerate(param_sets):
        steps = params.get('steps', n_steps)
        print(f"  Running {params['label']} ({steps} steps)...")
        t0 = time.time()
        u, v, _ = gray_scott_simulate(
            N=N, f=params['f'], k=params['k'],
            n_steps=steps, snapshot_steps=[]
        )
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

        ax = axes[idx]
        im = ax.imshow(v, origin='lower', cmap='inferno')
        ax.set_title(params['label'], fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f'Gray-Scott Patterns (N={N})', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'exercise_e_parameter_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: exercise_e_parameter_comparison.png")


if __name__ == "__main__":
    exercise_e()
