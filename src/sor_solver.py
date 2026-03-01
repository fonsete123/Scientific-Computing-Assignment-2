"""
SOR (Successive Over-Relaxation) solver for the Laplace equation on a 2D grid.

Supports:
- Object mask (Dirichlet c=0 at object sites)
- Boundary conditions: c=1 at top, c=0 at bottom, periodic in x
- Both naive (loop-based) and vectorized (red-black) implementations
- Warm-start from a previous solution
"""

import numpy as np
import time


def initialize_concentration(N, object_mask=None):
    """Initialize concentration field with linear gradient c(y) = y/(N-1)."""
    c = np.zeros((N, N))
    for j in range(N):
        c[j, :] = j / (N - 1)
    if object_mask is not None:
        c[object_mask == 1] = 0.0
    # Enforce boundary conditions
    c[0, :] = 0.0   # bottom
    c[-1, :] = 1.0   # top
    return c


def sor_solve_naive(c, object_mask, omega=1.8, tol=1e-5, max_iter=50000):
    """
    Naive loop-based SOR solver for the Laplace equation.

    Parameters
    ----------
    c : ndarray (N, N)
        Initial concentration field (modified in-place).
    object_mask : ndarray (N, N)
        1 where object exists (c=0), 0 where free.
    omega : float
        SOR relaxation parameter (1 < omega < 2 for over-relaxation).
    tol : float
        Convergence tolerance on max absolute change.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    c : ndarray
        Converged concentration field.
    iters : int
        Number of iterations performed.
    """
    N = c.shape[0]
    for iteration in range(1, max_iter + 1):
        delta = 0.0
        for j in range(1, N - 1):       # skip top/bottom boundaries
            for i in range(N):
                if object_mask[j, i] == 1:
                    continue
                # Periodic BC in x
                ip = (i + 1) % N
                im = (i - 1) % N
                c_new = 0.25 * (c[j, ip] + c[j, im] + c[j + 1, i] + c[j - 1, i])
                change = omega * (c_new - c[j, i])
                c[j, i] += change
                delta = max(delta, abs(change))
        if delta < tol:
            return c, iteration
    return c, max_iter


def sor_solve_vectorized(c, object_mask, omega=1.8, tol=1e-5, max_iter=50000):
    """
    Vectorized red-black SOR solver for the Laplace equation.

    Uses checkerboard (red-black) ordering so each color can be updated
    simultaneously with NumPy array operations.

    Parameters
    ----------
    c : ndarray (N, N)
        Initial concentration field (modified in-place).
    object_mask : ndarray (N, N)
        1 where object exists (c=0), 0 where free.
    omega : float
        SOR relaxation parameter.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    c : ndarray
        Converged concentration field.
    iters : int
        Number of iterations performed.
    """
    N = c.shape[0]

    # Precompute red-black masks for interior points (rows 1..N-2)
    # Red: (i+j) % 2 == 0, Black: (i+j) % 2 == 1
    jj, ii = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    red_mask = ((ii + jj) % 2 == 0)
    black_mask = ((ii + jj) % 2 == 1)

    # Only update interior rows (not top/bottom boundary)
    interior = np.zeros((N, N), dtype=bool)
    interior[1:N-1, :] = True

    free = (object_mask == 0)

    red_update = red_mask & interior & free
    black_update = black_mask & interior & free

    for iteration in range(1, max_iter + 1):
        # --- Red sweep ---
        # Neighbors with periodic x BC
        c_right = np.roll(c, -1, axis=1)
        c_left = np.roll(c, 1, axis=1)
        c_up = np.roll(c, -1, axis=0)     # j+1
        c_down = np.roll(c, 1, axis=0)    # j-1

        c_avg = 0.25 * (c_right + c_left + c_up + c_down)
        change_red = omega * (c_avg - c)
        c[red_update] += change_red[red_update]
        c[object_mask == 1] = 0.0

        # --- Black sweep ---
        c_right = np.roll(c, -1, axis=1)
        c_left = np.roll(c, 1, axis=1)
        c_up = np.roll(c, -1, axis=0)
        c_down = np.roll(c, 1, axis=0)

        c_avg = 0.25 * (c_right + c_left + c_up + c_down)
        change_black = omega * (c_avg - c)
        c[black_update] += change_black[black_update]
        c[object_mask == 1] = 0.0

        # Enforce boundary conditions
        c[0, :] = 0.0
        c[-1, :] = 1.0

        # Convergence check
        delta = max(np.max(np.abs(change_red[red_update])) if np.any(red_update) else 0,
                     np.max(np.abs(change_black[black_update])) if np.any(black_update) else 0)

        if delta < tol:
            return c, iteration

    return c, max_iter


def benchmark_solvers(N=100, omega=1.8):
    """Compare naive vs vectorized SOR on an empty domain."""
    print(f"Benchmarking SOR solvers on {N}x{N} grid (omega={omega})")

    mask = np.zeros((N, N), dtype=int)

    # Naive
    c_naive = initialize_concentration(N, mask)
    t0 = time.time()
    c_naive, iters_naive = sor_solve_naive(c_naive, mask, omega=omega)
    t_naive = time.time() - t0
    print(f"  Naive:      {iters_naive:5d} iters, {t_naive:.3f}s")

    # Vectorized
    c_vec = initialize_concentration(N, mask)
    t0 = time.time()
    c_vec, iters_vec = sor_solve_vectorized(c_vec, mask, omega=omega)
    t_vec = time.time() - t0
    print(f"  Vectorized: {iters_vec:5d} iters, {t_vec:.3f}s")

    # Verify both converge to linear gradient
    y = np.linspace(0, 1, N)
    exact = np.tile(y.reshape(-1, 1), (1, N))
    err_naive = np.max(np.abs(c_naive - exact))
    err_vec = np.max(np.abs(c_vec - exact))
    print(f"  Max error (naive):      {err_naive:.2e}")
    print(f"  Max error (vectorized): {err_vec:.2e}")
    print(f"  Speedup: {t_naive / t_vec:.1f}x")

    return t_naive, t_vec


if __name__ == "__main__":
    benchmark_solvers(N=50, omega=1.5)
