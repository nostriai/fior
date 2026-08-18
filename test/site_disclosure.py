# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""What does a published Normal site disclose about the publisher's data?

Companion to fior_sim.py, for the "What a published site discloses"
section of protocol.md.

For a Normal group with known noise precision tau, the site is exactly
    Lam = tau * X'X ,  h = tau * X'y
i.e. the sufficient statistics.  No inversion attack is needed: the
question is only what those statistics determine.
"""
import numpy as np

rng = np.random.default_rng(0)
d = 100
tau = 1.0 / 0.5**2

for n in (1, 2, 20):
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = X @ w + 0.5 * rng.standard_normal(n)
    Lam, h = tau * (X.T @ X), tau * (X.T @ y)

    G = Lam / tau                      # X'X, exact
    gram_err = np.abs(G - X.T @ X).max()

    # Recover X up to a left-orthogonal factor: X = Q B with B'B = X'X.
    ev, V = np.linalg.eigh(G)
    k = int((ev > 1e-8 * max(ev.max(), 1)).sum())
    B = (V[:, -k:] * np.sqrt(ev[-k:])).T          # k x d, B'B = X'X
    # y in that basis: B'yb = X'y  =>  yb = (BB')^-1 B X'y
    yb = np.linalg.solve(B @ B.T, B @ (h / tau))

    # Best row-space match: is there an orthogonal Q with QB = X?
    U, _, Vt = np.linalg.svd(X @ B.T)
    Q = U @ Vt
    row_err = np.abs(Q @ B - X).max()
    yq = Q @ yb
    y_err = np.abs(yq - y).max()

    print(f"n_local={n:3d}  rank={k:3d}  "
          f"max|X'X recovered - true| = {gram_err:.2e}   "
          f"max|X_hat - X| = {row_err:.2e}   max|y_hat - y| = {y_err:.2e}")

print()
print("The Gram matrix is exact in every case: feature variances and all")
print("pairwise feature correlations in the publisher's sample are public.")
print("Rows are determined only up to an n x n orthogonal mixing Q, which is")
print("not published -- but for n_local = 1 that group is just {+1, -1}, so a")
print("single-observation site discloses the observation itself.")
