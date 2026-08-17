# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""Does the implementation's dF equal protocol.md's definition?

The implementation reaches the cavity by SUBTRACTING p_n*delta_eta_n from the
working prior, then tops the peer back up to unit weight.  protocol.md defines
the cavity by SUMMING over m != n.  Check they are the same number, at a state
where the weights in play are unambiguous.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import fior_sim as F

cfg = F.Config(rounds=1, attack="targeted", spectral_cap=0.0, seed=0, d=20,
               n_a=4, n_b=4, n_bad=2, honest_attest_frac=0.0)
w = F.build_world(cfg)
N, A_ = w.N, F.log_partition

# exactly the state run() starts round 0 from
p = np.full((N, N), cfg.p_init)
np.fill_diagonal(p, 0.0)
bad = np.where(w.label == 2)[0]
for i in bad:
    p[i, :] = 0.0; p[i, bad] = 1.0; p[i, i] = 0.0
clip = np.ones(N)
sites = w.sites

impl = np.zeros((N, N)); spec = np.zeros((N, N)); pw = np.zeros((N, N))
for a in range(N):
    if w.label[a] == 2:
        continue
    prior = F.compose_prior(w.eta0, sites, p[a] * clip, exclude=a)
    for n in range(N):
        if n == a:
            continue
        # --- implementation: subtract-and-top-up, exactly as run() does
        contrib = clip[n] * sites[n]
        p_full = prior + clip[n] * (1.0 - p[a, n]) * sites[n]
        impl[a, n] = F.bmr_delta_f(p_full + w.lik[a], p_full, contrib)

        # --- protocol.md: cavity built by summing m != n, peer at unit weight
        cav = F.Eta(w.eta0.h.copy(), w.eta0.Lam.copy())
        for m in range(N):
            if m == a or m == n:
                continue
            cav.h += p[a, m] * clip[m] * sites[m].h
            cav.Lam += p[a, m] * clip[m] * sites[m].Lam
        full = cav + clip[n] * sites[n]
        spec[a, n] = (A_(full + w.lik[a]) + A_(cav)
                      - A_(cav + w.lik[a]) - A_(full))

        # --- the variant under suspicion: increment is p_n * delta_eta_n
        f2 = cav + p[a, n] * clip[n] * sites[n]
        pw[a, n] = (A_(f2 + w.lik[a]) + A_(cav)
                    - A_(cav + w.lik[a]) - A_(f2))

hon = w.label != 2
m = np.outer(hon, np.ones(N, bool)) & ~np.eye(N, dtype=bool)
print("scale of dF                      :", np.abs(spec)[m].max().round(3))
print("max |implementation - protocol.md|:", np.abs(impl - spec)[m].max())
print("  -> identical to f64 round-off" if np.abs(impl-spec)[m].max() < 1e-8
      else "  -> MISMATCH")
print()
print("p-weighted-increment variant (what p being a *probability* forbids):")
print("  max |protocol.md - p-weighted| :", np.abs(spec - pw)[m].max().round(3))
print()
print("evaluator 0, every peer at p = 0.5:")
print(f"{'n':>3} {'label':>6} {'impl':>11} {'spec':>11} {'p-weighted':>11}")
for n in range(1, N):
    print(f"{n:3d} {int(w.label[n]):6d} {impl[0,n]:11.4f} {spec[0,n]:11.4f} {pw[0,n]:11.4f}")
