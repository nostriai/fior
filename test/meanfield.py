# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""How much does the fractional-weight cavity cost?

The implemented dF_n evaluates the log Bayes factor at ONE reference: the
mean-field prior with every other peer m at its inclusion probability p_m.
Under the Bernoulli model, z_m is 0 or 1 -- never p_m -- so the honest object
is the expectation over all 2^k inclusion vectors:

  E_z[dF_n] = sum_z P(z) [ A(q^{+n}(z)) + A(p^{-n}(z))
                           - A(q^{-n}(z)) - A(p^{+n}(z)) ]

A() is nonlinear, so E_z[dF] != dF(E_z).  Measure the gap.
"""
import sys, itertools, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import fior_sim as F

A_ = F.log_partition
cfg = F.Config(rounds=1, attack="targeted", spectral_cap=0.0, seed=0, d=20,
               n_a=4, n_b=4, n_bad=2, honest_attest_frac=0.0)
w = F.build_world(cfg)
N = w.N


def gap(p_row, a, label_note):
    others_all = [m for m in range(N) if m != a]
    rows = []
    for n in others_all:
        rest = [m for m in others_all if m != n]
        # --- mean-field reference (what the protocol computes)
        cav = F.Eta(w.eta0.h.copy(), w.eta0.Lam.copy())
        for m in rest:
            cav.h += p_row[m] * w.sites[m].h
            cav.Lam += p_row[m] * w.sites[m].Lam
        full = cav + w.sites[n]
        mf = A_(full + w.lik[a]) + A_(cav) - A_(cav + w.lik[a]) - A_(full)

        # --- exact expectation over 2^|rest| inclusion vectors
        acc, tot = 0.0, 0.0
        for z in itertools.product((0, 1), repeat=len(rest)):
            pr = 1.0
            c = F.Eta(w.eta0.h.copy(), w.eta0.Lam.copy())
            for zi, m in zip(z, rest):
                pr *= p_row[m] if zi else (1.0 - p_row[m])
                if zi:
                    c.h += w.sites[m].h
                    c.Lam += w.sites[m].Lam
            if pr < 1e-12:
                continue
            f = c + w.sites[n]
            acc += pr * (A_(f + w.lik[a]) + A_(c) - A_(c + w.lik[a]) - A_(f))
            tot += pr
        ex = acc / tot
        rows.append((n, int(w.label[n]), mf, ex))
    print(f"\n--- evaluator {a} ({label_note}), {len(others_all)-1} peers marginalised, "
          f"2^{len(others_all)-1} configurations")
    print(f"{'n':>3} {'label':>6} {'mean-field dF':>15} {'exact E_z[dF]':>15} {'gap':>10} {'rel':>8}")
    for n, lab, mf, ex in rows:
        rel = abs(mf - ex) / max(abs(ex), 1e-9)
        print(f"{n:3d} {lab:6d} {mf:15.4f} {ex:15.4f} {mf-ex:10.4f} {rel:8.4f}")
    g = np.array([abs(mf - ex) for _, _, mf, ex in rows])
    s = np.array([abs(ex) for _, _, _, ex in rows])
    print(f"    max |gap| {g.max():.4f}   max rel {(g/np.maximum(s,1e-9)).max():.4f}"
          f"   sign flips: {sum(1 for _,_,mf,ex in rows if np.sign(mf)!=np.sign(ex))}")


p = np.full((N, N), 0.5); np.fill_diagonal(p, 0.0)
gap(p[0], 0, "all peers at p = 0.5, maximum ambiguity")

# a converged-ish state: in-cluster high, out-of-cluster low
p2 = np.zeros(N)
for m in range(N):
    p2[m] = 0.97 if w.label[m] == w.label[0] else 0.02
p2[0] = 0.0
gap(p2, 0, "converged: in-cluster 0.97, others 0.02")
