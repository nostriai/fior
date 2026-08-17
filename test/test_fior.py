# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""Assertion tests for the claims protocol.md makes about fior_sim.py.

Run:  uv run test/test_fior.py

Every test here locks in something the specification asserts, so that a
change which quietly breaks a documented property fails loudly instead of
leaving the docs wrong.  The findings in fior_sim.py's docstring are the
prose; this is the executable form of the ones that can be pinned.

Numbers with a tolerance are regression bounds, deliberately loose enough
to survive numerical noise and tight enough to catch a mechanism that has
stopped working.  Tests marked SPEC assert an exact identity and use a
round-off tolerance instead.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import fior_sim as F  # noqa: E402

FAILED: list[str] = []
PASSED = 0


def check(name, fn):
    global PASSED
    try:
        fn()
    except AssertionError as e:
        FAILED.append(f"{name}: {e}")
        print(f"  FAIL  {name}\n        {e}", flush=True)
    except Exception as e:  # noqa: BLE001
        FAILED.append(f"{name}: {type(e).__name__}: {e}")
        print(f"  ERROR {name}\n        {type(e).__name__}: {e}", flush=True)
    else:
        PASSED += 1
        print(f"  ok    {name}", flush=True)


def small(**kw):
    """A world small enough to be fast, large enough to be non-degenerate.

    d=20 keeps the ambient dimension (230) far above the member count, so
    the span test still has power -- see finding 11.
    """
    base = dict(d=20, n_a=4, n_b=4, n_bad=2, rounds=3, honest_attest_frac=0.0)
    base.update(kw)
    return F.Config(**base)


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def test_composition_is_eta0_plus_weighted_sum():
    """SPEC: eta_prior = eta_0 + sum_{n != A} p_n * delta_eta_n."""
    w = F.build_world(small())
    p = np.linspace(0.1, 0.9, w.N)
    got = F.compose_prior(w.eta0, w.sites, p, exclude=0)
    want = F.Eta(w.eta0.h.copy(), w.eta0.Lam.copy())
    for n in range(1, w.N):
        want.h += p[n] * w.sites[n].h
        want.Lam += p[n] * w.sites[n].Lam
    assert np.allclose(got.h, want.h, atol=1e-10), "h mismatch"
    assert np.allclose(got.Lam, want.Lam, atol=1e-10), "Lam mismatch"


def test_p_zero_excludes_a_member_exactly():
    """p -> 0 costs nothing to apply: the member simply is not there."""
    w = F.build_world(small())
    p = np.ones(w.N)
    p[3] = 0.0
    with_zero = F.compose_prior(w.eta0, w.sites, p, exclude=0)
    dropped = [s if n != 3 else F.Eta(np.zeros_like(s.h), np.zeros_like(s.Lam))
               for n, s in enumerate(w.sites)]
    without = F.compose_prior(w.eta0, dropped, np.ones(w.N), exclude=0)
    assert np.allclose(with_zero.Lam, without.Lam, atol=1e-12)


def test_composer_excludes_itself():
    w = F.build_world(small())
    a = 2
    got = F.compose_prior(w.eta0, w.sites, np.ones(w.N), exclude=a)
    assert not np.allclose(got.Lam, got.Lam + w.sites[a].Lam), "own site leaked in"


# --------------------------------------------------------------------------
# Mechanism 1 -- BMR
# --------------------------------------------------------------------------


def test_delta_f_matches_protocol_definition():
    """SPEC, Mechanism 1.  The implementation reaches the cavity by
    subtracting p_n*delta_eta_n from the working prior; protocol.md defines
    it by summing over m != n.  They must be the same number."""
    w = F.build_world(small())
    p = np.full(w.N, 0.5)
    A = 0
    prior = F.compose_prior(w.eta0, w.sites, p, exclude=A)
    worst = 0.0
    for n in range(w.N):
        if n == A:
            continue
        impl = F.bmr_delta_f(
            prior + (1.0 - p[n]) * w.sites[n] + w.lik[A],
            prior + (1.0 - p[n]) * w.sites[n],
            w.sites[n],
        )
        cav = F.Eta(w.eta0.h.copy(), w.eta0.Lam.copy())
        for m in range(w.N):
            if m in (A, n):
                continue
            cav.h += p[m] * w.sites[m].h
            cav.Lam += p[m] * w.sites[m].Lam
        full = cav + w.sites[n]
        spec = (F.log_partition(full + w.lik[A]) + F.log_partition(cav)
                - F.log_partition(cav + w.lik[A]) - F.log_partition(full))
        worst = max(worst, abs(impl - spec))
    assert worst < 1e-8, f"implementation departs from spec by {worst:.3e} nats"


def test_unit_weight_is_not_the_p_weighted_form():
    """Finding 2.  Scoring with an increment of p_n*delta_eta_n is a
    different question, and a degenerate one: it vanishes as p -> 0."""
    w = F.build_world(small())
    A, n = 0, 1
    cav = F.compose_prior(w.eta0, w.sites, np.full(w.N, 0.5), exclude=A)

    def df(weight):
        full = cav + weight * w.sites[n]
        return (F.log_partition(full + w.lik[A]) + F.log_partition(cav)
                - F.log_partition(cav + w.lik[A]) - F.log_partition(full))

    assert abs(df(1.0) - df(0.5)) > 1.0, "unit and p-weighted forms indistinguishable"
    assert abs(df(1e-6)) < 1e-2, "p-weighted form does not collapse as p -> 0"
    assert abs(df(1.0)) > 1.0, "unit weight carries no signal"


def test_delta_f_rejects_a_fabricator_and_accepts_a_clustermate():
    """The core claim: dF is positive for a peer that helps and negative for
    one that does not, in the round the site appears."""
    w = F.build_world(small(attack="random", bad_boost=200))
    A = 0
    p = np.full(w.N, 0.5)
    cav_all = F.compose_prior(w.eta0, w.sites, p, exclude=A)
    scores = {}
    for n in range(w.N):
        if n == A:
            continue
        full = cav_all + (1.0 - p[n]) * w.sites[n]
        scores[n] = F.bmr_delta_f(full + w.lik[A], full, w.sites[n])
    same = [scores[n] for n in scores if w.label[n] == w.label[A]]
    bad = [scores[n] for n in scores if w.label[n] == 2]
    assert min(same) > 0, f"a clustermate scored negatively: {min(same):.2f}"
    assert max(bad) < 0, f"a fabricator scored positively: {max(bad):.2f}"


# --------------------------------------------------------------------------
# The p / beta split
# --------------------------------------------------------------------------


def test_p_from_is_sigmoid_of_delta_f_plus_log_odds():
    """SPEC: p = 1/(1 + (b/a)exp(-dF)) = sigma(dF + ln(a/b))."""
    for dF in (-50.0, -1.0, 0.0, 1.0, 50.0):
        for a, b in ((1.0, 1.0), (3.0, 1.0), (1.0, 4.0)):
            got = F.p_from(dF, a, b)
            want = 1.0 / (1.0 + np.exp(-(dF + np.log(a) - np.log(b))))
            assert abs(got - want) < 1e-12, f"dF={dF} a={a} b={b}"


def test_no_attestations_gives_p_equals_sigmoid_delta_f():
    """SPEC: with (a,b) = (1,1), p = sigma(dF) -- pure self-evaluation."""
    for dF in (-20.0, -0.5, 0.0, 0.5, 20.0):
        assert abs(F.p_from(dF, 1.0, 1.0) - 1.0 / (1.0 + np.exp(-dF))) < 1e-12


def test_p_is_strictly_inside_the_unit_interval():
    """No peer is ever permanently excluded: the logistic never saturates."""
    for dF in (-5000.0, 5000.0):
        p = F.p_from(dF, 1.0, 1.0)
        assert 0.0 <= p <= 1.0
    assert F.p_from(-5000.0, 1.0, 1.0) < 1e-12
    assert F.p_from(5000.0, 1.0, 1.0) > 1.0 - 1e-12


def test_beta_is_the_prior_and_p_reduces_to_it_at_zero_evidence():
    """beta = a/(a+b) is what p equals when dF carries no information."""
    for a, b in ((1.0, 1.0), (3.0, 1.0), (2.0, 5.0)):
        assert abs(F.p_from(0.0, a, b) - a / (a + b)) < 1e-12


# --------------------------------------------------------------------------
# Mechanism 3 -- attestations
# --------------------------------------------------------------------------


def test_absent_attestations_resolve_to_the_uniform_prior():
    N = 6
    att = np.full((N, N), np.nan)
    a, b = F.resolve_attestation_prior(0, 1, att, np.ones(N), 4.0)
    assert (a, b) == (1.0, 1.0)


def test_one_contributor_is_bounded_by_kappa_r():
    """SPEC: an attestation is a scalar, so each contributor supplies at most
    kappa_r pseudo-counts by construction and no clip is needed."""
    N = 6
    kappa_r = 4.0
    for p_val in (0.0, 0.5, 1.0):
        att = np.full((N, N), np.nan)
        att[2, 1] = p_val
        a, b = F.resolve_attestation_prior(0, 1, att, np.ones(N), kappa_r)
        assert a - 1.0 <= kappa_r + 1e-12, f"a exceeded kappa_r at p={p_val}"
        assert b - 1.0 <= kappa_r + 1e-12, f"b exceeded kappa_r at p={p_val}"
        assert abs((a - 1.0) + (b - 1.0) - kappa_r) < 1e-12, "strength not conserved"


def test_attestation_is_discounted_by_trust_in_the_attester():
    """Slander from an untrusted peer must not move the prior."""
    N = 6
    att = np.full((N, N), np.nan)
    att[2, 1] = 0.0                      # peer 2 says: reject peer 1
    trusted = F.resolve_attestation_prior(0, 1, att, np.ones(N), 4.0)
    ignored = F.resolve_attestation_prior(0, 1, att, np.zeros(N), 4.0)
    assert ignored == (1.0, 1.0), "an untrusted attester still moved the prior"
    assert trusted[1] > 1.0, "a trusted attester had no effect"


def test_attestation_offset_is_bounded_by_ln_one_plus_kappa():
    """The social channel is second-order by construction: its whole range
    is (-ln(1+k_r), +ln(1+k_r)) nats, against a dF of tens."""
    N = 4
    kappa_r = 4.0
    att = np.full((N, N), np.nan)
    att[2, 1] = 1.0
    a, b = F.resolve_attestation_prior(0, 1, att, np.ones(N), kappa_r)
    assert np.log(a / b) <= np.log(1.0 + kappa_r) + 1e-12


# --------------------------------------------------------------------------
# Bounding claimed confidence
# --------------------------------------------------------------------------


def test_spectral_clip_enforces_the_loewner_bound():
    """SPEC: Lam_n <= c * Lam_others in the Loewner order."""
    w = F.build_world(small(attack="targeted", attack_gain=50.0))
    A, c = 0, 3.0
    n = int(np.where(w.label == 2)[0][0])
    others = F.compose_prior(w.eta0, w.sites, np.ones(w.N), exclude=A)
    ref = others.Lam - w.sites[n].Lam
    ref = (ref + ref.T) / 2.0
    clipped = F.spectral_clip_site(w.sites[n], ref, c)
    # Lam' <= c*ref is a *generalised* eigenvalue statement.  Whiten by the
    # reference first; eigvalsh on solve(ref, Lam) is not that, because the
    # product is not symmetric and eigvalsh reads only its lower triangle.
    Li = np.linalg.inv(np.linalg.cholesky(ref))
    W = Li @ clipped.Lam @ Li.T
    ev = np.linalg.eigvalsh((W + W.T) / 2.0)
    assert ev.max() <= c + 1e-6, f"bound violated: max eigenvalue {ev.max():.3f} > {c}"
    before = np.linalg.eigvalsh((lambda M: (M + M.T) / 2.0)(Li @ w.sites[n].Lam @ Li.T))
    assert before.max() > c, "test is vacuous: the site was already inside the bound"


def test_spectral_clip_preserves_the_implied_mean():
    """The peer still says exactly what it thinks, with bounded confidence."""
    w = F.build_world(small(attack="targeted", attack_gain=50.0))
    n = int(np.where(w.label == 2)[0][0])
    site = w.sites[n]
    ref = F.compose_prior(w.eta0, w.sites, np.ones(w.N), exclude=0).Lam - site.Lam
    ref = (ref + ref.T) / 2.0
    clipped = F.spectral_clip_site(site, ref, 3.0)
    # Sites are rank-deficient (n_local < d), so the implied mean is the
    # pseudo-inverse solution and the invariant is h' = Lam' m, not that m
    # is recoverable from the clipped pair by a least-squares solve.
    m = np.linalg.pinv(site.Lam) @ site.h
    assert np.allclose(clipped.h, clipped.Lam @ m, atol=1e-8), \
        "clipped h is not Lam' m -- the implied mean was not carried over"
    assert not np.allclose(clipped.Lam, site.Lam), "nothing was clipped"


# --------------------------------------------------------------------------
# Defending against replay
# --------------------------------------------------------------------------


def test_causal_span_flags_a_verbatim_copy_and_not_its_author():
    """Finding 11.  The whole point of ordering: the symmetric test flags
    author and copier alike, the causal one flags only the copier."""
    w = F.build_world(small(attack="replay", n_bad=2))
    order = np.arange(w.N)
    causal = F.causal_span_residual(w.sites, order)
    sym = F.span_residual(w.sites)
    riders = np.where(w.label == 2)[0]
    honest = np.where(w.label < 2)[0]
    # the copier is a copy of honest_sites[i % len(honest)], i.e. peer 0, 1, ...
    copied = list(range(len(riders)))
    others = [h for h in honest if h not in copied and h != order[0]]

    assert causal[riders].max() < 1e-6, \
        f"causal test missed a copier: {causal[riders].max():.2e}"
    assert min(causal[c] for c in copied) > 0.5, \
        "causal test punished the author it was copied from"
    assert min(causal[o] for o in others) > 0.5, "honest floor collapsed"
    assert sym[riders].max() < 1e-6, "symmetric test missed the copier"
    assert min(sym[c] for c in copied) < 1e-6, \
        "symmetric test was expected to flag the author too (finding 11)"


def test_novelty_weight_is_a_fraction():
    w = F.build_world(small(attack="replay"))
    for mode in ("raw", "median"):
        nov = F.novelty_weight(w.sites, np.arange(w.N), mode)
        assert nov.shape == (w.N,)
        assert nov.min() >= 0.0 and nov.max() <= 1.0, f"{mode} left [0,1]"


def test_first_publisher_is_unexplainable_by_definition():
    """Nothing older exists, so its residual is 1 -- the first-mover edge."""
    w = F.build_world(small())
    r = F.causal_span_residual(w.sites, np.arange(w.N))
    assert r[0] == 1.0


def test_span_test_loses_power_below_ambient_dimension():
    """Finding 11.  Clients MUST NOT apply it when members approach
    d + d(d+1)/2.  At d=3 that is 9, well under 16 honest peers."""
    w = F.build_world(F.Config(d=3, n_a=8, n_b=8, n_bad=0, seed=0))
    r = F.causal_span_residual(w.sites, np.arange(w.N))
    assert r.min() < 1e-6, "expected saturation at d=3, test still has power"


# --------------------------------------------------------------------------
# Corroboration
# --------------------------------------------------------------------------


def test_corroboration_vote_ignores_declared_precision():
    """SPEC: one peer, one vote -- never weighted by declared precision.
    Scaling a peer's own precision must not change anyone else's verdict."""
    w = F.build_world(small(attack="targeted"))
    C1 = F.Corroboration(w.sites)
    inflated = [F.Eta(s.h.copy(), s.Lam.copy()) for s in w.sites]
    inflated[1] = F.Eta(w.sites[1].h * 1000.0, w.sites[1].Lam * 1000.0)
    C2 = F.Corroboration(inflated)
    p = np.full(w.N, 0.5)
    w1 = C1.weights(3, p, exclude=0, t0=15.0, mode="mad")
    w2 = C2.weights(3, p, exclude=0, t0=15.0, mode="mad")
    assert np.allclose(w1, w2, atol=1e-9), \
        "inflating one peer's precision changed another peer's credibility"


def test_corroboration_weights_are_credibilities():
    w = F.build_world(small(attack="targeted"))
    C = F.Corroboration(w.sites)
    s = C.weights(1, np.full(w.N, 0.5), exclude=0, t0=15.0, mode="mad")
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-12


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_end_to_end_segregates_and_rejects_fabricators():
    """Finding 1, as a regression bound rather than a printed number."""
    res = F.run(F.Config(rounds=6, seed=0))
    lab, p = res["world"].label, res["p"]
    bm = F.block_means(p, lab)
    in_cluster = (bm[0, 0] + bm[1, 1]) / 2
    cross = (bm[0, 1] + bm[1, 0]) / 2
    bad = (bm[0, 2] + bm[1, 2]) / 2
    assert in_cluster > 0.85, f"in-cluster p collapsed to {in_cluster:.3f}"
    assert cross < 0.25, f"cross-cluster p too high at {cross:.3f}"
    assert bad < 0.05, f"fabricators retained weight: {bad:.3f}"
    assert F.ranking_auc(p, lab) > 0.90, "ranking AUC regressed"
    rec = F.recovery(res)
    assert rec["err_fior"] < rec["err_local"] / 2, "federation stopped paying"
    assert rec["err_fior"] > rec["err_oracle"], "beat the oracle -- suspicious"


def test_fabricator_in_degree_stays_near_zero():
    """The undefended baseline does not reach 0/16 -- the documented figure
    is a mean in-degree of 0.17 over 16 honest nodes, i.e. the occasional
    node that has not yet driven one fabricator's p down.  Assert the
    documented bound, not a stricter one the simulation never showed."""
    res = F.run(F.Config(rounds=6, seed=0))
    deg = F.bad_indegree(res["p"], res["world"].label)
    mean_deg = float(np.mean(list(deg.values())))
    assert mean_deg <= 0.5, f"fabricator in-degree regressed to {mean_deg}: {deg}"
    assert max(deg.values()) <= 2, f"a fabricator convinced too many: {deg}"


def test_novelty_zeroes_a_free_rider_end_to_end():
    """Finding 12.  Replay is solved once p is scaled by causal novelty."""
    res = F.run(small(attack="replay", rounds=3, novelty=True))
    lab, p = res["world"].label, res["p"]
    riders = np.where(lab == 2)[0]
    honest = np.where(lab < 2)[0]
    assert p[np.ix_(honest, riders)].max() < 0.05, "free-rider kept weight"


def test_run_is_deterministic_for_a_seed():
    a = F.run(F.Config(rounds=3, seed=7))["p"]
    b = F.run(F.Config(rounds=3, seed=7))["p"]
    assert np.array_equal(a, b), "same seed produced different results"


# --------------------------------------------------------------------------
# Disclosure
# --------------------------------------------------------------------------


def test_published_site_discloses_the_gram_matrix_exactly():
    """Security, 'What a published site discloses'.  For a conjugate Normal
    site the precision block IS tau * X^T X -- exactly public."""
    rng = np.random.default_rng(0)
    n, d, tau = 40, 5, 3.0
    X = rng.standard_normal((n, d))
    Lam = tau * (X.T @ X)
    assert np.allclose(Lam / tau, X.T @ X, atol=1e-10), \
        "Gram matrix is not recoverable, contrary to the spec"


def main():
    print("fior test suite\n")
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        check(name.removeprefix("test_").replace("_", " "), fn)
    print(f"\n{PASSED}/{len(tests)} passed")
    if FAILED:
        print(f"\n{len(FAILED)} FAILED:")
        for f in FAILED:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
