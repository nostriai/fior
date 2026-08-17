# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""
Core-algorithm simulation for FIOR v3.  No Nostr, no relays, no blobs --
just the arithmetic of protocol.md sections 2 and 3 on a fully connected
network of nodes.

Model: Bayesian linear regression with known noise precision
(https://en.wikipedia.org/wiki/Bayesian_linear_regression), d coefficients,
n_local observations per node.  With d=100 and n_local=20 no node can
identify the model alone -- the local likelihood has rank 20 in a
100-dimensional space -- so federation is not an optimisation, it is the
only way any node gets an answer.

Population:
  - 8 nodes drawn from cluster A: coefficients w_A, noise sigma_A
  - 8 nodes drawn from cluster B: coefficients w_B, noise sigma_B
  - 4 colluding bad actors publishing fabricated sites that all point at a
    shared w_bad, with inflated precision (they claim many more samples
    than the honest nodes have), and that endorse each other by attestation.

Question: using only local BMR scoring, does the trust matrix segregate
into the two honest clusters and isolate the bad actors?

Findings, in the order they were established.  All are reproducible from
the commands below.

1.  Segregation works, and federation is worth a great deal.  Both
    clusters averaged, 3 seeds:

      rounds  in-cluster  cross-cluster  fabricator   AUC   err_fior
         6       0.916        0.096        0.0104    0.963   0.34
        25       0.943        0.097        0.0000    0.949   0.25

    against err_local 0.89 and an oracle handed the true cluster
    membership at 0.10.  Note the error figure moves a long way between
    6 and 25 rounds while the block means barely do, so quote it with a
    round count or not at all.  Note also that AUC does not improve with
    more rounds -- segregation is essentially decided early.

2.  dF must be evaluated at unit reference weight, not at the peer's
    current p.  An earlier revision of protocol.md scaled the contribution
    by p_n, but then p_n -> 0 makes the contribution vanish, so dF -> 0, so
    p returns to the prior mean beta: a degenerate fixed point that makes
    stable rejection impossible.  The spec now says unit weight, and gives
    this as the reason.  Measured directly -- under the spec formula a
    poisoned peer scores dF = +0.0 and drifts back up, while at unit
    reference the same peer scores dF = -3110.  Run --ref-weight current
    to reproduce; final error is 1.19 versus 0.26.

3.  The marginal test is useless and the leave-one-out test is essential.
    Scoring a peer against the base prior alone gives dF = +2.5 for
    same-cluster and cross-cluster peers alike -- with a vague base prior
    any precision looks good regardless of where it points.  Only the
    leave-one-out form, against a prior that already contains the other
    peers, separates them (+112 same, -86 cross).

4.  Cluster similarity is not the limiting factor.  w_A and w_B here are
    independent draws in d=100 and so are essentially orthogonal; the
    residual cross-cluster edges come from the thinness of local evidence,
    not from the clusters resembling each other.  --cluster-cos varies it.

5.  Attestations help, but only at the margin, and the transitive rule
    does neutralise collusion.  Going from no honest attesters to half of
    them takes error from 0.49 to 0.34; past that it is flat to within
    seed noise -- three seeds, so read 0.49, 0.38, 0.34, 0.30, 0.34 at
    0/25/50/75/100% as "improves then flattens", not as a ranking.
    The reason the ceiling is low is structural: dF runs to hundreds of
    nats while an attestation contributes ln(a/b), a few nats at most, so
    direct evidence dominates by construction.  Meanwhile the four
    attackers attest maximally against every honest node in every run, and
    p->bad is 0.0104 at every attestation fraction -- their slander is
    weighted by p_{A->B}, which is already ~0.  Weighting an
    attester's testimony by the trust it has earned does what it is meant
    to do here.

    (Re-measured under the scalar attestation format of finding 13, at
    kappa_r = 4.  The superseded publisher-side (a, b) scheme gave 0.46 to
    0.25 on this sweep.  Both are 3-seed numbers and the gap is within the
    spread this file shows elsewhere, so do not read it as a regression.)

6.  A tailored attack defeats the trust layer, and does so without ever
    being trusted.  See make_attack_sites() for the construction and the
    note below for the mechanism.

The attack (finding 5) is the substantive negative result.  An attacker
that publishes a rank-one, hugely overconfident claim along the direction
the federation observes least ends up correctly rejected -- p 0.012,
trusted by 0.08 of 16 honest nodes -- and still drives relative error from
0.17 to 0.98, a 5.7x degradation, with the damage sitting in the 96
directions it never attacked rather than the 4 it did.

The mechanism is transient trust.  protocol.md sets p^(0) = a/(a+b),
which is 0.5 for a stranger, and nothing bounds the precision a site may
claim.  For the rounds before it is rejected the attacker therefore
dominates every honest node's prior; the dF each honest node computes for
every *other* honest node in those rounds is measured against that
corrupted prior; and since p = sigmoid(dF) with dF in the hundreds of
nats, those judgements saturate and never recover.  Pinning the attackers'
p to zero from round 0 reproduces the attacker-free run exactly
(0.967 / 0.031 / 0.1699 on both), which isolates transient trust as the
whole of the effect.

Neither obvious mitigation helps.  p^(0) = 0 only delays the exposure
by one round, because a peer scored against the base prior alone scores
+2.5 (finding 3) and so is admitted at ~0.92 in round 1 anyway.  A cap on
trace(Lam) does not see a rank-one spike: the attacker's trace is 2.7e4
against a median honest 4.9e3, well inside any cap loose enough to admit
honest nodes, while its precision along the attacked direction is 200x the
entire federation's evidence there.

7.  What does work is a per-direction bound.  Since the lie is directional,
    the invariant has to be directional too: cap each peer at c times what
    the evaluator's *other* peers provide, in every direction, in the
    Loewner order -- Lam_n <= c * Lam_others -- preserving the peer's
    implied mean so it still gets to say what it thinks, with bounded
    confidence.  Excluding the peer from its own reference is essential; a
    spike that counts toward its own budget bounds nothing.

    At c = 3 this fully neutralises the independent tailored attack:
    degradation versus an attacker-free world goes from 5.4x to 0.8x, and
    in-cluster p recovers from 0.803 to 0.977.  Cost to honest nodes is
    small (clean error 0.184 against 0.170 uncapped).  --spectral-cap 3.

    It does not fully solve collusion.  With all four attackers pushing the
    same direction, degradation falls from 7.3x only to 1.9x, because along
    the chosen direction the coalition *is* the other peers and so inflates
    its own budget.  The obvious repair -- replace the sum with a
    per-direction median over peers taken individually -- is worse, not
    better: it clips honest sites hardest, since an honest node's precision
    along its own data directions is far above the median peer's in those
    same directions.  Measured, with no attacker at all, median-ref c=10
    drives error to 0.464 against a 0.140 baseline.  --cap-quantile 0.5
    reproduces this.

    So: single tailored attacker, solved; coalition, bounded but open.

8.  The coalition case closes when a second, independent bound is added --
    and the design constraint that emerges is sharper than either bound.

    Every site already publishes Lam_n = tau X'X, which *is* its
    observability, so who can see which directions is public.  Blind spots
    differ: an attack tuned to an 8-node group is 26x more visible to the 8
    nodes outside it (100% of 60 draws).  A client can therefore ask, per
    direction, whether the peers it already trusts corroborate a claim --
    no data shared, no new events.  See Corroboration.

    The first version of this INVERTED under collusion.  Weighting the
    consensus by each peer's declared precision let four colluders at 2.7e4
    apiece outvote all sixteen honest nodes' 133: they became the
    consensus, honest peers were penalised for disagreeing with them, and
    p -> attacker reached 0.88 with error 18x.  --corr-mode precision
    reproduces it.

    The general lesson, which rules out a whole family of otherwise natural
    designs: **a peer's influence must be bounded only by quantities it
    does not control.**  Declared precision is free to fabricate, so it may
    never set voting weight, set a cap's reference, or scale a discrepancy.
    Every defence measured here that respects this helps; every one that
    violates it is capturable -- the precision-weighted vote above, and the
    median-reference cap in finding 7.

    The working version uses one peer one vote weighted only by earned
    trust, with discrepancy scaled by the pool's *observed* dispersion
    (MAD).  Alone it is not enough -- honest cross-cluster disagreement
    inflates MAD exactly where the attacker hides.  But it composes with
    the finding-7 cap, because the two bound different things: the cap
    bounds how *surely* a peer may assert, corroboration bounds *what*.
    Escaping one tightens the other -- lowering precision to slip the cap
    also removes the weight needed to move the consensus.

      attack             none   cap c=3   MAD t0=15 + cap3
      targeted          1.358     0.251              0.243
      targeted-collude  1.277     0.584              0.405
      random            0.251     0.289              0.239
      clean network     0.205     0.181              0.170

    (relative L2 error, 8 rounds, 3 seeds).  The combination dominates: at
    least as good on every attack, and best of all in a clean network, so
    it costs honest nodes nothing.  Collusion degradation falls from 6.2x
    to 2.4x.  A tighter t0=5 reaches 1.5x on collusion but regresses to
    1.9x on the single attacker and costs ~35% honest accuracy, so it is a
    trade rather than an improvement.

    Collusion is narrowed, not closed.  2.4x is not 1.0x.

9.  Relative L2 error on the posterior mean -- used for findings 1 and 6-8
    above -- is the wrong harm measure, and flatters every result.  It sees
    only the mean, while a fabricated precision corrupts the *variance*:
    the attack manufactures posteriors that are confidently wrong,
    collapsing the spread along a direction while displacing the mean
    there, so residuals land where the posterior asserts they cannot.

    Recomputed as held-out negative log predictive density, which scores
    the whole posterior (see free_energy()), in excess nats per test point
    over the same configuration with no attacker present:

      scenario              none    cap c=3   cap + corrob    (3 seeds)
      tailored attacker    +97.3      +1.0           +1.3
      colluding coalition  +11.1      +9.0           +2.1
      naive fabricators     +3.2      +4.4           +4.5
      absolute, no attack   4.35      3.22           2.31

    THE FIRST TWO ROWS ABOVE DO NOT REPRODUCE.  See finding 12 -- they
    are single draws from a heavy-tailed distribution, and two conclusions
    drawn from them are withdrawn there.  The rows are kept so the
    correction has something to point at.  What survives on this measure
    is the qualitative claim only: L2 on the posterior mean is the wrong
    harm measure, because it sees a displaced mean and not a collapsed
    variance, and it flatters every configuration.

    Naive fabricators are the one case where neither defence contributes;
    dF already handled them, and the excess even rises because the
    defences improve the attacker-free baseline faster than the attacked
    one.  The last row is the reason to run both anyway: absolute
    predictive loss falls from 4.35 to 2.31 nats with no attacker present.

    One further observation with no defence attached to it.  The victim's
    own free energy is ~1000 nats worse than it should be, so the damage
    is plainly visible in its own evidence -- but not *attributable*.
    Per-peer BMR asks whether removing a peer helps, and removing the
    attacker does not restore the p it corrupted for everyone else.  A
    client can see that its inference has gone wrong and have no way to
    identify who did it.  Detecting this needs an "my evidence is
    anomalously poor" trigger separate from scoring any individual peer.

10. Replay -- the free-rider of the federated learning literature -- is
    the attack every mechanism here misses, and it is the simplest one.
    A peer holding no data republishes another peer's site verbatim.  The
    claim is true, so dF scores it positively, the cap admits precision
    that is by construction a genuine peer's, and corroboration endorses
    a site that agrees exactly with a trusted peer.  Four copiers against
    sixteen honest nodes, three seeds:

                            p ->   p ->    p ->
      config                copier    author     uninvolved    nlpd
      -----------------------------------------------------------------
      no free-riders          --        --         0.983       4.354
      4 free-riders          0.957     0.118       0.468       3.301
      8 free-riders          0.963     0.026        --         2.990
      4, cap + corroborate   0.963     0.110       0.502       3.358

    The damage is to credit, not to accuracy.  Leave-one-out asks whether
    removing a peer costs the evaluator evidence, and with a duplicate
    present the answer is no for *either* member of the pair -- they are
    interchangeable to the test.  p = sigmoid(dF) with large |dF| has
    no stable interior point, so it breaks the tie rather than splitting
    it, and here it broke consistently against the original author
    (0.983 -> 0.118, and -> 0.026 once every peer is copied).  Which way
    it breaks is not something the design controls; that is the finding.
    Uninvolved honest peers are collateral (0.983 -> 0.468), and the
    defences move nothing (<0.01).

    Prediction does *not* degrade here -- 4.35 -> 3.30 -> 2.99.  The
    composition rule sums delta_eta assuming disjoint data, so a replayed
    site does enter the same observations twice and the posterior is
    over-precise; but the redundant precision points where the evaluator
    already has support, so held-out NLPD improves.  An earlier draft of
    protocol.md asserted the opposite from first principles.  It was
    wrong, and the correction is recorded rather than edited away.  This
    is not a claim that double counting is harmless at higher duplication
    or when the copied evidence is misaligned; neither was measured.

    Replaying the *pooled* sum of every honest site is caught by dF alone
    (p = 0.000): the pool carries the other cluster's evidence, which
    lowers a cluster-A evaluator's log evidence.  Over-reaching exposes
    it.  But cap + corroboration *rehabilitates* it to 0.167 and raises
    NLPD from 2.08 to 5.36.  Corroboration never adds precision -- its
    weight is at most 1 -- so this is relative: shrinking the disagreeing
    honest peers changes the cavity every dF is computed against, and a
    peer mirroring the consensus gains.  This is the invariant of finding
    7 failing in a form it was not stated strongly enough to catch.
    Excluding a peer by *identity* from its own reference is not enough
    when its *content* is a copy of that reference.  The one measured
    case here where a defence makes an attack materially worse.

11. Replay *is* defensible, by a span test -- but only once publication
    order is available.  See span_residual / causal_span_residual.

    The general free-rider publishes an arbitrary linear combination of
    the sites it has read (attack "replay-mix"), duplicating nobody.  So
    ask, for each site, how much of it is unreachable as a linear
    combination of the others: one least squares solve on public data,
    metric-free for exact dependence, no local data needed.

    Symmetric, this does not work.  d=100, 16 honest, 4 riders:

      free-riders                     ~4e-15
      honest peers whose site was used ~2e-14
      honest peers not used            0.90 - 0.92

    Linear dependence is a property of a set, not a direction.  The
    author is flagged exactly as hard as the copier -- the same failure
    that stopped dF telling them apart, reappearing in the detector.  At
    mix_k=16 every honest node is used by some rider and all 20 sites
    score zero: the test says nothing at all.

    Projecting only onto STRICTLY OLDER sites fixes it completely:

      rider strategy            honest min  honest med   rider max
      ---------------------------------------------------------------
      verbatim copy               0.9115      0.9564     4.1e-15
      blend of 4                  0.9115      0.9564     5.0e-15
      blend of 16                 0.9115      0.9564     1.4e-15
      pooled sum of all sites     0.9115      0.9564     2.7e-15
      blend of 4, +1% noise       0.9115      0.9564     9.9e-03

    The honest floor never moves, because ordering only removes candidate
    explanations.  Camouflage does not pay: to clear 0.9 the rider needs
    noise 3x the norm of its own blend, and by then dF rejects it anyway
    (p 0.14, nlpd 35 against a ~4 baseline).

      noise   0     0.01    0.1     1.0     3.0
      resid   0.000 0.010   0.098   0.680   0.906
      p    0.20  0.23    0.12    0.34    0.14

    dF prices what a site asserts, the residual prices what it adds, and
    no perturbation satisfies both.  Note also that blending is already
    the weaker attack -- p 0.20 versus 0.957 for a verbatim copy,
    since leave-one-out finds a blend redundant against the originals.

    Power runs out when peers approach a site's ambient dimension,
    d + d(d+1)/2.  Clean network, 16 honest, no riders, min residual:

      d      ambient   causal   symmetric
      100      5150     0.922     0.905
       20       230     0.638     0.604
        6        27     0.247     0.179
        3         9     0.000     0.000
        2         5     0.000     0.000

    Means over 5 seeds.  Causal is the deployed test; the symmetric column
    is the sharper statement of the geometry.  An earlier revision of this
    block quoted the symmetric minima (0.897 / 0.610 / 0.135) as though
    they described the causal test.

    Below saturation everyone is a combination of everyone else and the
    test reports the whole network as free-riders.  Same effect shows up
    as a first-mover advantage -- causal residual by publication order
    decays 1.00 -> 0.94 at d=100, 1.00 -> 0.70 at d=20, 1.00 -> 0.34 at
    d=6 -- so scaling p by the raw residual penalises late joiners for
    arriving late.  novelty_weight(mode="median") normalises that decay
    away; mode="raw" does not.

12. Novelty run inside the round loop (cfg.novelty).  It solves replay
    completely and it must never be deployed alone.

    Free-rider attacks, cluster-A evaluators, 8 rounds, 3 seeds:

      attack        defence       ->rider  ->author  ->uninvolved
      -----------------------------------------------------------
      replay        none            0.957     0.118        0.468
      replay        cap+corrob      0.963     0.110        0.502
      replay        novelty         0.000     0.987        0.980
      replay-mix    none            0.200     0.885        0.870
      replay-mix    novelty         0.000     0.977        0.980

    Credit is fully restored -- the robbed author returns to 0.987
    against 0.983 in an attacker-free network.  It also repairs the
    finding-10 regression: the pooled-sum rider that corroboration
    had rehabilitated to 0.167 goes back to 0.000, excess +2.8 -> +0.3.
    Clean-network cost is small and maybe negative: within-cluster p
    0.983 -> 0.976, AUC unchanged, and all three mechanisms together give
    the best absolute NLPD and AUC measured anywhere here (2.38, 0.991).

    NOW THE NEGATIVE RESULT.  Used alone, novelty makes both confidence
    fabrication attacks much worse.  8 seeds, excess NLPD, mean (max):

      defence      tailored           colluding
      -------------------------------------------
      none          +44.4 (181)        +10.4 (23.1)
      cap c=3        +1.4 (  3.4)       +6.1 (14.5)
      cap+corrob     +1.7 (  6.9)       +4.0 (28.4)
      novelty       +76.3 (196.8)      +14.6 (60.4)
      all three      +4.1 ( 20.4)       +2.7 ( 9.5)

    A tailored attack is *novel by construction*: it plants its lie along
    the direction the federation observes least, which is exactly the
    direction no existing site can explain, so it scores 0.993 -- higher
    than any honest peer.  Novelty rewards precisely what the cap exists
    to punish.  The two are in tension, not alignment.

    Note what that does to the trust metrics: novelty alone drives
    p -> attacker to 0.003, the best value in the whole table, while
    predictive loss triples.  A mechanism can optimise every p figure
    in this file and wreck the inference.

    No configuration wins everywhere.  Cap alone is best on a lone
    attacker (+1.4, max 3.4); all three is best on a coalition (+2.7,
    max 9.5) and its contribution there is tail control, not the mean.

    These 8-seed numbers also retire the 3-seed table in finding 9.  The
    spread is an order of magnitude wider than most differences between
    defences, so two claims made there are withdrawn: that the cap gives
    only a "19% reduction" against a coalition, and that corroboration is
    "the whole defence" there.  Measured properly the cap takes a
    coalition from +10.4 to +6.1 and corroboration to +4.0 -- both
    contribute and neither dominates.  Quote spreads, not point
    estimates, for anything in this file measured under attack.

13. The attestation pair (a, b) has one degree of freedom here, and the
    kappa_max clip makes the second one actively misleading.

    An honest attester publishes (1 + kappa*p, 1 + kappa*(1-p)) with
    cfg.kappa a fixed constant, so a + b - 2 = kappa is identical for every
    attestation ever published in this file.  The pair carries the mean and
    nothing else.  That is not an artifact of the simulation: dF is never
    accumulated (protocol.md is explicit), so a BMR-derived attestation has
    no round count and no sample count from which a confidence could be
    formed.  An earlier revision had a second path -- a benchmarked
    leave-one-out pair -- which did carry a real a + b; benchmarks were
    since removed from the protocol entirely, so attestations are now the
    only source of (a, b) and none of them carries a count.

    Worse, resolve_attestation_prior clips (a-1, b-1) per contributor at
    kappa_max, and clipping an unnormalised pair does not preserve its
    mean.  For one attester at p = 0.9, kappa_max = 4:

      kappa   pseudo-counts     after clip      offset (nats)
      --------------------------------------------------------
        2     ( 1.80, 0.20)     (1.80, 0.20)        0.847
        4     ( 3.60, 0.40)     (3.60, 0.40)        1.190   <- peak
        8     ( 7.20, 0.80)     (4.00, 0.80)        1.022
       32     (28.80, 3.20)     (4.00, 3.20)        0.174
       64     (57.60, 6.40)     (4.00, 4.00)        0.000   <- neutral

    Above kappa = kappa_max/p the numerator saturates while the
    denominator keeps growing, so publishing more confidence buys less
    influence; above kappa = kappa_max/(1-p) both sides pin and the reader
    recovers ln(1) = 0.  An attester asserting certainty is heard as
    having no opinion.

    End to end this is visible as a non-monotone optimum.  Targeted
    attack, kappa_max at the 4.0 default, 3 seeds, p -> attacker:

      kappa    0.5      1       2       8      32     128
      -----------------------------------------------------
      ->bad   0.213   0.172   0.158   0.126   0.191   0.215

    The minimum sits at the hard-coded default, which is luck rather than
    design.  With kappa_max scaled as kappa/2 instead, the same sweep is
    monotone (0.206 -> 0.038) -- confirming the effect is the clip and not
    the strength.

    The channel is second-order either way.  Its whole range at p = 0.9 is
    0.32 to 2.20 nats across all kappa, against |dF| with median 4.12 and
    p95 58.17 in the same run.  Nothing about a + b can make the social
    layer decisive, which is the intended ordering.

    THE FIX, now implemented here and in protocol.md: publish the scalar
    p and let the reader form (1 + k_r p, 1 + k_r (1-p)) with its own
    k_r.  ln(a/b) is the only thing the reader ever consumes -- the Beta
    variance is never touched anywhere in the trust path -- so a + b's sole
    function was as a per-attestation weight, and a weight chosen by the
    publisher is exactly what one never lets the publisher choose.  Under
    reader-derived k_r the offset ranges over
    (-ln(1+k_r), +ln(1+k_r)), monotone in p with p = 0.5 |-> 0, so every
    offset the reader is willing to grant is reachable from p alone.  The
    two-parameter form's extra reach is precisely the part kappa_max
    existed to remove, and it is now bounded structurally instead.  The
    per-contributor bound is automatic (each contributes at most k_r), so
    kappa_max is gone: one reader-side knob replaces two that fought.

    Measured, the change is neutral.  5 seeds, 25 rounds:

      config                    targeted            collude       random
                            ->bad  recov  nlpd  ->bad recov  AUC   ->bad
      ------------------------------------------------------------------
      publisher k=8,kmax=4  0.133  0.228  2.43  0.353 0.334 0.951  0.000
      reader k_r=1          0.176  0.245  2.47  0.353 0.351 0.947  0.000
      reader k_r=2          0.161  0.244  2.44  0.363 0.329 0.940  0.000
      reader k_r=4          0.134  0.249  2.61  0.337 0.363 0.923  0.000
      reader k_r=16         0.181  0.233  2.30  0.364 0.353 0.847  0.000
      no attestations       0.551  0.380  2.71  0.709 0.687 0.596  0.000

    At matched budget (k_r = the old kappa_max = 4) the two are
    indistinguishable.  Every difference down the k_r column is small
    against the gap to the no-attestation row, so the channel matters and
    its parameterisation barely does -- and at 5 seeds with this file's
    spread, read the column as "no difference" rather than ranking it.

    k_r must stay small.  At 16 a colluding minority's mutual praise is
    bounded by nothing and AUC falls 0.951 -> 0.847, while on the targeted
    attack the same setting has the best NLPD and the worst p -> bad --
    finding 12's warning again, that p figures and predictive loss can
    move in opposite directions.  cfg.kappa_r defaults to 4.

14. dF is computed correctly, and its reference is mean-field.

    Two separate questions, checked separately.

    (a) Does the implementation compute what protocol.md defines?  Yes.
    The spec builds the cavity by SUMMING over m != n; run() instead
    subtracts p_n * delta_eta_n from the working prior and tops the peer
    back up to weight 1.  Those are algebraically the same, and are the
    same numerically: max |implementation - spec| = 3.2e-12 on a dF scale
    of 153 nats, over every honest evaluator and peer.

    The increment really is unit weight.  Scoring with an increment of
    p_n * delta_eta_n instead differs by up to 39 nats at p = 0.5, and
    collapses to exactly 0.0000 once p_n reaches 0 -- the finding-2
    degenerate fixed point, reached by a different route.

    (b) Is that definition the exact Bayes factor?  No, and this had not
    been stated.  The cavity holds every other peer m at fractional p_m,
    which is no configuration of a Bernoulli z_m.  So dF is the Bayes
    factor at the mean of the inclusion posterior, not its expectation
    over the 2^(|M|-1) configurations, and A() is not linear.

    Brute-force marginalisation, 10 nodes, d=20, one seed, 2^8 = 256
    configurations per peer:

      state                        max |gap|   max rel   sign flips
      -------------------------------------------------------------
      all peers at p = 0.5          67.0 nats     0.91        0
      converged 0.97 / 0.02          6.8 nats     1.26        0

    No sign flipped in either state, which is the part that matters: the
    sign decides whether p lands above or below the prior mean beta.  The
    error is largest in the early rounds, when nothing is resolved, and
    shrinks as the assignment sharpens.  It is not free even then --
    dF = 5.2 against an exact 2.3 is p = 0.994 against p = 0.908.

    Exact marginalisation is exponential in member count, so mean-field is
    not a choice so much as the only option; EP takes the same step when it
    forms a cavity from other sites' approximate factors.  Recorded so the
    approximation is not mistaken for an identity.

    Reproduce: scratch drivers dfcheck2.py (a) and meanfield.py (b).

15. Corroboration only separates large lies.  The spec's tolerance is a
    threshold on a peer's WORST direction -- max_g |P - med| / MAD -- and
    never on a statistic averaged over directions, since a rank-one lie
    agrees in d-1 of them and any average buries it.  Measured that way,
    3 seeds, evaluator A0:

      shift    same-cluster        cross-cluster        fabricator
      ---------------------------------------------------------------
        3    med 10.6 max 29.1   med 10.4 max 15.5   med  21.0 max  62.6
       10    med  8.5 max 20.4   med  7.5 max 25.6   med  80.0 max 211.6
       30    med  6.6 max 17.8   med  6.4 max 10.4   med 248.7 max 637.0

    At shift 30 the populations are 2.5 orders apart and any tolerance in
    the gap works.  At shift 3 they OVERLAP -- honest max 29.1 above
    fabricator median 21.0 -- so no threshold separates them and picking
    one penalises honest peers to catch nothing.  The tolerance therefore
    cannot be a constant in the spec; each client must read its own
    distribution off public sites and decline to act when it is not
    separated.

    Same-cluster and cross-cluster measure alike at every displacement
    (10.6 vs 10.4, 8.5 vs 7.5, 6.6 vs 6.4), which is the division of
    labour: dF decides who is useful, corroboration decides who is
    fabricating.

    An earlier revision of protocol.md quoted "honest around 2 with a tail
    to 17, fabricator at 180" as though separation were general.  It is
    not, and those numbers did not reproduce at any displacement tested.

    Reproduce: test/corroboration_separation.py

Usage:
    uv run test/fior_sim.py                    # attestation-fraction sweep
    uv run test/fior_sim.py --detail           # single run, full trust matrix
    uv run test/fior_sim.py --detail --attack targeted --attack-gain 200 \
        --attack-shift 30 --rounds 12          # the attack above
    uv run test/fior_sim.py --ref-weight current   # finding 2
    uv run test/fior_sim.py --detail --attack targeted --attack-gain 200 \
        --attack-shift 30 --rounds 12 --spectral-cap 3   # finding 7, the fix
    uv run test/fior_sim.py --detail --attack targeted-collude --attack-gain 200 \
        --attack-shift 30 --rounds 8 --spectral-cap 3 --corroborate 15  # finding 8
    uv run test/fior_sim.py --detail --attack replay --rounds 8   # finding 10
    uv run test/fior_sim.py --detail --attack replay-mix --mix-k 4 \
        --mix-noise 0.01 --rounds 8            # finding 11, the general case
    uv run test/fior_sim.py --detail --attack replay --rounds 8 --novelty
                                               # finding 12, the fix
    uv run test/fior_sim.py --detail --attack targeted --attack-gain 200 \
        --attack-shift 30 --rounds 8 --novelty  # finding 12, do NOT do this
    uv run test/fior_sim.py --detail --attack replay-consensus --rounds 8 \
        --spectral-cap 3 --corroborate 15   # finding 10, the rehabilitation
    uv run test/fior_sim.py --n-local 4        # weak-local-evidence regime
    uv run test/site_disclosure.py             # what a published site leaks
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Exponential family: multivariate Normal natural parameters
# --------------------------------------------------------------------------


@dataclass
class Eta:
    """Natural parameters of a multivariate Normal, carried in
    (precision-mean, precision) form:  p(w) is proportional to
    exp(h'w - 0.5 w'Lam w).  In protocol notation eta_1 = h, eta_2 = -Lam/2.
    Scaling the pair by p scales both, which is exactly the protocol's
    claim that p leaves the implied mean alone and inflates the implied
    variance by 1/p."""

    h: np.ndarray
    Lam: np.ndarray

    def __add__(self, o: "Eta") -> "Eta":
        return Eta(self.h + o.h, self.Lam + o.Lam)

    def __sub__(self, o: "Eta") -> "Eta":
        return Eta(self.h - o.h, self.Lam - o.Lam)

    def __mul__(self, c: float) -> "Eta":
        return Eta(c * self.h, c * self.Lam)

    __rmul__ = __mul__

    def copy(self) -> "Eta":
        return Eta(self.h.copy(), self.Lam.copy())

    def mean(self) -> np.ndarray:
        return np.linalg.solve(self.Lam, self.h)


def log_partition(e: Eta) -> float:
    """A(eta) = 0.5 h'Lam^-1 h - 0.5 ln|Lam|.

    The (d/2)ln(2*pi) term is dropped: BMR takes a difference of four A()
    evaluations of equal dimension, so it cancels identically.
    """
    L = np.linalg.cholesky(e.Lam)  # raises LinAlgError outside the domain
    z = np.linalg.solve(L, e.h)
    return 0.5 * float(z @ z) - float(np.log(np.diag(L)).sum())


def in_domain(e: Eta) -> bool:
    """Normal validity domain: eta_2 negative definite, i.e. Lam positive
    definite."""
    try:
        np.linalg.cholesky(e.Lam)
        return True
    except np.linalg.LinAlgError:
        return False


# --------------------------------------------------------------------------
# Protocol section 2: composition
# --------------------------------------------------------------------------


def site_clip(sites: list[Eta], cap: float) -> np.ndarray:
    """Per-site multiplier bounding any one site's mass to `cap` times the
    median site's trace(Lam).  Computed from public data only, before and
    independently of any trust judgement, so it is a sanity bound rather
    than an opinion.  Returns all-ones when cap <= 0."""
    n = len(sites)
    if cap <= 0 or n == 0:
        return np.ones(n)
    tr = np.array([float(np.trace(s.Lam)) for s in sites])
    med = float(np.median(tr[tr > 0])) if np.any(tr > 0) else 1.0
    return np.minimum(1.0, cap * med / np.maximum(tr, 1e-30))


def spectral_clip_site(site: Eta, ref_Lam: np.ndarray, c: float) -> Eta:
    """Cap a site's precision at `c` times a reference, per direction:
    return the site with Lam' the Loewner-largest matrix satisfying
    Lam' <= c * ref_Lam and Lam' <= Lam.

    Rationale.  A trace cap is a scalar summary and cannot see a rank-one
    spike -- the tailored attacker's trace is well inside any cap loose
    enough to admit honest nodes, while its precision along one direction
    is hundreds of times the whole federation's evidence there.  The
    invariant that actually bounds the attack is directional: no peer may
    be more than c times as sure as everyone else put together, about
    anything.

    The site's implied mean is preserved (h' = Lam' m).  The peer still
    gets to claim whatever location it likes; only the confidence behind
    the claim is bounded.  This is a local client policy -- protocol.md
    already leaves how a client sets p entirely to the client.
    """
    L = np.linalg.cholesky(ref_Lam)
    Li = np.linalg.inv(L)
    W = Li @ site.Lam @ Li.T
    W = (W + W.T) / 2.0
    D, U = np.linalg.eigh(W)
    if float(D.max()) <= c:
        return site
    Wc = (U * np.minimum(D, c)) @ U.T
    Lam_c = L @ Wc @ L.T
    Lam_c = (Lam_c + Lam_c.T) / 2.0
    m = np.linalg.pinv(site.Lam) @ site.h  # sites are rank-deficient
    return Eta(Lam_c @ m, Lam_c)


def robust_clip_site(
    site: Eta, peer_Lams: list[np.ndarray], c: float, q: float = 0.5
) -> Eta:
    """Cap a site's precision per direction at `c` times the `q`-quantile of
    what individual peers claim in that direction.

    The Loewner cap against the *sum* of the other peers stops one attacker
    but not a coalition: along the direction the coalition chose, the
    coalition is the other peers, so it inflates its own budget.  Replacing
    the sum with a per-direction quantile over peers taken individually
    fixes that for any coalition below the quantile -- the usual robust
    statistics trade, applied per direction because the lie is directional.

    Directions are taken to be the site's own eigenvectors: those are the
    only directions in which it is asserting anything.
    """
    D, V = np.linalg.eigh(site.Lam)
    # ref[i] = q-quantile over peers of v_i' Lam_m v_i
    Q = np.array([np.sum(V * (Lm @ V), axis=0) for Lm in peer_Lams])
    if Q.size == 0:
        return site
    ref = np.quantile(Q, q, axis=0)
    Dc = np.minimum(D, c * np.maximum(ref, 0.0))
    if np.allclose(Dc, D):
        return site
    Lam_c = (V * Dc) @ V.T
    Lam_c = (Lam_c + Lam_c.T) / 2.0
    m = np.linalg.pinv(site.Lam) @ site.h
    return Eta(Lam_c @ m, Lam_c)


def _wmedian(P: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """Weighted median down axis 0, independently per column."""
    order = np.argsort(P, axis=0)
    cw = np.cumsum(np.take_along_axis(wts, order, axis=0), axis=0)
    denom = np.where(cw[-1] > 0, cw[-1], 1.0)
    idx = (cw / denom >= 0.5).argmax(axis=0)
    return np.take_along_axis(
        np.take_along_axis(P, order, axis=0), idx[None, :], axis=0
    )[0]


class Corroboration:
    """Directional agreement between published sites -- no data, no new events.

    The problem the scalar p cannot express: a node with n_local < d is
    blind in d - n_local directions, so its own dF has *zero* power there,
    and that is exactly where an attacker puts the lie.  But the blindness
    is per-node and the blind spots differ; a direction invisible to one
    group of nodes is visible to others (measured: 26x more visible outside
    an 8-node group than inside it).

    Nothing needs to be shared to exploit that, because every site already
    publishes its own precision block, which *is* its observability.  So a
    client can ask, for each direction a peer asserts something in: do the
    peers I already trust -- who can see this direction even though I
    cannot -- corroborate it?

    This deliberately does not distinguish malice from divergence.  A peer
    from a different cluster fails the same test as a fabricator, because
    it is asserting something my trusted peers disagree with, and the
    consequence for my inference is identical.

    Everything below is precomputed once: sites are fixed, so the
    eigenbases and the cross-peer projections do not change between rounds.
    Only the p weights do.
    """

    def __init__(self, sites: list[Eta]) -> None:
        self.N = len(sites)
        self.means = [np.linalg.pinv(s.Lam) @ s.h for s in sites]
        self.D, self.V = [], []
        for s in sites:
            d, v = np.linalg.eigh(s.Lam)
            self.D.append(d)
            self.V.append(v)
        # Q[n][m, i] = how much precision peer m has along peer n's i-th
        # asserted direction.  P[n][m, i] = where peer m puts that direction.
        self.Q, self.P = [], []
        for n in range(self.N):
            V = self.V[n]
            self.Q.append(np.array([np.sum(V * (s.Lam @ V), axis=0) for s in sites]))
            self.P.append(np.array([V.T @ mu for mu in self.means]))

    def weights(
        self,
        n: int,
        p_incl: np.ndarray,
        exclude: int,
        t0: float,
        mode: str = "mad",
    ) -> np.ndarray:
        """Per-direction credibility in [0,1] for peer n's claims, judged by
        the evaluator's currently trusted peers."""
        w = p_incl.copy()
        w[n] = 0.0
        if 0 <= exclude < self.N:
            w[exclude] = 0.0
        P = self.P[n]
        if mode == "precision":
            # RECORDED FAILURE, kept reproducible.  Weighting each peer's
            # vote by the precision it declares hands the vote to whoever
            # declares the most.  Four colluders claiming 2.7e4 apiece hold
            # 1.08e5 against all sixteen honest nodes' 133: they become the
            # consensus, and honest peers are then penalised for disagreeing
            # with them.  Measured p -> attacker of 0.88.  Self-declared
            # precision is free to fabricate and must never set voting
            # weight.
            wts = w[:, None] * self.Q[n]
            tot = wts.sum(axis=0)
            med = _wmedian(P, wts)
            t = np.abs(P[n] - med) * np.sqrt(np.maximum(tot, 0.0))
            return np.where(tot > 0, 1.0 / (1.0 + (t / t0) ** 2), 1.0)

        # One peer, one vote, weighted only by trust already earned -- never
        # by anything the peer asserts about itself.  A coalition then holds
        # only its share of the headcount, and the median survives any
        # minority.
        wts = np.repeat(w[:, None], P.shape[1], axis=1)
        med = _wmedian(P, wts)
        # Scale by the pool's *observed* dispersion rather than its claimed
        # precision -- also unfakeable, and self-calibrating: where honest
        # peers genuinely disagree the tolerance widens on its own, and
        # where they agree it tightens.
        mad = _wmedian(np.abs(P - med), wts)
        scale = np.maximum(mad, 1e-9)
        t = np.abs(P[n] - med) / scale
        return 1.0 / (1.0 + (t / t0) ** 2)

    def apply(self, n: int, site: Eta, s: np.ndarray) -> Eta:
        """Rescale peer n's precision per direction, preserving its implied
        mean: it still says what it thinks, with credibility-scaled
        confidence."""
        V, D = self.V[n], self.D[n]
        Lam = (V * (D * s)) @ V.T
        Lam = (Lam + Lam.T) / 2.0
        return Eta(Lam @ self.means[n], Lam)


def compose_prior(
    eta0: Eta, sites: list[Eta], p_incl: np.ndarray, exclude: int
) -> Eta:
    """eta_prior = eta_0 + sum_{n != A} p_{A->n} * delta_eta_n

    Sites are differences, so a delta_eta may lower precision and a weighted
    sum of valid sites need not be valid.  in_domain() is the check; there is
    no recovery from a violation and none is specified.  It has never fired
    in any run in this file -- every site here is PSD and composition is in
    f64 with no quantization -- so a caller that trips it should treat that
    as a new finding, not a routine condition.
    """
    acc = eta0.copy()
    for n, s in enumerate(sites):
        if n == exclude:
            continue
        b = p_incl[n]
        if b == 0.0:
            continue
        acc.h += b * s.h
        acc.Lam += b * s.Lam
    return acc


# --------------------------------------------------------------------------
# Protocol section 3: trust
# --------------------------------------------------------------------------


def bmr_delta_f(eta_q: Eta, eta_p: Eta, contrib: Eta) -> float:
    """dF = A(eta_q) + A(eta_p - c) - A(eta_q - c) - A(eta_p)

    `c` is the peer's contribution at the weight the *comparison* is drawn
    at, and on the default path that weight is 1, not p_n.  Callers pass
    eta_p with the peer already topped up to unit weight, so eta_p - c is
    the cavity: the peer fully absent, everyone else at their current p.
    That makes this the log Bayes factor between z_n = 1 and z_n = 0, which
    is the quantity protocol.md defines.  Passing c = p_n * delta_eta
    instead asks a different and degenerate question -- see finding 2 and
    cfg.ref_weight.

    Two notes on what this is not.  The cavity holds the other peers at
    fractional p_m, which is no configuration of a Bernoulli z, so this is
    the Bayes factor at the mean-field reference rather than its expectation
    over inclusion vectors -- see finding 14.  And it is exact given its
    inputs, so no sampling noise justifies accumulating it across rounds.

    dF > 0 means the peer earns its weight against *this evaluator's* data.
    """
    p_red = eta_p - contrib
    q_red = eta_q - contrib
    if not (in_domain(p_red) and in_domain(q_red)):
        # Cannot form the reduced model; treat as no evidence either way.
        return 0.0
    return (
        log_partition(eta_q)
        + log_partition(p_red)
        - log_partition(q_red)
        - log_partition(eta_p)
    )


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    """Overflow-safe sigmoid.  dF reaches thousands of nats at d=100, so the
    naive 1/(1+(b/a)exp(-dF)) form of the protocol formula overflows."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out if out.ndim else float(out)


def p_from(delta_f: float, a: float, b: float) -> float:
    """p = 1 / (1 + (b/a) exp(-dF)) = sigmoid(dF + ln(a/b)).

    Written in log-odds form: dF is already in nats, and the attestation
    prior enters as an additive log-odds offset.  That is the whole
    interaction between the two channels.
    """
    return float(logistic(delta_f + np.log(a) - np.log(b)))


# --------------------------------------------------------------------------
# Attestations (kind 30104) and the transitive rule
# --------------------------------------------------------------------------


def resolve_attestation_prior(
    A: int,
    C: int,
    attestations: np.ndarray,  # (N, N) published p in (0,1), NaN where absent
    p_A: np.ndarray,  # A's current posterior inclusion prob for each attester
    kappa_r: float,  # reader-side strength granted to one attestation
) -> tuple[float, float]:
    """eta_beta_trans(A->C) = sum_B p_{A->B} * kappa_r * (p_BC, 1 - p_BC)

    An attestation is a single scalar p_{B->C}: the attester's posterior
    inclusion probability, and nothing about how strongly to hold it.  The reader supplies that,
    as kappa_r, so a publisher cannot inflate its own influence.

    This replaces an earlier form in which the publisher sent Beta
    pseudo-counts (a, b) and the reader clipped them per contributor at
    kappa_max.  Two things were wrong with it.  The publisher had nothing to
    derive a + b from -- dF is never accumulated, so there is no round count
    and no sample count behind a locally computed p -- and clipping an
    unnormalised pair does not preserve its mean, which made published
    strength non-monotone in recovered influence and, past
    kappa_max/(1-p), exactly neutral.  See finding 13.

    Here every contributor supplies at most kappa_r pseudo-counts by
    construction, so the per-contributor bound is structural and no clip is
    needed.  Measured, the two forms are indistinguishable at matched
    budget (kappa_r = the old kappa_max); see finding 13.
    """
    a_acc, b_acc = 1.0, 1.0
    for B in range(attestations.shape[0]):
        if B == A or B == C:
            continue
        p = attestations[B, C]
        if np.isnan(p):
            continue
        w = p_A[B]
        if w <= 1e-9:
            continue
        a_acc += w * kappa_r * p
        b_acc += w * kappa_r * (1.0 - p)
    return a_acc, b_acc


# --------------------------------------------------------------------------
# Population
# --------------------------------------------------------------------------


@dataclass
class Config:
    d: int = 100
    n_local: int = 20
    n_a: int = 8
    n_b: int = 8
    n_bad: int = 4
    sigma_a: float = 0.5
    sigma_b: float = 1.0
    # Cosine similarity between the two clusters' true coefficients.  0.0 is
    # two independent draws, which in d=100 are essentially orthogonal -- the
    # easiest case to tell apart.  Raise it to make the clusters genuinely
    # similar and find where the trust layer stops resolving them.
    cluster_cos: float = 0.0
    alpha0: float = 1e-2  # base prior precision: N(0, 100) per coefficient
    rounds: int = 6
    # Reader-side strength granted to one attestation.  An attestation is a
    # single scalar p on the wire; this is what the *reader* multiplies it
    # by to get Beta pseudo-counts.  Publisher-side strength does not exist,
    # so there is nothing to clip and no kappa_max.  Above ~4 a colluding
    # minority's mutual praise stops being bounded by anything: AUC falls
    # 0.951 -> 0.847 at kappa_r = 16.  See finding 13.
    kappa_r: float = 4.0
    honest_attest_frac: float = 0.5
    # Weight at which a peer's contribution enters the *full* model when its
    # dF is evaluated.  "current" is protocol.md as literally written; "unit"
    # asks the inclusion question at full weight.  See the note in the
    # module docstring -- "current" has a degenerate fixed point at p=0.
    ref_weight: str = "unit"
    damping: float = 1.0  # p <- (1-damping)*p_old + damping*sigmoid(...)
    # Weight an unevaluated peer gets in round 0.  protocol.md says
    # p^(0) = a/(a+b), which is 0.5 for a stranger -- so a stranger
    # publishing an arbitrarily large fabricated precision lands in
    # everyone's first prior at half weight.  0.0 is the skeptical
    # alternative: contribute nothing until you have been scored.
    p_init: float = 0.5
    # Robust cap on any single site's mass, as a multiple of the median
    # site's trace(Lam).  0 disables.  The protocol has no such cap: a site
    # may claim unbounded precision and nothing bounds its influence before
    # it is evaluated.
    site_cap: float = 0.0
    # Diagnostic only, not a deployable defence: pin the attackers' p to
    # zero from round 0, so they are never trusted even transiently.
    force_reject_bad: bool = False
    # Per-direction cap on any one peer's precision, as a multiple of what
    # the evaluator's other peers provide in that direction.  0 disables.
    # This is the defence that actually bounds the tailored attack; see
    # spectral_clip_site().
    spectral_cap: float = 0.0
    # When > 0, the cap's reference is the per-direction quantile over peers
    # taken individually (robust to a coalition below that quantile) rather
    # than the sum of the evaluator's other peers.
    cap_quantile: float = 0.0
    # Directional corroboration between published sites (see Corroboration).
    # t0 is the tolerance in pool standard deviations; 0 disables.
    corroborate: float = 0.0
    # "mad": one peer one vote, scaled by observed dispersion (correct).
    # "precision": votes weighted by declared precision (broken; see above).
    corr_mode: str = "mad"
    bad_boost: int = 200  # fake samples the bad actors claim
    bad_sigma: float = 0.2  # fake noise the bad actors claim
    # How the bad actors fabricate their q(w).  See make_attack_sites().
    attack: str = "random"
    mix_k: int = 4  # how many peers a "replay-mix" free-rider blends
    mix_noise: float = 0.0  # relative camouflage noise added to the blend
    # Scale p by how much of a peer's site is unreachable as a linear
    # combination of strictly older sites.  See novelty_weight().
    novelty: bool = False
    novelty_mode: str = "median"
    attack_shift: float = 3.0  # how far the targeted attacks displace the mean
    attack_gain: float = 5.0  # attacker precision, relative to what is observable
    seed: int = 0


@dataclass
class World:
    cfg: Config
    N: int
    label: np.ndarray  # 0 = cluster A, 1 = cluster B, 2 = bad actor
    w_true: list[np.ndarray]  # per-cluster true coefficients (index by label)
    tau: np.ndarray  # per-node noise precision
    lik: list[Eta]  # per-node likelihood contribution
    sites: list[Eta]  # per-node published delta_eta
    eta0: Eta
    X_test: list[np.ndarray]
    y_test: list[np.ndarray]


def make_attack_sites(
    cfg: Config,
    eta0: Eta,
    honest_sites: list[Eta],
    w_bad: np.ndarray,
    rng: np.random.Generator,
) -> list[Eta]:
    """Fabricate the bad actors' published q(w).

    Everything an attacker needs is public.  Each honest site carries
    Lam_n = tau * X_n'X_n, which *is* that node's row space -- so an attacker
    can read off, exactly, which directions in parameter space each honest
    node has data about and which it does not.

    Attacks, in increasing order of how much they exploit that:

    "random"
        Confidently wrong in a direction nobody asked about: a large
        fabricated design matrix pointing at a random w_bad.  The naive
        attack, and the one BMR is built to catch.

    "overconfident"
        Publishes the honest consensus mean -- no directional lie at all --
        but with `attack_gain` times the precision it is entitled to.  Tests
        whether the trust layer prices false confidence, as opposed to false
        location.  This is the attack a lazy free-rider mounts by accident.

    "targeted"
        The tailored one.  M = sum_n Lam_n over the honest sites is the
        network's total observability; its smallest eigenvectors are the
        directions the federation collectively constrains *least*.  Attacker
        i takes the i-th such direction, agrees with the consensus in every
        other direction, and plants a confident lie along that one.

        For an evaluator A this is very close to optimal.  A's BMR score is
        the change in evidence for A's own 20 datapoints; along a direction
        A has no data about, shifting the mean costs A no evidence, while
        the added precision *raises* it.  The attacker therefore earns a
        high p from the very test meant to expose it, and simultaneously
        moves A's posterior wherever it likes in that direction.

    "targeted-collude"
        All four attackers push the same direction, stacking four times the
        precision behind one lie.

    "replay"
        The free-rider.  Attacker i republishes honest node i's site
        verbatim, holding no data of its own.  Nothing about the claim is
        false -- it is a real posterior from real observations -- which is
        why every mechanism in the trust layer passes it.  The harm is
        double counting: composition sums delta_eta assuming disjoint data,
        so the replayed observations enter the prior twice and the result
        is over-precise rather than mislocated.

    "replay-consensus"
        The maximal free-rider.  Each attacker republishes the *sum* of
        every honest site -- the public pooled sum -- as its own
        contribution, claiming the whole federation's evidence.
    """
    d = cfg.d
    n_bad = cfg.n_bad
    if not honest_sites:
        return []

    # Public knowledge: consensus location and total observability.
    cons = eta0.copy()
    for s in honest_sites:
        cons.h += s.h
        cons.Lam += s.Lam
    w_hat = cons.mean()
    M = sum((s.Lam for s in honest_sites), np.zeros((d, d)))
    evals, evecs = np.linalg.eigh(M)  # ascending: evecs[:,0] is least observed
    lam_floor = float(max(evals[0], 1e-9))
    lam_typ = float(np.median(evals))

    sites: list[Eta] = []
    for i in range(n_bad):
        if cfg.attack == "random":
            Xf = rng.standard_normal((cfg.bad_boost, d))
            Lam = (1.0 / cfg.bad_sigma**2) * (Xf.T @ Xf)
            sites.append(Eta(Lam @ w_bad, Lam))

        elif cfg.attack == "overconfident":
            Lam = cfg.attack_gain * (M / len(honest_sites))
            sites.append(Eta(Lam @ w_hat, Lam))

        elif cfg.attack in ("targeted", "targeted-collude"):
            k = 0 if cfg.attack == "targeted-collude" else i % d
            v = evecs[:, k]
            # Rank-one claim along v: "I am very sure the v-coordinate is
            # (consensus + shift)."  Precision is set relative to what the
            # federation actually observes along v, so the lie is large
            # compared to the evidence but not absurd in absolute terms.
            P = cfg.attack_gain * max(float(evals[k]), lam_floor)
            Lam = P * np.outer(v, v)
            target = float(v @ w_hat) + cfg.attack_shift
            sites.append(Eta(P * target * v, Lam))

        elif cfg.attack == "replay":
            sites.append(honest_sites[i % len(honest_sites)].copy())

        elif cfg.attack == "replay-mix":
            # The general free-rider: an arbitrary positive linear
            # combination of several peers' sites.  Defeats any test that
            # looks for a duplicate, because it is not one.
            k = min(cfg.mix_k, len(honest_sites))
            pick = rng.choice(len(honest_sites), size=k, replace=False)
            c = rng.random(k) + 0.25
            c /= c.sum()
            mixed = Eta(np.zeros(d), np.zeros((d, d)))
            for w, j in zip(c, pick):
                mixed.h += w * honest_sites[j].h
                mixed.Lam += w * honest_sites[j].Lam
            if cfg.mix_noise > 0.0:
                # Camouflage: perturb the blend to escape an exact-dependence
                # test.  The perturbation has to leave a *valid* site, so it
                # is itself a fabricated design matrix -- positive
                # semi-definite, exactly as a real likelihood contribution
                # would be -- scaled to `mix_noise` of the blend's norm.
                Z = rng.standard_normal((cfg.n_local, d))
                Nz = Z.T @ Z
                Nz *= cfg.mix_noise * float(np.linalg.norm(mixed.Lam)) / float(
                    np.linalg.norm(Nz)
                )
                mixed.Lam = mixed.Lam + Nz
                mixed.h = mixed.h + cfg.mix_noise * float(
                    np.linalg.norm(mixed.h)
                ) / np.sqrt(d) * rng.standard_normal(d)
            sites.append(mixed)

        elif cfg.attack == "replay-consensus":
            pooled = Eta(np.zeros(d), np.zeros((d, d)))
            for s in honest_sites:
                pooled.h += s.h
                pooled.Lam += s.Lam
            sites.append(pooled)

        else:
            raise ValueError(f"unknown attack: {cfg.attack}")

    del lam_typ
    return sites


def _vec_site(e: Eta) -> np.ndarray:
    """Flatten a site to a vector whose Euclidean norm is the natural one:
    h stacked on the upper triangle of Lam with off-diagonals scaled by
    sqrt(2), so that ||vec(Lam)||^2 = ||Lam||_F^2."""
    d = len(e.h)
    iu = np.triu_indices(d)
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return np.concatenate([e.h, e.Lam[iu] * scale])


def span_residual(sites: list[Eta]) -> np.ndarray:
    """How much of each site is *not* reachable as a linear combination of
    the others.  Returns residual norm relative to the site's own norm, so
    0 means "carries nothing the rest of the network did not already
    publish" and 1 means "entirely orthogonal to all of them".

    This is the closed-form generalisation of testing whether the mean of
    some subset of neighbours already accounts for a peer: rather than
    guessing which subset and which weights, least squares searches every
    coefficient vector at once.  A free-rider republishing any linear
    combination of other peers -- one site, an average, an arbitrary
    weighted blend -- lands at residual 0 by construction.

    Costs nothing private: sites are public, so this is computable by any
    reader, including one holding no data at all.
    """
    return _residual(sites, None)


def causal_span_residual(sites: list[Eta], order: np.ndarray) -> np.ndarray:
    """span_residual restricted to *strictly older* sites.

    The symmetric test above finds the linearly dependent set but cannot
    say which member is the free-rider: if a copier's site is in the
    author's span, the author's is equally in the copier's.  A publication
    order breaks that symmetry -- nothing older can explain the original,
    while everything the copier published is explained by what preceded
    it -- so credit and influence fall out of the same computation.

    `order` must come from something the publisher cannot backdate; a
    self-asserted created_at is not enough.  See protocol.md.
    """
    return _residual(sites, np.asarray(order))


def _residual(sites: list[Eta], order: np.ndarray | None) -> np.ndarray:
    V = np.array([_vec_site(s) for s in sites])
    out = np.zeros(len(sites))
    for n in range(len(sites)):
        if order is None:
            A = np.delete(V, n, axis=0).T  # D x (N-1)
        else:
            older = np.where(order < order[n])[0]
            if len(older) == 0:
                out[n] = 1.0  # nothing preceded it; wholly its own
                continue
            A = V[older].T
        b = V[n]
        nb = float(np.linalg.norm(b))
        if nb == 0.0:
            out[n] = 0.0
            continue
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        out[n] = float(np.linalg.norm(b - A @ c) / nb)
    return out


def novelty_weight(
    sites: list[Eta], order: np.ndarray, mode: str = "median"
) -> np.ndarray:
    """Per-site multiplier on p: the fraction of a peer's claim that is
    its own, from causal_span_residual.

    "raw"     use the residual directly.  Correct against free-riders and
              unfair to late joiners, who score lower purely because more
              explanations exist by the time they publish.
    "median"  divide by the median residual over all sites before
              clipping, which removes most of that decay.  Degenerates
              when free-riders are the majority: the median is then ~0,
              every weight saturates at 1, and the test is off.  That is
              the right failure direction -- it stops discriminating
              rather than inverting.
    """
    r = causal_span_residual(sites, order)
    if mode == "raw":
        return np.clip(r, 0.0, 1.0)
    ref = float(np.median(r))
    if ref <= 1e-9:
        return np.ones(len(sites))
    return np.clip(r / ref, 0.0, 1.0)


def build_world(cfg: Config) -> World:
    rng = np.random.default_rng(cfg.seed)
    d = cfg.d
    N = cfg.n_a + cfg.n_b + cfg.n_bad
    label = np.array([0] * cfg.n_a + [1] * cfg.n_b + [2] * cfg.n_bad)

    w_a = rng.standard_normal(d)
    # w_b at a controlled angle to w_a, with the same norm: project out the
    # w_a component of an independent draw, then mix back in at cos(theta).
    perp = rng.standard_normal(d)
    perp -= (perp @ w_a) / (w_a @ w_a) * w_a
    c = cfg.cluster_cos
    w_b = c * w_a + np.sqrt(max(0.0, 1.0 - c * c)) * perp * (
        np.linalg.norm(w_a) / np.linalg.norm(perp)
    )
    w_bad = rng.standard_normal(d)
    w_true = [w_a, w_b, w_bad]
    sigma = [cfg.sigma_a, cfg.sigma_b, cfg.bad_sigma]

    eta0 = Eta(np.zeros(d), cfg.alpha0 * np.eye(d))

    tau = np.zeros(N)
    lik: list[Eta] = []
    sites: list[Eta] = []

    for i in range(N):
        g = label[i]
        if g < 2:
            # Honest node: n_local real observations from its cluster.
            X = rng.standard_normal((cfg.n_local, d))
            y = X @ w_true[g] + sigma[g] * rng.standard_normal(cfg.n_local)
            t = 1.0 / sigma[g] ** 2
            tau[i] = t
            e = Eta(t * (X.T @ y), t * (X.T @ X))
            lik.append(e)
            sites.append(e.copy())  # placeholder, recomputed as post - cavity

    # Bad actors hold no data.  They read the honest sites -- which are
    # public -- and fabricate from them.
    fake = make_attack_sites(cfg, eta0, list(sites), w_bad, rng)
    for i, e in enumerate(fake):
        tau[cfg.n_a + cfg.n_b + i] = 1.0 / cfg.bad_sigma**2
        lik.append(e)
        sites.append(e.copy())

    X_test, y_test = [], []
    for g in range(2):
        Xt = rng.standard_normal((500, d))
        X_test.append(Xt)
        y_test.append(Xt @ w_true[g] + sigma[g] * rng.standard_normal(500))

    return World(cfg, N, label, w_true, tau, lik, sites, eta0, X_test, y_test)


# --------------------------------------------------------------------------
# The round loop
# --------------------------------------------------------------------------


def run(cfg: Config, world: World | None = None, verbose: bool = False):
    w = world if world is not None else build_world(cfg)
    N = w.N
    rng = np.random.default_rng(cfg.seed + 9999)

    honest = np.where(w.label < 2)[0]
    bad = np.where(w.label == 2)[0]

    # Which honest nodes publish attestations.  Bad actors always do -- an
    # adversary has no reason to abstain.
    n_att = int(round(cfg.honest_attest_frac * len(honest)))
    honest_attesters = set(rng.choice(honest, size=n_att, replace=False).tolist())

    # Sites are fixed (see the note in the round loop), so the robustness
    # cap is computed once, from public data, before any trust judgement.
    clip = site_clip(w.sites, cfg.site_cap)
    corr = Corroboration(w.sites) if cfg.corroborate > 0 else None

    # Publication order.  build_world emits honest nodes first and the
    # free-riders last, which is the true order: a copier cannot publish
    # before the site it copies.  A real client would use its own
    # first-seen order, which is per-client; here every client sees the
    # same fixed set, so one global order stands in for all of them.
    nov = (
        novelty_weight(w.sites, np.arange(N), cfg.novelty_mode)
        if cfg.novelty
        else np.ones(N)
    )

    # p[A, n] = precision A assigns to n's contribution.
    # Round 0 with no attestations is p = a/(a+b) = 1/2 per protocol.md.
    p_incl = np.full((N, N), cfg.p_init)
    np.fill_diagonal(p_incl, 0.0)
    # Bad actors do not run the algorithm: they endorse each other and
    # discard everyone else.
    for i in bad:
        p_incl[i, :] = 0.0
        p_incl[i, bad] = 1.0
        p_incl[i, i] = 0.0

    if cfg.force_reject_bad:
        p_incl[:, bad] = 0.0

    attest = np.full((N, N), np.nan)
    delta_f = np.zeros((N, N))
    priors: list[Eta] = [None] * N  # type: ignore
    posts: list[Eta] = [None] * N  # type: ignore

    for t in range(cfg.rounds):
        # ---- per-evaluator spectral cap on each peer's confidence ---------
        # eff[A] is the site list A actually uses, after bounding any one
        # peer's precision to cfg.spectral_cap times what A's *other* peers
        # provide.  Excluding the peer from its own reference is the whole
        # point: a spike that counts toward its own budget bounds nothing.
        eff = [w.sites] * N
        if corr is not None:
            eff = [
                [
                    w.sites[n]
                    if n == A
                    else corr.apply(
                        n,
                        w.sites[n],
                        corr.weights(
                            n, p_incl[A], A, cfg.corroborate, cfg.corr_mode
                        ),
                    )
                    for n in range(N)
                ]
                for A in range(N)
            ]
            if cfg.spectral_cap > 0:
                # Compose with the Loewner cap rather than replacing it: the
                # two bound different things -- corroboration bounds *what*
                # you may assert, the cap bounds *how surely*.
                capped = []
                for A in range(N):
                    base = compose_prior(w.eta0, eff[A], p_incl[A] * clip, exclude=A)
                    row = []
                    for n in range(N):
                        ref = base.Lam - p_incl[A, n] * clip[n] * eff[A][n].Lam
                        ref = (ref + ref.T) / 2.0
                        if n == A or not in_domain(Eta(base.h, ref)):
                            row.append(eff[A][n])
                        else:
                            row.append(
                                spectral_clip_site(eff[A][n], ref, cfg.spectral_cap)
                            )
                    capped.append(row)
                eff = capped
        elif cfg.spectral_cap > 0 and cfg.cap_quantile > 0:
            # The quantile reference is "what peers generally claim", which
            # does not depend on who is evaluating -- so it is computed once
            # per round rather than once per (evaluator, peer) pair.
            eff = [
                [
                    robust_clip_site(
                        w.sites[n],
                        [w.sites[m].Lam for m in range(N) if m != n],
                        cfg.spectral_cap,
                        cfg.cap_quantile,
                    )
                    for n in range(N)
                ]
            ] * N
        elif cfg.spectral_cap > 0:
            eff = []
            for A in range(N):
                base = compose_prior(w.eta0, w.sites, p_incl[A] * clip, exclude=A)
                row = []
                for n in range(N):
                    if n == A:
                        row.append(w.sites[n])
                        continue
                    ref = base.Lam - p_incl[A, n] * clip[n] * w.sites[n].Lam
                    ref = (ref + ref.T) / 2.0
                    if not in_domain(Eta(base.h, ref)):
                        row.append(w.sites[n])
                        continue
                    row.append(spectral_clip_site(w.sites[n], ref, cfg.spectral_cap))
                eff.append(row)

        # ---- inference: compose, fit, publish -----------------------------
        for A in range(N):
            p = compose_prior(w.eta0, eff[A], p_incl[A] * clip, exclude=A)
            priors[A] = p
            posts[A] = p + w.lik[A]
            if w.label[A] < 2:
                # delta_eta = eta_post - eta_cavity.  For a conjugate model
                # fitted exactly this is identically the local likelihood,
                # independent of the prior -- so honest sites are fixed
                # across rounds and all the dynamics live in p.
                w.sites[A] = posts[A] - p

        # ---- trust: score every peer by BMR, then set p ----------------
        new_p = p_incl.copy()
        for A in honest:
            for n in range(N):
                if n == A:
                    continue
                if cfg.ref_weight == "unit":
                    # Full model = everyone else at their current p, plus
                    # this peer at weight 1.  Reduced model drops it entirely.
                    # Composition is linear, so topping the peer up from
                    # p_n to 1 is a rank-update on the existing prior --
                    # no need to re-sum the other 18 sites.
                    contrib = clip[n] * eff[A][n]
                    p_full = priors[A] + clip[n] * (1.0 - p_incl[A, n]) * eff[A][n]
                else:
                    p_full = priors[A]
                    contrib = p_incl[A, n] * clip[n] * eff[A][n]
                delta_f[A, n] = bmr_delta_f(p_full + w.lik[A], p_full, contrib)
                a, b = resolve_attestation_prior(
                    A, n, attest, p_incl[A], cfg.kappa_r
                )
                tgt = p_from(delta_f[A, n], a, b)
                new_p[A, n] = (1 - cfg.damping) * p_incl[A, n] + cfg.damping * tgt
        if cfg.force_reject_bad:
            new_p[:, bad] = 0.0
        p_incl = new_p
        if cfg.novelty:
            # Applied *after* scoring, never to the unit-weight contribution
            # BMR is evaluated on.  Folding it into `clip` instead would
            # shrink what dF tests, driving dF -> 0 and p back to the
            # prior mean -- the degenerate fixed point of finding 2, in a
            # new costume.  A peer is re-tested at full weight every round
            # and its *influence* is what novelty scales.
            p_incl[honest] = p_incl[honest] * nov[None, :]
            np.fill_diagonal(p_incl, 0.0)

        # ---- attestations published for the next round --------------------
        # An attestation is one scalar: the attester's own p.  How hard to
        # hold it is the reader's call (cfg.kappa_r), not the publisher's.
        attest = np.full((N, N), np.nan)
        for B in honest_attesters:
            for n in range(N):
                if n == B:
                    continue
                attest[B, n] = p_incl[B, n]
        for B in bad:
            for n in range(N):
                if n == B:
                    continue
                # Collude: praise the other bad actors, slander everyone else.
                # Note the attackers publish the extreme values, which under
                # the old two-parameter form they could also have backed with
                # a fabricated a + b.  They no longer can.
                attest[B, n] = 1.0 if w.label[n] == 2 else 0.0

        if verbose:
            print(f"  round {t}: {block_summary(p_incl, w.label)}")

    return dict(
        world=w,
        p=p_incl,
        delta_f=delta_f,
        priors=priors,
        posts=posts,
        clip=clip,
        nov=nov,
        eff=eff,
        attesters=honest_attesters,
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def block_means(p_incl: np.ndarray, label: np.ndarray) -> np.ndarray:
    """3x3 block means of the trust matrix, honest rows only for rows 0/1."""
    out = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            m = np.outer(label == i, label == j)
            np.fill_diagonal(m, False)
            out[i, j] = p_incl[m].mean() if m.any() else np.nan
    return out


def block_summary(p_incl: np.ndarray, label: np.ndarray) -> str:
    bm = block_means(p_incl, label)
    return (
        f"A->A {bm[0,0]:.3f}  A->B {bm[0,1]:.3f}  A->bad {bm[0,2]:.3f} | "
        f"B->B {bm[1,1]:.3f}  B->A {bm[1,0]:.3f}  B->bad {bm[1,2]:.3f}"
    )


def separation_margin(p_incl: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Per honest node: min p to a same-cluster peer minus max p to any
    other node.  Positive means that node's own weighting cleanly separates
    its cluster from everything else."""
    out = []
    for A in np.where(label < 2)[0]:
        same = [n for n in range(len(label)) if n != A and label[n] == label[A]]
        other = [n for n in range(len(label)) if label[n] != label[A]]
        out.append(p_incl[A, same].min() - p_incl[A, other].max())
    return np.array(out)


def components(p_incl: np.ndarray, label: np.ndarray, thresh: float = 0.5):
    """Connected components of the *mutual* trust graph over honest nodes:
    an edge exists when A and C each weight the other above `thresh`.

    Mutuality matters.  A single stray directed edge -- one node in cluster
    A that has not yet driven one cluster-B peer's p to zero -- bridges
    the two clusters under a union rule and destroys the partition, even
    when every block mean is clean.  Requiring agreement is both the more
    honest reading of "the network segregated" and how a peer graph would
    actually be built.

    Bad actors are excluded here and reported separately by in-degree: their
    own rows are adversarial, so any mutual rule would isolate them by
    construction rather than by evidence.
    """
    honest = np.where(label < 2)[0]
    adj = {i: set() for i in honest}
    for A in honest:
        for C in honest:
            if A < C and p_incl[A, C] > thresh and p_incl[C, A] > thresh:
                adj[A].add(C)
                adj[C].add(A)
    seen, comps = set(), []
    for i in honest:
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def bad_indegree(p_incl: np.ndarray, label: np.ndarray, thresh: float = 0.5):
    """For each bad actor, how many honest nodes weight it above `thresh`.
    Zero across the board is the isolation result."""
    honest = np.where(label < 2)[0]
    return {int(x): int((p_incl[honest, x] > thresh).sum()) for x in np.where(label == 2)[0]}


def ranking_auc(p_incl: np.ndarray, label: np.ndarray) -> float:
    """Fraction of (same-cluster peer, other peer) pairs a node orders
    correctly.  Less brittle than the strict separation margin, which a
    single stubborn edge drives to -1."""
    tot = good = 0
    for A in np.where(label < 2)[0]:
        same = p_incl[A, (label == label[A])]
        same = same[np.arange(len(same)) != list(np.where(label == label[A])[0]).index(A)]
        other = p_incl[A, label != label[A]]
        cmpm = same[:, None] - other[None, :]
        good += float((cmpm > 0).sum() + 0.5 * (cmpm == 0).sum())
        tot += cmpm.size
    return good / tot if tot else float("nan")


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    ua, ub = np.unique(a), np.unique(b)
    c = np.zeros((len(ua), len(ub)))
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            c[i, j] = np.sum((a == x) & (b == y))

    def nc2(v):
        return v * (v - 1) / 2

    sij = nc2(c).sum()
    si = nc2(c.sum(1)).sum()
    sj = nc2(c.sum(0)).sum()
    tot = nc2(n)
    exp = si * sj / tot
    mx = (si + sj) / 2
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def recovery(res) -> dict[str, float]:
    """Relative L2 error of each honest node's posterior mean against its own
    cluster's true coefficients, under four weighting policies."""
    w = res["world"]
    cfg, N, label = w.cfg, w.N, w.label
    p_incl = res["p"]
    clip = res.get("clip", np.ones(N))
    eff = res.get("eff", [w.sites] * N)
    keys = ("local", "uniform", "fior", "fior_nobad", "oracle")
    out: dict[str, list[float]] = {k: [] for k in keys}
    pred: dict[str, list[float]] = {k: [] for k in keys}

    for A in np.where(label < 2)[0]:
        g = label[A]
        wt = w.w_true[g]
        # Same learned weights, but with the attackers excised.  The gap
        # between "fior" and "fior_nobad" is the damage the attackers
        # actually did, given the trust the network chose to give them --
        # which is the outcome that matters, not p itself.
        b_nobad = p_incl[A].copy()
        b_nobad[label == 2] = 0.0
        policies = {
            "local": np.zeros(N),
            "uniform": np.ones(N),
            "fior": p_incl[A],
            "fior_nobad": b_nobad,
            "oracle": (label == g).astype(float),
        }
        for name, bw in policies.items():
            bw = bw.copy() * clip
            bw[A] = 0.0
            p = compose_prior(w.eta0, eff[A], bw, exclude=A)
            m = (p + w.lik[A]).mean()
            out[name].append(np.linalg.norm(m - wt) / np.linalg.norm(wt))
            resid = w.y_test[g] - w.X_test[g] @ m
            pred[name].append(float(np.sqrt((resid**2).mean())))

    d = {f"err_{k}": float(np.mean(v)) for k, v in out.items()}
    d.update({f"rmse_{k}": float(np.mean(v)) for k, v in pred.items()})
    return d


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def free_energy(res) -> dict[str, float]:
    """Two evidence-based measures, which coefficient L2 error cannot see.

    F_train: the node's own variational free energy, -ln p(y_A | prior),
    up to a constant in y_A that is identical across configurations. This
    is the quantity BMR differences, so it is what the trust layer can
    actually perceive.

    NLPD: negative log predictive density on held-out data from the node's
    own distribution, which scores the *whole* posterior rather than its
    mean -- a posterior whose variance has been corrupted by fabricated
    precision is penalised here and is invisible to an L2 error on the
    mean.  This is the honest measure of harm done.
    """
    w = res["world"]
    N, label, p_incl = w.N, w.world.label if hasattr(w, "world") else w.label, res["p"]
    clip = res.get("clip", np.ones(N))
    eff = res.get("eff", [w.sites] * N)
    out = {k: [] for k in ("F_train", "nlpd")}
    for A in np.where(label < 2)[0]:
        g = label[A]
        bw = p_incl[A].copy() * clip
        bw[A] = 0.0
        p = compose_prior(w.eta0, eff[A], bw, exclude=A)
        q = p + w.lik[A]
        out["F_train"].append(-(log_partition(q) - log_partition(p)))
        S = np.linalg.inv(q.Lam)
        m = q.mean()
        X, y = w.X_test[g], w.y_test[g]
        var = 1.0 / w.tau[A] + np.einsum("ij,jk,ik->i", X, S, X)
        r = y - X @ m
        out["nlpd"].append(
            float(np.mean(0.5 * np.log(2 * np.pi * var) + r**2 / (2 * var)))
        )
    return {k: float(np.mean(v)) for k, v in out.items()}


def report_detail(res) -> None:
    w = res["world"]
    label, p_incl, dF = w.label, res["p"], res["delta_f"]
    names = [f"A{i}" for i in range(w.cfg.n_a)]
    names += [f"B{i}" for i in range(w.cfg.n_b)]
    names += [f"X{i}" for i in range(w.cfg.n_bad)]

    print("\nTrust matrix p[A -> n]  (rows = evaluator; X* are the bad actors)")
    print("      " + " ".join(f"{n:>5}" for n in names))
    for A in range(w.N):
        row = " ".join(
            "    ." if n == A else f"{p_incl[A, n]:5.2f}" for n in range(w.N)
        )
        tag = "  <- adversary, not computed" if label[A] == 2 else ""
        print(f"{names[A]:>5} {row}{tag}")

    print("\nBMR log Bayes factor dF[A -> n], nats (honest rows)")
    print("      " + " ".join(f"{n:>8}" for n in names))
    for A in np.where(label < 2)[0]:
        row = " ".join(
            "       ." if n == A else f"{dF[A, n]:8.1f}" for n in range(w.N)
        )
        print(f"{names[A]:>5} {row}")

    bm = block_means(p_incl, label)
    print("\nBlock means of p")
    print("             ->A      ->B    ->bad")
    for i, nm in enumerate(("A  ", "B  ", "bad")):
        print(f"  {nm}   " + "  ".join(f"{bm[i, j]:7.4f}" for j in range(3)))

    marg = separation_margin(p_incl, label)
    print(
        f"\nSeparation margin (min in-cluster p - max out-of-cluster p):"
        f"\n  min {marg.min():+.4f}   mean {marg.mean():+.4f}   "
        f"clean on {int((marg > 0).sum())}/{len(marg)} honest nodes"
        f"\nRanking AUC (same-cluster peer ranked above an outsider): "
        f"{ranking_auc(p_incl, label):.4f}"
    )

    comps = components(p_incl, label)
    print("\nMutual-trust components over the 16 honest nodes (p > 0.5 both ways):")
    for c in comps:
        print("  {" + ", ".join(names[i] for i in c) + "}")
    honest_idx = np.where(label < 2)[0]
    pred = np.zeros(len(honest_idx), dtype=int)
    pos = {v: k for k, v in enumerate(honest_idx)}
    for k, c in enumerate(comps):
        for i in c:
            pred[pos[i]] = k
    print(f"  ARI vs ground truth: {adjusted_rand_index(label[honest_idx], pred):.3f}")

    deg = bad_indegree(p_incl, label)
    print("\nBad actors: honest nodes weighting them above 0.5")
    for x, k in deg.items():
        print(f"  {names[x]}: {k}/16")


    rec = recovery(res)
    print("\nCoefficient recovery, averaged over honest nodes")
    print("  policy    rel. L2 error   test RMSE")
    for k, lbl in (
        ("local", "local only"),
        ("uniform", "all p=1"),
        ("fior", "FIOR p"),
        ("oracle", "oracle"),
    ):
        print(f"  {lbl:<12} {rec['err_' + k]:9.4f} {rec['rmse_' + k]:11.4f}")


def sweep(cfg: Config, fracs: list[float], seeds: list[int]) -> None:
    print(
        f"\nAttestation sweep  (d={cfg.d}, n_local={cfg.n_local}, "
        f"{cfg.n_a}+{cfg.n_b} honest + {cfg.n_bad} bad, "
        f"{cfg.rounds} rounds, seeds={seeds})"
    )
    print(
        f"  ref_weight={cfg.ref_weight}  damping={cfg.damping}\n"
        "\n  honest    in-clust  cross-clust    ->bad   ranking   2-cluster   bad    "
        "err_fior  err_local  err_oracle"
    )
    print(
        "  attest %     p          p           p       AUC       ARI     in-deg"
    )
    print("  " + "-" * 104)
    for f in fracs:
        acc = []
        for s in seeds:
            c = Config(**{**cfg.__dict__, "honest_attest_frac": f, "seed": s})
            res = run(c)
            b, lab = res["p"], res["world"].label
            bm = block_means(b, lab)
            rec = recovery(res)
            comps = components(b, lab)
            hidx = np.where(lab < 2)[0]
            pred = np.zeros(len(hidx), dtype=int)
            pos = {v: k for k, v in enumerate(hidx)}
            for k, cc in enumerate(comps):
                for i in cc:
                    pred[pos[i]] = k
            acc.append(
                (
                    (bm[0, 0] + bm[1, 1]) / 2,
                    (bm[0, 1] + bm[1, 0]) / 2,
                    (bm[0, 2] + bm[1, 2]) / 2,
                    ranking_auc(b, lab),
                    adjusted_rand_index(lab[hidx], pred),
                    np.mean(list(bad_indegree(b, lab).values())),
                    rec["err_fior"],
                    rec["err_local"],
                    rec["err_oracle"],
                )
            )
        m = np.mean(acc, axis=0)
        print(
            f"  {f*100:5.0f}   {m[0]:9.4f}  {m[1]:10.4f}   {m[2]:9.4f}  {m[3]:7.4f}  "
            f"{m[4]:8.3f}  {m[5]:6.2f}  {m[6]:9.4f} {m[7]:10.4f} {m[8]:11.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true", help="single run, full dump")
    ap.add_argument("--d", type=int, default=100)
    ap.add_argument("--n-local", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--attest-frac", type=float, default=0.5)
    ap.add_argument(
        "--ref-weight",
        choices=("unit", "current"),
        default="unit",
        help="weight at which a peer enters the full model when scored; "
        "'current' is protocol.md as written",
    )
    ap.add_argument("--damping", type=float, default=1.0)
    ap.add_argument(
        "--attack",
        choices=(
            "random",
            "overconfident",
            "targeted",
            "targeted-collude",
            "replay",
            "replay-mix",
            "replay-consensus",
        ),
        default="random",
        help="how the bad actors fabricate q(w); see make_attack_sites()",
    )
    ap.add_argument("--attack-gain", type=float, default=5.0)
    ap.add_argument("--attack-shift", type=float, default=3.0)
    ap.add_argument("--mix-k", type=int, default=4,
                    help="peers a replay-mix free-rider blends")
    ap.add_argument("--mix-noise", type=float, default=0.0,
                    help="camouflage noise a replay-mix free-rider adds")
    ap.add_argument("--novelty", action="store_true",
                    help="scale p by the causal span residual")
    ap.add_argument("--novelty-mode", choices=("median", "raw"),
                    default="median")
    ap.add_argument("--p-init", type=float, default=0.5)
    ap.add_argument("--site-cap", type=float, default=0.0)
    ap.add_argument("--cluster-cos", type=float, default=0.0)
    ap.add_argument("--spectral-cap", type=float, default=0.0,
                    help="per-direction Loewner cap on any peer's precision "
                         "(3 neutralises the tailored attack)")
    ap.add_argument("--corroborate", type=float, default=0.0,
                    help="directional corroboration tolerance t0 in pool MADs; "
                         "15 composes well with --spectral-cap 3")
    ap.add_argument("--corr-mode", choices=("mad","precision"), default="mad",
                    help="'precision' reproduces the coalition-capture failure")
    ap.add_argument("--cap-quantile", type=float, default=0.0,
                    help="use a per-direction quantile over peers as the cap "
                         "reference instead of the sum; measured to be worse")
    args = ap.parse_args()

    cfg = Config(
        d=args.d,
        n_local=args.n_local,
        rounds=args.rounds,
        seed=args.seed,
        honest_attest_frac=args.attest_frac,
        ref_weight=args.ref_weight,
        damping=args.damping,
        attack=args.attack,
        attack_gain=args.attack_gain,
        attack_shift=args.attack_shift,
        mix_k=args.mix_k,
        mix_noise=args.mix_noise,
        novelty=args.novelty,
        novelty_mode=args.novelty_mode,
        p_init=args.p_init,
        site_cap=args.site_cap,
        cluster_cos=args.cluster_cos,
        spectral_cap=args.spectral_cap,
        cap_quantile=args.cap_quantile,
        corroborate=args.corroborate,
        corr_mode=args.corr_mode,
    )

    if args.detail:
        print(
            f"FIOR core simulation -- d={cfg.d}, n_local={cfg.n_local}, "
            f"honest attestation fraction {cfg.honest_attest_frac:.2f}, seed {cfg.seed}"
        )
        res = run(cfg, verbose=True)
        report_detail(res)
    else:
        sweep(cfg, [0.0, 0.25, 0.5, 0.75, 1.0], list(range(args.seeds)))


if __name__ == "__main__":
    main()
