/**
 * Trust layer implementation for FIOR.
 */

import { Eta } from "./types.js";

/**
 * Per-peer, per-group inclusion probabilities.
 */
export class TrustTable {
  constructor() {
    // p[peer][group] = inclusion probability
    this._p = new Map();
  }

  /**
   * Get inclusion probability for a peer and group.
   * @param {string} peer - Peer pubkey hex
   * @param {string} group - Group name
   * @param {number} defaultP - Default p value
   * @returns {number}
   */
  getP(peer, group, defaultP = 0.5) {
    const peerMap = this._p.get(peer);
    if (!peerMap) return defaultP;
    return peerMap.get(group) ?? defaultP;
  }

  /**
   * Set inclusion probability for a peer and group.
   * @param {string} peer - Peer pubkey hex
   * @param {string} group - Group name
   * @param {number} p - Inclusion probability
   */
  setP(peer, group, p) {
    if (!this._p.has(peer)) {
      this._p.set(peer, new Map());
    }
    this._p.get(peer).set(group, Math.max(1e-10, Math.min(1 - 1e-10, p)));
  }

  /**
   * List all peers with p values.
   * @returns {string[]}
   */
  peers() {
    return Array.from(this._p.keys());
  }

  /**
   * Reset p values.
   * @param {string} [peer] - Specific peer, or all if omitted
   * @param {string} [group] - Specific group, or all for peer if omitted
   */
  reset(peer, group) {
    if (peer === undefined) {
      this._p.clear();
    } else if (group === undefined) {
      this._p.delete(peer);
    } else {
      const peerMap = this._p.get(peer);
      if (peerMap) peerMap.delete(group);
    }
  }
}

/**
 * One-peer-one-vote with MAD scale.
 */
export class Corroboration {
  /**
   * @param {number} threshold - How many MADs from median before penalizing
   */
  constructor(threshold = 3.0) {
    this.threshold = threshold;
  }

  /**
   * Score a site against trusted peers.
   * @param {Site} site - Site to evaluate
   * @param {Site[]} trustedSites - List of trusted sites
   * @param {number[]} pValues - p values for trusted sites
   * @returns {number} Multiplicative factor in (0, 1]
   */
  score(site, trustedSites, pValues) {
    if (trustedSites.length === 0) return 1.0;

    // Extract precision blocks (-2 * η₂)
    const sitePrecision = -2 * site.deltaEta.Lam[0]; // simplified for single dim
    const trustedPrecisions = trustedSites.map((s) => -2 * s.deltaEta.Lam[0]);

    // Weighted median
    const weights = pValues.map((p) => p / pValues.reduce((a, b) => a + b, 0));
    const median = this._weightedMedian(trustedPrecisions, weights);

    // Weighted MAD
    const mad = this._weightedMAD(trustedPrecisions, weights, median);

    // Discrepancy
    const discrepancy = Math.abs(sitePrecision - median) / Math.max(mad, 1e-9);

    if (discrepancy > this.threshold) {
      return 1.0 / (1.0 + (discrepancy - this.threshold));
    }

    return 1.0;
  }

  _weightedMedian(values, weights) {
    const sorted = values
      .map((v, i) => ({ v, w: weights[i] }))
      .sort((a, b) => a.v - b.v);

    let cumsum = 0;
    for (const { v, w } of sorted) {
      cumsum += w;
      if (cumsum >= 0.5) return v;
    }
    return sorted[sorted.length - 1].v;
  }

  _weightedMAD(values, weights, median) {
    const deviations = values.map((v) => Math.abs(v - median));
    return this._weightedMedian(deviations, weights);
  }
}

/**
 * Apply Loewner confidence bound to a site.
 * @param {Eta} site - Site to clip
 * @param {Eta} othersPrior - Prior from other peers
 * @param {number} cap - Confidence bound constant
 * @returns {Eta} Clipped site
 */
export function spectralClipSite(site, othersPrior, cap = 3.0) {
  const ratio = site.Lam[0] / othersPrior.Lam[0]; // Both negative, ratio positive
  const clippedRatio = Math.min(ratio, cap);

  const clippedLam = othersPrior.Lam[0] * clippedRatio;
  const clippedH = site.h[0] * (clippedLam / site.Lam[0]);

  return new Eta(new Float64Array([clippedH]), new Float64Array([clippedLam]));
}
