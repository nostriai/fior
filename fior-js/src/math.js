/**
 * Core mathematical functions for FIOR.
 */

import { Eta, FAMILY_NORMAL } from "./types.js";

/**
 * Compute log-partition function A(η).
 * @param {Eta} eta - Natural parameters
 * @param {string} family - Distribution family
 * @returns {number} Log-partition function value
 */
export function logPartition(eta, family = FAMILY_NORMAL) {
  const { h, Lam } = eta;

  if (family === FAMILY_NORMAL) {
    // A(η) = -h²/(4*Lam) - 0.5 * log(-Lam)
    let sum = 0;
    for (let i = 0; i < eta.dim; i++) {
      sum += (-h[i] * h[i]) / (4 * Lam[i]) - 0.5 * Math.log(-Lam[i]);
    }
    return sum;
  }

  throw new Error(`Unknown family: ${family}`);
}

/**
 * Build a local prior from trusted peers.
 * η_prior = η₀ + Σ p_n · Δη_n
 * @param {Eta} eta0 - Base prior
 * @param {Site[]} sites - List of Site objects
 * @param {number[]} pValues - Inclusion probabilities
 * @returns {Eta} Composed prior
 */
export function composePrior(eta0, sites, pValues) {
  let result = eta0.copy();

  for (let i = 0; i < sites.length; i++) {
    const site = sites[i];
    const p = pValues[i];
    result = result.add(site.deltaEta.mul(p));
  }

  return result;
}

/**
 * Compute log Bayes factor from BMR.
 * ΔF = A(η_q^+) + A(η_p^-) - A(η_q^-) - A(η_p^+)
 * @param {Eta} etaQPlus - Prior without peer + local data
 * @param {Eta} etaPMinus - Prior without peer
 * @param {Eta} etaQMinus - Prior without peer + local data
 * @param {Eta} etaPPlus - Prior with peer at full weight
 * @param {string} family - Distribution family
 * @returns {number} Log Bayes factor
 */
export function bmrDeltaF(etaQPlus, etaPMinus, etaQMinus, etaPPlus, family = FAMILY_NORMAL) {
  return (
    logPartition(etaQPlus, family) +
    logPartition(etaPMinus, family) -
    logPartition(etaQMinus, family) -
    logPartition(etaPPlus, family)
  );
}

/**
 * Convert ΔF and prior belief to inclusion probability.
 * p = σ(ΔF + ln(β/(1-β)))
 * @param {number} deltaF - Log Bayes factor from BMR
 * @param {number} beta - Prior inclusion probability
 * @returns {number} Inclusion probability p in (0, 1)
 */
export function pFrom(deltaF, beta) {
  // Clamp beta to (0, 1) exclusive
  const clampedBeta = Math.max(1e-10, Math.min(1 - 1e-10, beta));

  // Compute log-odds of beta
  const logOddsBeta = Math.log(clampedBeta / (1 - clampedBeta));

  // Compute p via logistic
  const x = deltaF + logOddsBeta;
  return 1 / (1 + Math.exp(-x));
}

/**
 * Logistic sigmoid function.
 * @param {number} x
 * @returns {number}
 */
export function logistic(x) {
  return 1 / (1 + Math.exp(-x));
}
