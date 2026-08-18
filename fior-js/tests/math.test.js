/**
 * Tests for FIOR math module.
 */

import { describe, it } from "node:test";
import assert from "node:assert";
import { Eta, FAMILY_NORMAL } from "../src/types.js";
import { logPartition, composePrior, bmrDeltaF, pFrom } from "../src/math.js";

describe("logPartition", () => {
  it("computes A(η) for Normal family", () => {
    const eta = new Eta(new Float64Array([1.0]), new Float64Array([-0.5]));
    const result = logPartition(eta, FAMILY_NORMAL);
    // A(η) = -h²/(4*Lam) - 0.5*log(-Lam)
    const expected = -1.0 / (4 * -0.5) - 0.5 * Math.log(0.5);
    assert(Math.abs(result - expected) < 1e-10);
  });

  it("computes A(η) for multi-dimensional Normal", () => {
    const eta = new Eta(new Float64Array([1.0, 2.0]), new Float64Array([-0.5, -1.0]));
    const result = logPartition(eta, FAMILY_NORMAL);
    const expected =
      -1.0 / (4 * -0.5) - 0.5 * Math.log(0.5) + (-4.0) / (4 * -1.0) - 0.5 * Math.log(1.0);
    assert(Math.abs(result - expected) < 1e-10);
  });
});

describe("composePrior", () => {
  it("returns eta0 with no sites", () => {
    const eta0 = new Eta(new Float64Array([0.0]), new Float64Array([-1.0]));
    const result = composePrior(eta0, [], []);
    assert.deepStrictEqual(Array.from(result.h), [0.0]);
    assert.deepStrictEqual(Array.from(result.Lam), [-1.0]);
  });

  it("composes with single site at full weight", () => {
    const eta0 = new Eta(new Float64Array([0.0]), new Float64Array([-1.0]));
    const site = { deltaEta: new Eta(new Float64Array([1.0]), new Float64Array([-0.5])) };
    const result = composePrior(eta0, [site], [1.0]);
    assert.deepStrictEqual(Array.from(result.h), [1.0]);
    assert.deepStrictEqual(Array.from(result.Lam), [-1.5]);
  });

  it("composes with single site at half weight", () => {
    const eta0 = new Eta(new Float64Array([0.0]), new Float64Array([-1.0]));
    const site = { deltaEta: new Eta(new Float64Array([2.0]), new Float64Array([-2.0])) };
    const result = composePrior(eta0, [site], [0.5]);
    assert.deepStrictEqual(Array.from(result.h), [1.0]);
    assert.deepStrictEqual(Array.from(result.Lam), [-2.0]);
  });
});

describe("bmrDeltaF", () => {
  it("returns 0 for identical posteriors", () => {
    const eta = new Eta(new Float64Array([1.0]), new Float64Array([-0.5]));
    const deltaF = bmrDeltaF(eta, eta, eta, eta, FAMILY_NORMAL);
    assert(Math.abs(deltaF) < 1e-10);
  });
});

describe("pFrom", () => {
  it("returns beta when deltaF is 0", () => {
    const p = pFrom(0.0, 0.5);
    assert(Math.abs(p - 0.5) < 1e-10);
  });

  it("returns p > beta when deltaF > 0", () => {
    const p = pFrom(2.0, 0.5);
    assert(p > 0.5);
  });

  it("returns p < beta when deltaF < 0", () => {
    const p = pFrom(-2.0, 0.5);
    assert(p < 0.5);
  });

  it("returns p close to 1 for large positive deltaF", () => {
    const p = pFrom(100.0, 0.5);
    assert(p > 0.99);
  });

  it("returns p close to 0 for large negative deltaF", () => {
    const p = pFrom(-100.0, 0.5);
    assert(p < 0.01);
  });
});
