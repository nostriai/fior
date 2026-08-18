/**
 * Core data types for FIOR.
 */

// Event kind constants
export const KIND_MODEL_CARD = 30100;
export const KIND_SITE_CONTRIBUTION = 30101;
export const KIND_TRUST_ATTESTATION = 30102;

// Distribution family constants
export const FAMILY_NORMAL = "normal";
export const FAMILY_GAMMA = "gamma";
export const FAMILY_BETA = "beta";
export const FAMILY_DIRICHLET = "dirichlet";
export const FAMILY_CATEGORICAL = "cat";

// Blob encoding constants
export const ENCODING_F64LE = "f64le";
export const ENCODING_F32LE = "f32le";
export const ENCODING_I16LE = "i16le";
export const ENCODING_I8 = "i8";

/**
 * Natural parameters for an exponential family distribution.
 */
export class Eta {
  /**
   * @param {Float64Array} h - η₁ (mean parameters)
   * @param {Float64Array} Lam - η₂ (precision parameters)
   */
  constructor(h, Lam) {
    this.h = h instanceof Float64Array ? h : new Float64Array(h);
    this.Lam = Lam instanceof Float64Array ? Lam : new Float64Array(Lam);
    
    if (this.h.length !== this.Lam.length) {
      throw new Error("h and Lam must have same length");
    }
  }

  get dim() {
    return this.h.length;
  }

  copy() {
    return new Eta(new Float64Array(this.h), new Float64Array(this.Lam));
  }

  add(other) {
    const h = new Float64Array(this.dim);
    const Lam = new Float64Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      h[i] = this.h[i] + other.h[i];
      Lam[i] = this.Lam[i] + other.Lam[i];
    }
    return new Eta(h, Lam);
  }

  sub(other) {
    const h = new Float64Array(this.dim);
    const Lam = new Float64Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      h[i] = this.h[i] - other.h[i];
      Lam[i] = this.Lam[i] - other.Lam[i];
    }
    return new Eta(h, Lam);
  }

  mul(scalar) {
    const h = new Float64Array(this.dim);
    const Lam = new Float64Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      h[i] = this.h[i] * scalar;
      Lam[i] = this.Lam[i] * scalar;
    }
    return new Eta(h, Lam);
  }

  impliedMean() {
    const mean = new Float64Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      mean[i] = -this.h[i] / (2 * this.Lam[i]);
    }
    return mean;
  }

  impliedVariance() {
    const variance = new Float64Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      variance[i] = -0.5 / this.Lam[i];
    }
    return variance;
  }

  inDomain(family) {
    if (family === FAMILY_NORMAL) {
      return this.Lam.every((l) => l < 0);
    }
    // Add other families as needed
    return true;
  }
}

/**
 * Named set of initializers sharing a distribution.
 */
export class Group {
  constructor(name, family, initializers) {
    this.name = name;
    this.family = family;
    this.initializers = initializers;
  }

  get dim() {
    if ([FAMILY_NORMAL, FAMILY_GAMMA, FAMILY_BETA].includes(this.family)) {
      return this.initializers.length;
    }
    return this.initializers.length;
  }
}

/**
 * Model definition.
 */
export class ModelCard {
  constructor({ id, title, version, groups, eta0, onnxBlob, blossomServers = [], summary = null }) {
    this.id = id;
    this.title = title;
    this.version = version;
    this.groups = groups;
    this.eta0 = eta0;
    this.onnxBlob = onnxBlob;
    this.blossomServers = blossomServers;
    this.summary = summary;
  }
}

/**
 * Published contribution.
 */
export class Site {
  constructor({ author, modelId, modelVersion, deltaEta, cavityMembers = [], eventId = null, samples = null, freeEnergy = null, durationSec = null }) {
    this.author = author;
    this.modelId = modelId;
    this.modelVersion = modelVersion;
    this.deltaEta = deltaEta;
    this.cavityMembers = cavityMembers;
    this.eventId = eventId;
    this.samples = samples;
    this.freeEnergy = freeEnergy;
    this.durationSec = durationSec;
  }
}

/**
 * Trust rating for a peer.
 */
export class Attestation {
  constructor({ target, scope, p, eventId = null }) {
    this.target = target;
    this.scope = scope;
    this.p = p;
    this.eventId = eventId;
  }
}
