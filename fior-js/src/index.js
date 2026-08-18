/**
 * FIOR - Federated Inference Over Relays
 *
 * A Nostr protocol for collaborative machine learning without sharing raw data.
 */

// Types
export {
  Eta,
  Group,
  ModelCard,
  Site,
  Attestation,
  KIND_MODEL_CARD,
  KIND_SITE_CONTRIBUTION,
  KIND_TRUST_ATTESTATION,
  FAMILY_NORMAL,
  FAMILY_GAMMA,
  FAMILY_BETA,
  FAMILY_DIRICHLET,
  FAMILY_CATEGORICAL,
  ENCODING_F64LE,
  ENCODING_F32LE,
  ENCODING_I16LE,
  ENCODING_I8,
} from "./types.js";

// Math
export {
  logPartition,
  composePrior,
  bmrDeltaF,
  pFrom,
  logistic,
} from "./math.js";

// Trust
export {
  TrustTable,
  Corroboration,
  spectralClipSite,
} from "./trust.js";

// Nostr
export {
  Event,
  computeEventId,
  signEvent,
  createModelCardEvent,
  createSiteContributionEvent,
  createTrustAttestationEvent,
  publishEvent,
} from "./nostr.js";

// Blob
export {
  computeSha256,
  uploadBlob,
  fetchBlob,
  encodeEtaBlob,
  decodeEtaBlob,
} from "./blob.js";
