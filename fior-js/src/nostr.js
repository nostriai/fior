/**
 * Nostr event construction and signing for FIOR.
 */

import { sha256 } from "@noble/hashes/sha256";
import * as secp from "@noble/secp256k1";

/**
 * Signed Nostr event.
 */
export class Event {
  constructor({ kind, tags, content, createdAt, pubkey, id = "", sig = "" }) {
    this.kind = kind;
    this.tags = tags;
    this.content = content;
    this.createdAt = createdAt;
    this.pubkey = pubkey;
    this.id = id;
    this.sig = sig;
  }

  toDict() {
    return {
      id: this.id,
      pubkey: this.pubkey,
      created_at: this.createdAt,
      kind: this.kind,
      tags: this.tags,
      content: this.content,
      sig: this.sig,
    };
  }

  serialize() {
    return JSON.stringify([
      0,
      this.pubkey,
      this.createdAt,
      this.kind,
      this.tags,
      this.content,
    ]);
  }
}

/**
 * Compute event ID (SHA-256 of serialized event).
 * @param {Event} event
 * @returns {string} Hex string
 */
export function computeEventId(event) {
  const serialized = event.serialize();
  const hash = sha256(new TextEncoder().encode(serialized));
  return Buffer.from(hash).toString("hex");
}

/**
 * Sign a Nostr event with BIP-340.
 * @param {Event} event - Unsigned event
 * @param {string} privateKey - Private key as hex
 * @returns {Event} Signed event
 */
export function signEvent(event, privateKey) {
  event.id = computeEventId(event);
  const sig = secp.sign(event.id, privateKey);
  event.sig = sig.toCompactHex();
  return event;
}

/**
 * Create a Model Card event (30100).
 */
export function createModelCardEvent({
  privateKey,
  modelId,
  title,
  version,
  onnxBlob,
  groups,
  eta0Blob = null,
  blossomServers = [],
  summary = null,
  ttl = null,
}) {
  const pubkey = secp.getPublicKey(privateKey, true);

  const tags = [
    ["d", modelId],
    ["t", title],
    ["v", String(version)],
    ["o", onnxBlob],
  ];

  if (summary) tags.push(["s", summary]);

  for (const [name, family, ...initializers] of groups) {
    tags.push(["g", name, family, ...initializers]);
  }

  if (eta0Blob) tags.push(["x", eta0Blob]);
  if (blossomServers.length > 0) tags.push(["b", ...blossomServers]);
  if (ttl) tags.push(["l", String(ttl)]);

  const event = new Event({
    kind: 30100,
    tags,
    content: "{}",
    createdAt: Math.floor(Date.now() / 1000),
    pubkey,
  });

  return signEvent(event, privateKey);
}

/**
 * Create a Site Contribution event (30101).
 */
export function createSiteContributionEvent({
  privateKey,
  modelId,
  modelVersion,
  deltaEtaBlob,
  cavityMembers = [],
  samples = null,
  freeEnergy = null,
  durationSec = null,
  expiration = null,
}) {
  const pubkey = secp.getPublicKey(privateKey, true);

  const tags = [
    ["d", modelId],
    ["a", `30100:${pubkey}:${modelId}`, "", "model"],
    ["v", String(modelVersion)],
    ["x", deltaEtaBlob],
  ];

  for (const [memberPubkey, memberEventId, pVector] of cavityMembers) {
    tags.push(["m", memberPubkey, memberEventId, pVector]);
    tags.push(["p", memberPubkey]);
  }

  if (expiration) tags.push(["E", String(expiration)]);

  const content = {};
  if (samples !== null) content.samples = samples;
  if (freeEnergy !== null) content.free_energy = freeEnergy;
  if (durationSec !== null) content.duration_sec = durationSec;

  const event = new Event({
    kind: 30101,
    tags,
    content: JSON.stringify(content),
    createdAt: Math.floor(Date.now() / 1000),
    pubkey,
  });

  return signEvent(event, privateKey);
}

/**
 * Create a Trust Attestation event (30102).
 */
export function createTrustAttestationEvent({
  privateKey,
  targetPubkey,
  scope,
  inclusionProbability,
  modelRef = null,
  expiration = null,
}) {
  const pubkey = secp.getPublicKey(privateKey, true);

  const dValue = scope === "peer" ? targetPubkey : `${targetPubkey}:${scope}`;

  const tags = [
    ["d", dValue],
    ["p", targetPubkey],
    ["i", String(inclusionProbability)],
  ];

  if (modelRef) tags.push(["a", modelRef, "", "model"]);
  if (expiration) tags.push(["E", String(expiration)]);

  const event = new Event({
    kind: 30102,
    tags,
    content: "{}",
    createdAt: Math.floor(Date.now() / 1000),
    pubkey,
  });

  return signEvent(event, privateKey);
}

/**
 * Publish event to a Nostr relay via WebSocket.
 * @param {Event} event - Signed event
 * @param {string} relayUrl - WebSocket URL
 * @returns {Promise<boolean>}
 */
export async function publishEvent(event, relayUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(relayUrl);

    ws.onopen = () => {
      const message = JSON.stringify(["EVENT", event.toDict()]);
      ws.send(message);
    };

    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data);
      ws.close();
      resolve(data[0] === "OK");
    };

    ws.onerror = (err) => {
      reject(err);
    };
  });
}
