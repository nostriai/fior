/**
 * Blossom blob upload and download for FIOR.
 */

import { sha256 } from "@noble/hashes/sha256";

/**
 * Compute SHA-256 hash of data.
 * @param {Uint8Array} data
 * @returns {string} Hex string
 */
export function computeSha256(data) {
  const hash = sha256(data);
  return Buffer.from(hash).toString("hex");
}

/**
 * Upload a blob to a Blossom server.
 * @param {Uint8Array} data - Raw bytes
 * @param {string} serverUrl - Blossom server root URL
 * @param {string} [sha256Hex] - Optional pre-computed hash
 * @returns {Promise<string>} SHA-256 hash
 */
export async function uploadBlob(data, serverUrl, sha256Hex = null) {
  if (!sha256Hex) {
    sha256Hex = computeSha256(data);
  }

  const url = `${serverUrl}/upload`;
  const response = await fetch(url, {
    method: "POST",
    body: data,
    headers: { "Content-Type": "application/octet-stream" },
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }

  return sha256Hex;
}

/**
 * Fetch a blob from Blossom servers.
 * @param {string} sha256Hex - SHA-256 hash
 * @param {string[]} serverUrls - Server URLs to try
 * @param {boolean} verify - Whether to verify SHA-256
 * @returns {Promise<Uint8Array>}
 */
export async function fetchBlob(sha256Hex, serverUrls, verify = true) {
  let lastError = null;

  for (const serverUrl of serverUrls) {
    try {
      const url = `${serverUrl}/${sha256Hex}`;
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = new Uint8Array(await response.arrayBuffer());

      if (verify) {
        const computed = computeSha256(data);
        if (computed !== sha256Hex) {
          throw new Error(`SHA-256 mismatch: expected ${sha256Hex}, got ${computed}`);
        }
      }

      return data;
    } catch (e) {
      lastError = e;
      continue;
    }
  }

  throw new Error(`Failed to fetch blob from any server: ${lastError}`);
}

/**
 * Encode natural parameters to binary blob.
 * @param {Array} etaGroups - List of [name, Eta] tuples
 * @param {string} encoding - Blob encoding
 * @returns {Uint8Array}
 */
export function encodeEtaBlob(etaGroups, encoding = "f64le") {
  if (encoding !== "f64le") {
    throw new Error(`Encoding ${encoding} not yet implemented`);
  }

  // Header: group_count (u16le)
  const headerSize = 2 + etaGroups.length; // 2 bytes + 1 byte per group (scale_count)
  const bodySize = etaGroups.reduce((sum, [, eta]) => sum + eta.dim * 16, 0); // h + Lam

  const buffer = new ArrayBuffer(headerSize + bodySize);
  const view = new DataView(buffer);
  const uint8 = new Uint8Array(buffer);

  let offset = 0;

  // group_count
  view.setUint16(offset, etaGroups.length, true);
  offset += 2;

  // Each group
  for (const [, eta] of etaGroups) {
    // scale_count = 0 for float encodings
    uint8[offset] = 0;
    offset += 1;
  }

  // Body
  for (const [, eta] of etaGroups) {
    // h values
    const hBytes = Buffer.from(eta.h.buffer);
    uint8.set(hBytes, offset);
    offset += eta.dim * 8;

    // Lam values
    const lamBytes = Buffer.from(eta.Lam.buffer);
    uint8.set(lamBytes, offset);
    offset += eta.dim * 8;
  }

  return uint8;
}

/**
 * Decode binary blob to natural parameters.
 * @param {Uint8Array} data - Binary blob
 * @param {number[]} groupDims - Dimensionality of each group
 * @param {string} encoding - Blob encoding
 * @returns {Array} List of [name, Eta] tuples
 */
export function decodeEtaBlob(data, groupDims, encoding = "f64le") {
  if (encoding !== "f64le") {
    throw new Error(`Encoding ${encoding} not yet implemented`);
  }

  const view = new DataView(data.buffer);
  let offset = 0;

  // group_count
  const groupCount = view.getUint16(offset, true);
  offset += 2;

  const result = [];

  for (let i = 0; i < groupCount; i++) {
    // scale_count (skip)
    offset += 1;

    // Skip scales
    const scaleCount = data[offset - 1];
    offset += scaleCount * 8;

    const dim = groupDims[i] ?? groupDims[groupDims.length - 1];

    // h values
    const h = new Float64Array(data.buffer, offset, dim);
    offset += dim * 8;

    // Lam values
    const Lam = new Float64Array(data.buffer, offset, dim);
    offset += dim * 8;

    result.push([`group_${i}`, new Eta(new Float64Array(h), new Float64Array(Lam))]);
  }

  return result;
}
