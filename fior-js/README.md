# FIOR JavaScript Library

JavaScript implementation of the FIOR protocol for federated inference over Nostr relays.

## Installation

```bash
npm install fior
```

## Usage

```javascript
import {
  Eta,
  composePrior,
  bmrDeltaF,
  pFrom,
  TrustTable,
  signEvent,
} from "fior";

// Create natural parameters
const eta0 = new Eta(new Float64Array([0.0]), new Float64Array([-1.0]));

// Compose prior from sites
const prior = composePrior(eta0, sites, [0.9, 0.7]);

// Compute BMR log Bayes factor
const deltaF = bmrDeltaF(etaQPlus, etaPMinus, etaQMinus, etaPPlus);

// Resolve inclusion probability
const p = pFrom(deltaF, 0.5);

// Manage trust
const trustTable = new TrustTable();
trustTable.setP("peer1", "fc-layers", 0.85);
```

## API

### Types

- `Eta` - Natural parameters
- `Group` - Named set of initializers
- `ModelCard` - Model definition
- `Site` - Published contribution
- `Attestation` - Trust rating

### Math

- `logPartition(eta, family)` - Log-partition function A(η)
- `composePrior(eta0, sites, pValues)` - Build local prior
- `bmrDeltaF(...)` - BMR log Bayes factor
- `pFrom(deltaF, beta)` - Resolve inclusion probability

### Trust

- `TrustTable` - Per-peer p values
- `Corroboration` - One-peer-one-vote scoring
- `spectralClipSite(site, others, cap)` - Loewner cap

### Nostr

- `Event` - Signed event class
- `signEvent(event, privateKey)` - Sign with BIP-340
- `createModelCardEvent(...)` - Create 30100
- `createSiteContributionEvent(...)` - Create 30101
- `createTrustAttestationEvent(...)` - Create 30102
- `publishEvent(event, relayUrl)` - Publish to relay

### Blob

- `computeSha256(data)` - Hash data
- `uploadBlob(data, server)` - Upload to Blossom
- `fetchBlob(sha256, servers)` - Fetch from Blossom
- `encodeEtaBlob(groups)` - Encode η to binary
- `decodeEtaBlob(data, dims)` - Decode binary to η
