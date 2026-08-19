# FIOR CLI

Command-line interface for interacting with the FIOR protocol.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Commands](#commands)
   - [config](#config)
   - [fetch](#fetch)
   - [compose](#compose)
   - [score](#score)
   - [publish](#publish)
   - [blob](#blob)
   - [trust](#trust)
4. [Examples](#examples)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### From PyPI

```bash
pip install fior
```

### From source

```bash
git clone https://github.com/nostriai/fior.git
cd fior/fior-python
pip install -e .
```

### Standalone binary

Download from releases and make executable:

```bash
chmod +x fior
./fior --help
```

---

## Quick Start

```bash
# Initialize config
fior config init

# Edit config to add your private key
nano ~/.config/fior/config.toml

# Fetch a model
fior fetch model pump-failure-v1

# List trust table
fior trust list
```

---

## Commands

### config

Manage FIOR configuration.

```bash
# Initialize config file
fior config init

# Show current config
fior config show

# Set config value
fior config set relay.url wss://nos.lol
fior config set blossom.server https://blossom.example.com
```

### fetch

Fetch data from relay.

```bash
# Fetch model card
fior fetch model <model_id> [--relay <url>] [--json]

# Fetch sites for a model
fior fetch sites <model_id> [--relay <url>] [--json]

# Fetch attestations for a peer
fior fetch attestations <pubkey> [--relay <url>] [--json]
```

### compose

Compose prior from trusted peers.

```bash
# Compose with default peers
fior compose <model_id> [--relay <url>] [--json]

# Compose with specific peers
fior compose <model_id> --peers <pubkey1,pubkey2,...> [--relay <url>] [--json]
```

### score

Score peers using BMR.

```bash
# Score all peers
fior score <model_id> [--relay <url>] [--json]

# Score specific peer
fior score <model_id> --peer <pubkey> [--relay <url>] [--json]
```

### publish

Publish site or attestation.

```bash
# Publish site contribution
fior publish site <model_id> --data <blob_path> [--relay <url>] [--json]

# Publish trust attestation
fior publish attestation <target_pubkey> --p <value> [--relay <url>] [--json]
```

### blob

Manage Blossom blobs.

```bash
# Upload blob
fior blob upload <file> --server <blossom_url> [--json]

# Fetch blob
fior blob fetch <sha256> --server <blossom_url> [--output <path>] [--json]
```

### trust

Manage trust table.

```bash
# List trust entries
fior trust list [--model <model_id>] [--json]

# Set trust value
fior trust set <pubkey> <group> <p_value>

# Reset trust values
fior trust reset [<pubkey>] [--group <group>]
```

---

## Examples

### Fetch and display model

```bash
$ fior fetch model pump-failure-v1

┌─────────────────────────────────────┐
│ Model: pump-failure-v1              │
│ Title: Pump Failure Predictor       │
│ Version: 3                          │
│ Groups: fc-layers, convs            │
└─────────────────────────────────────┘
```

### JSON output for scripting

```bash
$ fior fetch model pump-failure-v1 --json
{
  "id": "pump-failure-v1",
  "title": "Pump Failure Predictor",
  "version": 3,
  "groups": ["fc-layers", "convs"]
}
```

### Trust table

```bash
$ fior trust list

┌────────────────┬───────────┬───────┐
│ Peer           │ Group     │ p     │
├────────────────┼───────────┼───────┤
│ abc123...      │ fc-layers │ 0.92  │
│ def456...      │ convs     │ 0.75  │
└────────────────┴───────────┴───────┘
```

---

## Configuration

Config file location: `~/.config/fior/config.toml`

```toml
# FIOR Configuration

[relay]
url = "wss://relay.example.com"

[blossom]
server = "https://blossom.example.com"

[identity]
# private_key = "your_private_key_hex_here"

[trust]
kappa_r = "4"
```

### Key Management

- NIP-07 doesn't apply to CLI
- Generate key elsewhere (e.g., `nsec` tool)
- Place private key in config file
- CLI signs events using this key

---

## Troubleshooting

### Config not found

```bash
fior config init
```

### Connection errors

Check relay URL in config:

```bash
fior config set relay.url wss://nos.lol
```

### Permission errors

Ensure config directory exists:

```bash
mkdir -p ~/.config/fior
```

---

## Output Formats

### Human-readable (default)

Pretty printed tables with colors.

### Machine-readable (--json)

JSON output for scripting and automation.

### Plain output (--no-color)

Disable colors for piping or logging.

---

## Dependencies

- typer - CLI framework
- rich - Pretty printing
- tomli - TOML config parsing
- noble-secp256k1 - Cryptographic signing
- requests - HTTP client
- websockets - WebSocket client
