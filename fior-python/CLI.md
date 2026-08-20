# FIOR CLI

Command-line interface for the FIOR protocol, backed by `fior.client`. Every
command talks to a real relay and Blossom server; there are no placeholders.

## Install

```sh
pip install .            # from fior-python/, or `pip install fior` once published
fior --help
```

Requires Python 3.10+. The `fior` console script and `python -m fior.cli.main`
are equivalent entry points.

## Configuration

Config file: `~/.config/fior/config.toml` (managed by the CLI itself).

```sh
fior config init
fior config set relay.url wss://relay.example.com
fior config set blossom.server https://blossom.example.com
fior config set identity.private_key <hex secret key>
fior config set trust.kappa_r 4
fior config show --json
```

`kappa_r` is the reader-side attestation weight; the protocol recommends 1..4.
The private key lives only in this file and signs everything you publish.

## Commands

### fetch

```sh
fior fetch model <model-id> [--relay URL] [--json]
fior fetch sites <model-id> [--relay URL] [--json]
fior fetch attestations <pubkey> [--relay URL] [--json]
```

`fetch model` prints the parsed model card (groups, families, dims). `fetch
sites` lists the current site descriptors (one per author), no parameter
bytes. `fetch attestations` lists live attestations about a peer.

### compose

```sh
fior compose <model-id> [--relay URL] [--json]
```

Fetches the model card, its sites, and each site's Δη blob, then builds the
local prior `η0 + Σ p·Δη` from the current trust table. Raises (exit 1) if a
composed group leaves its natural parameter domain.

### score

```sh
fior score <model-id> --likelihood lik.json [--rounds N] [--relay URL] [--json]
```

Scores every peer per group with BMR at unit weight. `lik.json` is the caller's
local likelihood contribution per group, e.g.:

```json
{
  "coefs": {"h": [0.1, -0.2], "lam": [[-0.5, 0.0], [0.0, -0.5]]}
}
```

`--rounds` repeats the fixed-point iteration. Results are stored in the local
trust table (`~/.config/fior/trust.json`).

### publish

```sh
fior publish model <spec.json>               # model card
fior publish site --model <id> --delta delta.json \
      [--members members.json] [--encoding i8|i16le|f32le|f64le] [--relay URL]
fior publish attestation --peer <pubkey> --p 0.8 \
      [--model <id>] [--group <group>] [--relay URL]
```

- `publish model`: `spec.json` is `{id, title, version, groups:
  [{name, family, initials, dim}], onnx: {sha, size, servers},
  blossom_servers}`.
- `publish site`: `delta.json` holds one `{h, lam}` per model group. The blob
  is uploaded to the configured Blossom server(s) **before** the event is
  published, exactly as the protocol requires.
- `publish attestation`: one scalar `p` in (0,1).

### blob

```sh
fior blob upload <file> [--server URL] [--json]
fior blob fetch <sha256> [--server URL] [-o out.bin]
```

### trust

```sh
fior trust list [--json]
fior trust set <peer> <group> <p>
fior trust reset [<peer>] [--group <group>]
```

The trust table is stored at `~/.config/fior/trust.json`.

### withdraw

```sh
fior withdraw <model-id> [--relay URL]
```

Publishes a NIP-09 deletion for your site on the model.

## Output

Human-readable by default; `--json` on every command emits machine-readable
JSON; `--no-color` disables rich styling.

## Troubleshooting

- "no Blossom server configured": set one with `fior config set blossom.server
  URL` or pass `--server`.
- "read the model card first": load the card with `fetch model` / `compose`
  before operating on sites, or pass `initializer_dims` in code.
- "composed group ... left its natural parameter domain": the protocol defines
  no recovery; drop a member or refetch at a wider encoding.