"""Blossom blob upload/download and the natural-parameter blob codec.

Implements the protocol.md blob layout: a u16 group count, then per group a
u32 scale count plus scales, then groups concatenated in model card order.
Group value counts derive from the model card's group dimensions; a decoder
fails loudly on any mismatch rather than misparsing.
"""

import hashlib
import struct
from typing import Optional

import numpy as np

from .types import (
    Eta,
    Group,
    ONE_BLOCK_FAMILIES,
    INT_ENCODINGS,
    ENCODING_F64LE,
    ENCODING_F32LE,
    ENCODING_I16LE,
    ENCODING_I8,
    value_count,
)

_BYTES_PER_VALUE = {ENCODING_F64LE: 8, ENCODING_F32LE: 4, ENCODING_I16LE: 2, ENCODING_I8: 1}
_INT_RANGE = {ENCODING_I16LE: 32767, ENCODING_I8: 127}


class BlobError(ValueError):
    """A malformed or mismatched blob; carries the reason for display."""


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def upload_blob(data: bytes, server_url: str, sha256: Optional[str] = None) -> str:
    """Upload bytes to a Blossom server (BUD-02) and return their SHA-256."""
    import requests

    if sha256 is None:
        sha256 = compute_sha256(data)
    response = requests.post(
        f"{server_url}/upload",
        data=data,
        headers={"Content-Type": "application/octet-stream"},
        timeout=60,
    )
    response.raise_for_status()
    return sha256


def fetch_blob(
    sha256: str,
    server_urls: list[str],
    verify: bool = True,
    timeout: float = 30.0,
) -> bytes:
    """Fetch from the first server that serves the bytes; verify the hash.

    Raises RuntimeError if no server serves it.  Callers treat that as "the
    member is absent from composition", never as a protocol error.
    """
    import requests

    last_error: Optional[Exception] = None
    for server_url in server_urls:
        try:
            response = requests.get(f"{server_url}/{sha256}", timeout=timeout)
            response.raise_for_status()
            data = response.content
            if verify and compute_sha256(data) != sha256:
                raise ValueError(f"SHA-256 mismatch on {server_url}: wrong bytes")
            return data
        except Exception as e:  # noqa: BLE001 - try the next server
            last_error = e
    raise RuntimeError(f"blob {sha256[:16]}... unresolvable from any server: {last_error}")


# --------------------------------------------------------------------------
# Natural-parameter wire packing
# --------------------------------------------------------------------------


def _blocks(eta: Eta) -> list[np.ndarray]:
    """The group's natural-parameter index blocks, in wire order.

    `normal`/`gamma`/`beta`: [η1, η2]; `mvnormal`: [η1, packed η2 triangle];
    `dirichlet`/`cat`: [η].  Each block carries one quantization scale.
    """
    if eta.family == "mvnormal":
        tri = np.tril_indices(eta.dim)
        return [eta.h, eta.Lam[tri]]
    if eta.family in ONE_BLOCK_FAMILIES:
        return [eta.h]
    return [eta.h, eta.Lam]


def _unpack_values(family: str, dim: int, flat: np.ndarray) -> Eta:
    if family == "mvnormal":
        h = flat[:dim]
        tri = np.tril_indices(dim)
        lam = np.zeros((dim, dim))
        lam[tri] = flat[dim:]
        lam = lam + lam.T - np.diag(np.diag(lam))
        return Eta(family, h, lam)
    count = value_count(family, dim)
    if len(flat) != count:
        raise BlobError(
            f"group value count {len(flat)} != expected {count} for family {family}"
        )
    if family in ONE_BLOCK_FAMILIES:
        return Eta(family, flat)
    return Eta(family, flat[:dim], flat[dim:])


def _block_scales(family: str) -> int:
    """Number of quantization blocks a group's family carries."""
    from .types import FAMILY_MVNORMAL, ONE_BLOCK_FAMILIES

    if family == FAMILY_MVNORMAL or family not in ONE_BLOCK_FAMILIES:
        return 2
    return 1


def _block_sizes(group: Group) -> list[int]:
    """Per-block value counts, for dequantization of integer encodings."""
    if group.family in ONE_BLOCK_FAMILIES:
        return [group.dim]
    total = value_count(group.family, group.dim)
    return [group.dim, total - group.dim]


def _stochastic_round(x: float, rng: np.random.Generator) -> int:
    """Unbiased rounding: integer floor plus a Bernoulli on the fraction."""
    fl = int(np.floor(x))
    frac = float(x) - fl
    return fl + (1 if rng.random() < frac else 0)


# --------------------------------------------------------------------------
# Codec
# --------------------------------------------------------------------------


def encode_eta_blob(
    groups: list[Group],
    etas: list[tuple[str, Eta]],
    encoding: str = ENCODING_F64LE,
    seed: Optional[int] = None,
) -> bytes:
    """Encode groups in model card order to a blob.

    `etas` must be one (name, Eta) per model card group, in the same order and
    with the same family and dimension; a mismatch fails loudly instead of
    producing a blob that decodes into a different shape.  `encoding` selects
    the numeric type.  Integer encodings carry one f64 scale per
    natural-parameter index (protocol.md: per-index scaling is required) and
    use stochastic rounding, which federated summation depends on for
    unbiasedness.  `seed` makes the rounding reproducible in tests.
    """
    if encoding not in _BYTES_PER_VALUE:
        raise ValueError(f"unknown encoding: {encoding}")
    if len(groups) != len(etas):
        raise BlobError(
            f"{len(etas)} eta groups supplied for a model card with {len(groups)} groups"
        )
    for group, (name, eta) in zip(groups, etas):
        if name != group.name or group.family != eta.family:
            raise BlobError(
                f"group order/family mismatch: card group {group.name}:{group.family} "
                f"vs supplied {name}:{eta.family}"
            )
        if value_count(group.family, group.dim) != eta.size:
            raise BlobError(
                f"group {name}: card value count {value_count(group.family, group.dim)} "
                f"!= supplied params {eta.size}"
            )

    byte_width = _BYTES_PER_VALUE[encoding]
    rng = np.random.default_rng(seed)

    header_parts = [struct.pack("<H", len(groups))]
    body_parts: list[bytes] = []

    for group, (name, eta) in zip(groups, etas):
        blocks = _blocks(eta)
        if encoding in INT_ENCODINGS:
            qmax = _INT_RANGE[encoding]
            scales: list[float] = []
            data = bytearray()
            for block in blocks:
                scale = float(np.max(np.abs(block))) if block.size else 0.0
                if scale < 1e-300:
                    scale = 1.0
                scales.append(scale / qmax)
                for v in block:
                    q = _stochastic_round(v / (scale / qmax), rng)
                    data += struct.pack("<b" if encoding == ENCODING_I8 else "<h", q)
            header_parts.append(struct.pack("<I", len(scales)))
            header_parts.append(b"".join(struct.pack("<d", s) for s in scales))
            body_parts.append(bytes(data))
        else:
            header_parts.append(struct.pack("<I", 0))
            flat = np.concatenate(blocks)
            if encoding == ENCODING_F64LE:
                body_parts.append(flat.astype("<f8").tobytes())
            else:
                body_parts.append(flat.astype("<f4").tobytes())

    return b"".join(header_parts + body_parts)


def _read_blob_header(
    data: bytes, expected: int, encoding: str, byte_width: int
) -> tuple[list[list[float]], int]:
    """Parse the header; returns (per-group scales, body_offset)."""
    if len(data) < 2:
        raise BlobError("blob too short for group_count")
    group_count = struct.unpack_from("<H", data, 0)[0]
    if group_count != expected:
        raise BlobError(
            f"blob group_count {group_count} != model card groups {expected} "
            "- stale or reordered model card"
        )
    offset = 2
    scales: list[list[float]] = []
    for _ in range(group_count):
        if len(data) - offset < 4:
            raise BlobError("blob truncated in scale_count")
        n = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if encoding in INT_ENCODINGS:
            if len(data) - offset < n * 8:
                raise BlobError("blob truncated in scale block")
            sc = list(struct.unpack_from(f"<{n}d", data, offset))
            offset += n * 8
        else:
            if n != 0:
                raise BlobError(f"float encoding {encoding} must carry 0 scales, got {n}")
            sc = []
        scales.append(sc)
    return scales, offset


def decode_eta_blob(
    data: bytes,
    groups: list[Group],
    encoding: str = ENCODING_F64LE,
) -> list[tuple[str, Eta]]:
    """Decode a blob against the model card's groups (names and dims).

    Fails loudly on a group count mismatch, a truncated body, or a value
    count that disagrees with the family layout.
    """
    if encoding not in _BYTES_PER_VALUE:
        raise ValueError(f"unknown encoding: {encoding}")
    byte_width = _BYTES_PER_VALUE[encoding]
    scales, offset = _read_blob_header(data, len(groups), encoding, byte_width)

    out: list[tuple[str, Eta]] = []
    for i, group in enumerate(groups):
        count = value_count(group.family, group.dim)
        needed = count * byte_width
        if len(data) - offset < needed:
            raise BlobError(
                f"group {group.name}: body truncated, need {needed} bytes, "
                f"have {len(data) - offset}"
            )
        raw = data[offset : offset + needed]
        offset += needed
        block_sizes = _block_sizes(group)
        if encoding == ENCODING_F64LE:
            flat = np.frombuffer(raw, dtype="<f8").copy()
        elif encoding == ENCODING_F32LE:
            flat = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        else:
            ftype = "<i2" if encoding == ENCODING_I16LE else "<i1"
            q = np.frombuffer(raw, dtype=ftype).astype(np.float64)
            if len(scales[i]) != len(block_sizes):
                raise BlobError(
                    f"group {group.name}: scale_count {len(scales[i])} != "
                    f"block count {len(block_sizes)}"
                )
            # dequantize per natural-parameter index block
            chunks: list[np.ndarray] = []
            pos = 0
            for size, scale in zip(block_sizes, scales[i]):
                chunks.append(q[pos : pos + size] * scale)
                pos += size
            flat = np.concatenate(chunks)

        out.append((group.name, _unpack_values(group.family, group.dim, flat)))
    return out