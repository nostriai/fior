"""Blob codec tests: round-trips across families and encodings, stochastic
rounding unbiasedness, and loud failure on mismatches."""

import hashlib
import struct

import numpy as np
import pytest

from fior import Eta, Group, ENCODING_F64LE, ENCODING_F32LE, ENCODING_I16LE, ENCODING_I8
from fior.blob import (
    BlobError,
    compute_sha256,
    encode_eta_blob,
    decode_eta_blob,
    fetch_blob,
    upload_blob,
)

GROUPS = [
    Group("fc", "normal", ["fc1"], 3),
    Group("mv", "mvnormal", ["w1", "w2"], 2),
    Group("ga", "gamma", ["g"], 2),
    Group("be", "beta", ["b"], 2),
    Group("di", "dirichlet", ["d"], 4),
    Group("ca", "cat", ["c"], 5),
]

ETAS = [
    ("fc", Eta("normal", np.array([1.0, -2.0, 0.5]), np.array([-1.0, -3.0, -0.25]))),
    ("mv", Eta("mvnormal", np.array([1.0, 2.0]), np.array([[-1.0, 0.2], [0.2, -2.0]]))),
    ("ga", Eta("gamma", np.array([1.0, 2.0]), np.array([-1.5, -0.5]))),
    ("be", Eta("beta", np.array([0.5, 1.2]), np.array([2.1, 0.3]))),
    ("di", Eta("dirichlet", np.array([0.5, 1.0, 2.0, 3.0]))),
    ("ca", Eta("cat", np.array([0.1, -1.0, 0.0, 0.5, -0.2]))),
]


@pytest.mark.parametrize("enc", [ENCODING_F64LE, ENCODING_F32LE, ENCODING_I16LE, ENCODING_I8])
def test_roundtrip_all_families(enc):
    blob = encode_eta_blob(GROUPS, ETAS, enc, seed=42)
    dec = decode_eta_blob(blob, GROUPS, enc)
    assert [n for n, _ in dec] == [n for n, _ in ETAS]
    for (n1, e1), (n2, e2) in zip(ETAS, dec):
        assert n1 == n2 and e1.family == e2.family
        err = np.abs(e1.h - e2.h).max()
        if e1.Lam is not None:
            err = max(err, np.abs(e1.Lam - e2.Lam).max())
        tol = {ENCODING_F64LE: 1e-12, ENCODING_F32LE: 1e-6, ENCODING_I16LE: 1e-2, ENCODING_I8: 1e-1}[enc]
        assert err < tol, (n1, enc, err)


def test_group_count_mismatch_fails_loudly():
    blob = encode_eta_blob(GROUPS[:1], ETAS[:1])
    with pytest.raises(BlobError):
        decode_eta_blob(blob, GROUPS)


def test_truncated_body_fails_loudly():
    blob = encode_eta_blob(GROUPS, ETAS)
    with pytest.raises(BlobError):
        decode_eta_blob(blob[:-3], GROUPS)


def test_eta_group_order_mismatch_fails_on_encode():
    with pytest.raises(BlobError):
        encode_eta_blob([GROUPS[1]], [(GROUPS[0].name, ETAS[0][1])])


def test_stochastic_rounding_unbiased():
    # mean error over many quantizations of the same values is ~0, not -E/2
    g = Group("n", "normal", ["x"], 5)
    values = np.array([0.1, -0.7, 2.0, -3.0, 0.05])
    lam = -np.abs(values) * 0.5
    errs = []
    for _ in range(400):
        e = Eta("normal", values, lam)
        blob = encode_eta_blob([g], [("n", e)], "i8")
        back = decode_eta_blob(blob, [g], "i8")[0][1]
        errs.append(back.h - values)
    errs = np.array(errs)
    assert np.abs(errs.mean(axis=0)).max() < 0.05, f"bias: {errs.mean(axis=0)}"


def test_sha256_and_upload_fetch_roundtrip_with_stub():
    from tests.stubs import start_blossom

    store = {}
    blossom = start_blossom(store)
    payload = b"hello fior blob"
    sha = upload_blob(payload, blossom)
    assert sha == compute_sha256(payload) == hashlib.sha256(payload).hexdigest()
    assert fetch_blob(sha, [blossom]) == payload
    with pytest.raises(RuntimeError):
        fetch_blob("0" * 64, [blossom])