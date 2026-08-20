"""
FIOR - Federated Inference Over Relays

A Nostr protocol for collaborative machine learning without sharing raw data.
"""

from . import bip340
from .types import (
    Eta,
    Site,
    ModelCard,
    Group,
    Attestation,
    CavityMember,
    SiteDescriptor,
    KIND_MODEL_CARD,
    KIND_SITE_CONTRIBUTION,
    KIND_TRUST_ATTESTATION,
    FAMILY_NORMAL,
    FAMILY_MVNORMAL,
    FAMILY_GAMMA,
    FAMILY_BETA,
    FAMILY_DIRICHLET,
    FAMILY_CATEGORICAL,
    ONE_BLOCK_FAMILIES,
    ENCODING_F64LE,
    ENCODING_F32LE,
    ENCODING_I16LE,
    ENCODING_I8,
    value_count,
)
from .math import log_partition, compose_prior, bmr_delta_f, p_from, logistic
from .trust import TrustTable, Corroboration, loewner_clip, novelty_weight, causal_span_residual
from .nostr import Event, sign_event, publish_event
from .blob import upload_blob, fetch_blob, encode_eta_blob, decode_eta_blob
from .client import Client, ClientError, DomainError

__version__ = "0.2.0"

__all__ = [
    "Eta",
    "Site",
    "ModelCard",
    "Group",
    "Attestation",
    "CavityMember",
    "SiteDescriptor",
    "Client",
    "ClientError",
    "DomainError",
    "KIND_MODEL_CARD",
    "KIND_SITE_CONTRIBUTION",
    "KIND_TRUST_ATTESTATION",
    "FAMILY_NORMAL",
    "FAMILY_MVNORMAL",
    "FAMILY_GAMMA",
    "FAMILY_BETA",
    "FAMILY_DIRICHLET",
    "FAMILY_CATEGORICAL",
    "ENCODING_F64LE",
    "ENCODING_F32LE",
    "ENCODING_I16LE",
    "ENCODING_I8",
    "value_count",
    "log_partition",
    "compose_prior",
    "bmr_delta_f",
    "p_from",
    "logistic",
    "TrustTable",
    "Corroboration",
    "loewner_clip",
    "novelty_weight",
    "causal_span_residual",
    "Event",
    "sign_event",
    "publish_event",
    "upload_blob",
    "fetch_blob",
    "encode_eta_blob",
    "decode_eta_blob",
]