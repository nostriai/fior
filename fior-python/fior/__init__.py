"""
FIOR - Federated Inference Over Relays

A Nostr protocol for collaborative machine learning without sharing raw data.
"""

from .types import (
    Eta, Site, ModelCard, Group, Attestation,
    KIND_MODEL_CARD, KIND_SITE_CONTRIBUTION, KIND_TRUST_ATTESTATION,
    FAMILY_NORMAL, FAMILY_GAMMA, FAMILY_BETA, FAMILY_DIRICHLET, FAMILY_CATEGORICAL,
    ENCODING_F64LE, ENCODING_F32LE, ENCODING_I16LE, ENCODING_I8,
    TAG_MODEL_ID, TAG_TITLE, TAG_VERSION, TAG_SUMMARY, TAG_ONNX, TAG_DIST, TAG_GROUP,
    TAG_ETA0, TAG_BLOSSOM, TAG_TTL, TAG_GROUP_SEARCH, TAG_FAMILY_SEARCH,
    TAG_SITE_MODEL, TAG_SITE_COORD, TAG_SITE_VERSION, TAG_SITE_MEMBER, TAG_SITE_PUBKEY,
    TAG_SITE_EVENT, TAG_SITE_ETA, TAG_SITE_EXPIRATION,
    TAG_TRUST_TARGET, TAG_TRUST_PUBKEY, TAG_TRUST_PROB, TAG_TRUST_EXPIRATION,
)
from .math import log_partition, compose_prior, bmr_delta_f, p_from
from .trust import TrustTable, Corroboration
from .nostr import Event, sign_event, publish_event
from .blob import upload_blob, fetch_blob
from .client import Client

__version__ = "0.1.0"

__all__ = [
    "Eta",
    "Site",
    "ModelCard",
    "Group",
    "Attestation",
    "KIND_MODEL_CARD",
    "KIND_SITE_CONTRIBUTION",
    "KIND_TRUST_ATTESTATION",
    "FAMILY_NORMAL",
    "FAMILY_GAMMA",
    "FAMILY_BETA",
    "FAMILY_DIRICHLET",
    "FAMILY_CATEGORICAL",
    "ENCODING_F64LE",
    "ENCODING_F32LE",
    "ENCODING_I16LE",
    "ENCODING_I8",
    "TAG_MODEL_ID",
    "TAG_TITLE",
    "TAG_VERSION",
    "TAG_SUMMARY",
    "TAG_ONNX",
    "TAG_DIST",
    "TAG_GROUP",
    "TAG_ETA0",
    "TAG_BLOSSOM",
    "TAG_TTL",
    "TAG_GROUP_SEARCH",
    "TAG_FAMILY_SEARCH",
    "TAG_SITE_MODEL",
    "TAG_SITE_COORD",
    "TAG_SITE_VERSION",
    "TAG_SITE_MEMBER",
    "TAG_SITE_PUBKEY",
    "TAG_SITE_EVENT",
    "TAG_SITE_ETA",
    "TAG_SITE_EXPIRATION",
    "TAG_TRUST_TARGET",
    "TAG_TRUST_PUBKEY",
    "TAG_TRUST_PROB",
    "TAG_TRUST_EXPIRATION",
    "log_partition",
    "compose_prior",
    "bmr_delta_f",
    "p_from",
    "TrustTable",
    "Corroboration",
    "Event",
    "sign_event",
    "publish_event",
    "upload_blob",
    "fetch_blob",
    "Client",
]
