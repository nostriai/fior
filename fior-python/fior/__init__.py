"""
FIOR - Federated Inference Over Relays

A Nostr protocol for collaborative machine learning without sharing raw data.
"""

from .types import Eta, Site, ModelCard, Group, Attestation
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
