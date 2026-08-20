"""Core data types for FIOR."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Event kind constants
KIND_MODEL_CARD = 30100
KIND_SITE_CONTRIBUTION = 30101
KIND_TRUST_ATTESTATION = 30102

# Distribution family constants.
# `normal` is the independence special case: d scalar Normals, no off-diagonal
# precision.  `mvnormal` is the full-covariance multivariate Normal (h vector +
# symmetric precision matrix) that the directional trust machinery is defined
# for.  Both are kept; the client shipped with `normal` and the protocol adds
# `mvnormal` (pre-release, kinds and families may be extended).
FAMILY_NORMAL = "normal"
FAMILY_MVNORMAL = "mvnormal"
FAMILY_GAMMA = "gamma"
FAMILY_BETA = "beta"
FAMILY_DIRICHLET = "dirichlet"
FAMILY_CATEGORICAL = "cat"

FAMILIES = [
    FAMILY_NORMAL,
    FAMILY_MVNORMAL,
    FAMILY_GAMMA,
    FAMILY_BETA,
    FAMILY_DIRICHLET,
    FAMILY_CATEGORICAL,
]

# Families whose natural parameters split into an `h` (first) and a `Lam`
# (second) block.
TWO_BLOCK_FAMILIES = (FAMILY_NORMAL, FAMILY_MVNORMAL, FAMILY_GAMMA, FAMILY_BETA)
# Families whose natural parameters are a single vector.
ONE_BLOCK_FAMILIES = (FAMILY_DIRICHLET, FAMILY_CATEGORICAL)

# Blob encoding constants
ENCODING_F64LE = "f64le"
ENCODING_F32LE = "f32le"
ENCODING_I16LE = "i16le"
ENCODING_I8 = "i8"

FLOAT_ENCODINGS = (ENCODING_F64LE, ENCODING_F32LE)
INT_ENCODINGS = (ENCODING_I16LE, ENCODING_I8)

# Tag constants (single-letter for relay indexing)
# 30100 Model Card
TAG_MODEL_ID = "d"
TAG_TITLE = "t"
TAG_VERSION = "v"
TAG_SUMMARY = "s"
TAG_ONNX = "o"
TAG_DIST = "D"
TAG_GROUP = "g"
TAG_ETA0 = "x"
TAG_BLOSSOM = "b"
TAG_TTL = "l"
TAG_GROUP_SEARCH = "G"
TAG_FAMILY_SEARCH = "F"

# 30101 Site Contribution
TAG_SITE_MODEL = "d"
TAG_SITE_COORD = "a"
TAG_SITE_VERSION = "v"
TAG_SITE_MEMBER = "m"
TAG_SITE_PUBKEY = "p"
TAG_SITE_EVENT = "e"
TAG_SITE_ETA = "x"
TAG_SITE_EXPIRATION = "E"

# 30102 Trust Attestation
TAG_TRUST_TARGET = "d"
TAG_TRUST_PUBKEY = "p"
TAG_TRUST_PROB = "i"
TAG_TRUST_EXPIRATION = "E"


def value_count(family: str, dim: int) -> int:
    """Number of natural-parameter values a group of `dim` scalars carries.

    The model card's group dimensions are derived from ONNX initializer shapes
    (as the protocol requires); `value_count` is the family's multiplier from
    scalar parameters into wire values (protocol.md, Natural Parameter layout).
    """
    if family == FAMILY_MVNORMAL:
        return dim + dim * (dim + 1) // 2
    if family in (FAMILY_NORMAL, FAMILY_GAMMA, FAMILY_BETA):
        return 2 * dim
    if family in (FAMILY_DIRICHLET, FAMILY_CATEGORICAL):
        return dim
    raise ValueError(f"unknown family: {family}")


@dataclass
class Eta:
    """Natural parameters of one group of one exponential family.

    - `normal`, `gamma`, `beta`: h is the first block (`η₁`), Lam the second
      (`η₂`), both length `d`.
    - `mvnormal`: h is the mean vector (length `d`), Lam the symmetric `d×d`
      precision block (`η₂`), stored in full (not packed) for arithmetic;
      packing happens in the blob codec.
    - `dirichlet`, `cat`: h is the whole natural parameter vector (`η`), Lam
      is None.
    """

    family: str
    h: np.ndarray
    Lam: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family: {self.family}")
        self.h = np.asarray(self.h, dtype=np.float64)
        if self.Lam is not None:
            self.Lam = np.asarray(self.Lam, dtype=np.float64)
        if self.family in ONE_BLOCK_FAMILIES:
            if self.Lam is not None:
                raise ValueError(
                    f"family {self.family} carries a single natural parameter "
                    "vector; Lam must be None"
                )
        elif self.family == FAMILY_MVNORMAL:
            if self.Lam.ndim != 2 or self.Lam.shape != (len(self.h), len(self.h)):
                raise ValueError(
                    f"mvnormal Lam must be ({len(self.h)},{len(self.h)}), got {self.Lam.shape}"
                )
            if not np.allclose(self.Lam, self.Lam.T, atol=1e-12):
                raise ValueError("mvnormal Lam must be symmetric")
        else:
            if self.Lam is None:
                raise ValueError(f"family {self.family} requires a Lam block")
            if self.Lam.shape != self.h.shape:
                raise ValueError("h and Lam must have the same shape")

    @property
    def dim(self) -> int:
        """Number of scalar parameters (length of h)."""
        return len(self.h)

    @property
    def size(self) -> int:
        """Natural-parameter value count on the wire."""
        return value_count(self.family, self.dim)

    def copy(self) -> "Eta":
        lam = self.Lam.copy() if self.Lam is not None else None
        return Eta(self.family, self.h.copy(), lam)

    def __add__(self, other: "Eta") -> "Eta":
        if other.family != self.family:
            raise ValueError(f"family mismatch: {self.family} vs {other.family}")
        if other.dim != self.dim:
            raise ValueError(f"dim mismatch: {self.dim} vs {other.dim}")
        lam = None
        if self.Lam is not None:
            lam = self.Lam + other.Lam
        return Eta(self.family, self.h + other.h, lam)

    def __sub__(self, other: "Eta") -> "Eta":
        if other.family != self.family:
            raise ValueError(f"family mismatch: {self.family} vs {other.family}")
        if other.dim != self.dim:
            raise ValueError(f"dim mismatch: {self.dim} vs {other.dim}")
        lam = None
        if self.Lam is not None:
            lam = self.Lam - other.Lam
        return Eta(self.family, self.h - other.h, lam)

    def __mul__(self, scalar: float) -> "Eta":
        lam = None
        if self.Lam is not None:
            lam = self.Lam * scalar
        return Eta(self.family, self.h * scalar, lam)

    __rmul__ = __mul__

    def in_domain(self) -> bool:
        """Check the natural-parameter domain constraint of this group."""
        if self.family == FAMILY_NORMAL:
            return bool(np.all(self.Lam < 0))
        if self.family == FAMILY_MVNORMAL:
            try:
                np.linalg.cholesky(-2.0 * self.Lam)
                return True
            except np.linalg.LinAlgError:
                return False
        if self.family == FAMILY_GAMMA:
            return bool(np.all(self.h > -1) and np.all(self.Lam < 0))
        if self.family == FAMILY_BETA:
            return bool(np.all(self.h > -1) and np.all(self.Lam > -1))
        if self.family == FAMILY_DIRICHLET:
            return bool(np.all(self.h > -1))
        if self.family == FAMILY_CATEGORICAL:
            return True
        return False

    def implied_mean(self) -> np.ndarray:
        """Implied mean: `-h/(2*Lam)` per component; `Λ⁻¹h` (Λ = -2η2) for mvnormal."""
        if self.family == FAMILY_MVNORMAL:
            return np.linalg.solve(-2.0 * self.Lam, self.h)
        if self.Lam is not None:
            return -self.h / (2 * self.Lam)
        raise ValueError(f"family {self.family} has no implied mean in (h, Lam)")


@dataclass
class Group:
    """Named set of ONNX initializers sharing one distribution."""

    name: str
    family: str
    initializers: list[str] = field(default_factory=list)
    dim: int = 0  # total scalar parameters, derived from ONNX shapes

    @property
    def value_count(self) -> int:
        return value_count(self.family, self.dim)


@dataclass
class ModelCard:
    """Model definition (kind 30100)."""

    id: str
    title: str
    version: int
    groups: list[Group]
    eta0: dict[str, Eta] = field(default_factory=dict)
    onnx_blob: Optional[tuple] = None  # (sha256, enc, size, servers)
    eta0_ref: Optional[tuple] = None  # (sha256, enc, size, servers)
    blossom_servers: list[str] = field(default_factory=list)
    summary: Optional[str] = None
    creator: Optional[str] = None  # event pubkey of the model card
    event_id: Optional[str] = None
    ttl: Optional[int] = None

    @property
    def coordinate(self) -> str:
        return f"30100:{self.creator}:{self.id}" if self.creator else ""


@dataclass
class CavityMember:
    """One member of the cavity (one `m` tag)."""

    pubkey: str  # hex
    event_id: str  # hex
    p_vector: list[float]  # p per group, model card order


@dataclass
class Site:
    """A published site contribution (kind 30101), with its delta_eta fetched."""

    author: str
    model_id: str
    model_version: int
    delta_eta: dict[str, Eta]
    cavity_members: list[CavityMember] = field(default_factory=list)
    event_id: Optional[str] = None
    blob_ref: Optional[tuple] = None  # (sha256, enc, size, servers)
    created_at: Optional[int] = None
    samples: Optional[int] = None
    free_energy: Optional[float] = None
    duration_sec: Optional[float] = None


@dataclass
class SiteDescriptor:
    """A site's metadata without its parameter blob (list_sites output)."""

    author: str
    model_id: str
    model_version: int
    event_id: str
    blob_ref: tuple  # (sha256, enc, size, servers)
    cavity_members: list[CavityMember] = field(default_factory=list)
    created_at: Optional[int] = None
    expires_at: Optional[int] = None
    samples: Optional[int] = None
    free_energy: Optional[float] = None
    duration_sec: Optional[float] = None


@dataclass
class Attestation:
    """Trust rating for a peer (30102)."""

    attester: str  # hex
    target: str  # target pubkey hex
    p: float  # inclusion probability (0,1)
    model_id: Optional[str] = None
    group: Optional[str] = None
    event_id: Optional[str] = None