"""Core data types for FIOR."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Event kind constants
KIND_MODEL_CARD = 30100
KIND_SITE_CONTRIBUTION = 30101
KIND_TRUST_ATTESTATION = 30102


# Distribution family constants
FAMILY_NORMAL = "normal"
FAMILY_GAMMA = "gamma"
FAMILY_BETA = "beta"
FAMILY_DIRICHLET = "dirichlet"
FAMILY_CATEGORICAL = "cat"


# Blob encoding constants
ENCODING_F64LE = "f64le"
ENCODING_F32LE = "f32le"
ENCODING_I16LE = "i16le"
ENCODING_I8 = "i8"


@dataclass
class Eta:
    """Natural parameters for an exponential family distribution.
    
    For Normal family:
        h = η₁ = μ/σ² (mean parameters)
        Lam = η₂ = -1/(2σ²) (precision parameters)
    """
    h: np.ndarray
    Lam: np.ndarray
    
    def __post_init__(self):
        self.h = np.asarray(self.h, dtype=np.float64)
        self.Lam = np.asarray(self.Lam, dtype=np.float64)
        if self.h.shape != self.Lam.shape:
            raise ValueError("h and Lam must have same shape")
    
    def copy(self) -> "Eta":
        return Eta(self.h.copy(), self.Lam.copy())
    
    def __add__(self, other: "Eta") -> "Eta":
        return Eta(self.h + other.h, self.Lam + other.Lam)
    
    def __sub__(self, other: "Eta") -> "Eta":
        return Eta(self.h - other.h, self.Lam - other.Lam)
    
    def __mul__(self, scalar: float) -> "Eta":
        return Eta(self.h * scalar, self.Lam * scalar)
    
    def __rmul__(self, scalar: float) -> "Eta":
        return self.__mul__(scalar)
    
    @property
    def dim(self) -> int:
        """Number of parameters (length of h or Lam)."""
        return len(self.h)
    
    def implied_mean(self) -> np.ndarray:
        """Compute implied mean μ = -h/(2*Lam) for Normal family."""
        return -self.h / (2 * self.Lam)
    
    def implied_variance(self) -> np.ndarray:
        """Compute implied variance σ² = -1/(2*Lam) for Normal family."""
        return -0.5 / self.Lam
    
    def in_domain(self, family: str) -> bool:
        """Check if parameters are in the valid domain for the family."""
        if family == FAMILY_NORMAL:
            return np.all(self.Lam < 0)
        elif family == FAMILY_GAMMA:
            return np.all(self.h > -1) and np.all(self.Lam < 0)
        elif family == FAMILY_BETA:
            return np.all(self.h > -1) and np.all(self.Lam > -1)
        elif family == FAMILY_DIRICHLET:
            return np.all(self.h > -1)
        elif family == FAMILY_CATEGORICAL:
            return True  # unconstrained
        return False


@dataclass
class Group:
    """Named set of initializers sharing a distribution."""
    name: str
    family: str
    initializers: list[str]
    
    @property
    def dim(self) -> int:
        """Dimensionality per parameter (depends on family)."""
        # For Normal, Gamma, Beta: 2 values per initializer
        # For Dirichlet, Categorical: 1 value per initializer
        if self.family in (FAMILY_NORMAL, FAMILY_GAMMA, FAMILY_BETA):
            return len(self.initializers)
        return len(self.initializers)


@dataclass
class ModelCard:
    """Model definition."""
    id: str
    title: str
    version: int
    groups: list[Group]
    eta0: Eta
    onnx_blob: str  # SHA-256 hex
    blossom_servers: list[str] = field(default_factory=list)
    summary: Optional[str] = None


@dataclass
class CavityMember:
    """One member of the cavity (m tag)."""
    pubkey: str  # hex
    event_id: str  # hex
    p_vector: str  # comma-separated p per group


@dataclass
class Site:
    """Published contribution."""
    author: str  # pubkey hex
    model_id: str
    model_version: int
    delta_eta: Eta
    cavity_members: list[CavityMember] = field(default_factory=list)
    event_id: Optional[str] = None  # Nostr event ID
    samples: Optional[int] = None
    free_energy: Optional[float] = None
    duration_sec: Optional[float] = None


@dataclass
class Attestation:
    """Trust rating for a peer."""
    target: str  # target pubkey hex
    scope: str  # "peer", "peer:model", or "peer:model:group"
    p: float  # inclusion probability (0,1)
    event_id: Optional[str] = None
