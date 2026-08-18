"""Blossom blob upload and download for FIOR.

Implements BUD-01 (blob retrieval by SHA-256).
"""

import hashlib
from typing import Optional


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def upload_blob(
    data: bytes,
    server_url: str,
    sha256: Optional[str] = None,
) -> str:
    """Upload a blob to a Blossom server.
    
    Args:
        data: Raw bytes to upload
        server_url: Blossom server root URL
        sha256: Optional pre-computed hash (computed if omitted)
    
    Returns:
        SHA-256 hash of the uploaded blob
    """
    import requests
    
    if sha256 is None:
        sha256 = compute_sha256(data)
    
    # BUD-02: Upload endpoint
    url = f"{server_url}/upload"
    
    response = requests.post(
        url,
        data=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    
    response.raise_for_status()
    
    return sha256


def fetch_blob(
    sha256: str,
    server_urls: list[str],
    verify: bool = True,
) -> bytes:
    """Fetch a blob from Blossom servers.
    
    Tries each server in order until successful.
    
    Args:
        sha256: SHA-256 hash of the blob
        server_urls: List of Blossom server root URLs to try
        verify: Whether to verify SHA-256 on fetch
    
    Returns:
        Raw bytes of the blob
    
    Raises:
        RuntimeError: If all servers fail
        ValueError: If SHA-256 verification fails
    """
    import requests
    
    last_error = None
    
    for server_url in server_urls:
        try:
            # BUD-01: Fetch by SHA-256
            url = f"{server_url}/{sha256}"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.content
            
            # Verify SHA-256
            if verify:
                computed = compute_sha256(data)
                if computed != sha256:
                    raise ValueError(
                        f"SHA-256 mismatch: expected {sha256}, got {computed}"
                    )
            
            return data
        
        except Exception as e:
            last_error = e
            continue
    
    raise RuntimeError(f"Failed to fetch blob from any server: {last_error}")


def encode_eta_blob(
    eta_groups: list,
    encoding: str = "f64le",
) -> bytes:
    """Encode natural parameters to binary blob.
    
    Args:
        eta_groups: List of (group_name, eta) tuples in model card order
        encoding: Blob encoding ("f64le", "f32le", etc.)
    
    Returns:
        Binary blob
    """
    import struct
    import numpy as np
    
    if encoding != "f64le":
        raise NotImplementedError(f"Encoding {encoding} not yet implemented")
    
    # Header: group_count (u16le)
    header = struct.pack("<H", len(eta_groups))
    
    # Body: each group's h and Lam concatenated
    body = b""
    for name, eta in eta_groups:
        # For float encodings, no scales
        header += struct.pack("<B", 0)  # scale_count = 0
        
        # h values
        body += eta.h.astype(np.float64).tobytes()
        # Lam values
        body += eta.Lam.astype(np.float64).tobytes()
    
    return header + body


def decode_eta_blob(
    data: bytes,
    group_dims: list[int],
    encoding: str = "f64le",
) -> list:
    """Decode binary blob to natural parameters.
    
    Args:
        data: Binary blob
        group_dims: Dimensionality of each group
        encoding: Blob encoding
    
    Returns:
        List of (group_name, Eta) tuples
    """
    import struct
    import numpy as np
    from .types import Eta
    
    if encoding != "f64le":
        raise NotImplementedError(f"Encoding {encoding} not yet implemented")
    
    offset = 0
    
    # Header: group_count (u16le)
    group_count = struct.unpack_from("<H", data, offset)[0]
    offset += 2
    
    result = []
    
    for i in range(group_count):
        # Scale count (u8)
        scale_count = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        
        # Skip scales (not used for float encodings)
        offset += scale_count * 8  # f64 scales
        
        dim = group_dims[i] if i < len(group_dims) else group_dims[-1]
        
        # h values
        h = np.frombuffer(data[offset:offset + dim * 8], dtype="<f8")
        offset += dim * 8
        
        # Lam values
        Lam = np.frombuffer(data[offset:offset + dim * 8], dtype="<f8")
        offset += dim * 8
        
        result.append((f"group_{i}", Eta(h, Lam)))
    
    return result
