"""High-level FIOR client API.

Provides methods for participating in federated inference.
"""

import json
from typing import Optional
from .types import Eta, Site, ModelCard, Group, Attestation, CavityMember
from .math import compose_prior, bmr_delta_f, p_from
from .trust import TrustTable, Corroboration
from .nostr import (
    Event,
    create_model_card_event,
    create_site_contribution_event,
    create_trust_attestation_event,
    publish_event,
)
from .blob import upload_blob, fetch_blob, compute_sha256


class Client:
    """FIOR client for federated inference.
    
    Provides methods for fetching models, sites, and attestations,
    composing priors, and publishing contributions.
    """
    
    def __init__(
        self,
        relay_url: str = "wss://relay.example.com",
        blossom_servers: Optional[list[str]] = None,
    ):
        """Initialize client.
        
        Args:
            relay_url: Default relay WebSocket URL
            blossom_servers: Default Blossom server URLs
        """
        self.relay_url = relay_url
        self.blossom_servers = blossom_servers or []
        self.trust_table = TrustTable()
        self.corroboration = Corroboration()
    
    async def fetch_model_card(
        self,
        model_id: str,
        relay_url: Optional[str] = None,
    ) -> Optional[ModelCard]:
        """Fetch a model card from the relay.
        
        Args:
            model_id: Model identifier (d tag)
            relay_url: Optional relay override
        
        Returns:
            ModelCard or None if not found
        """
        import websockets
        
        relay = relay_url or self.relay_url
        
        async with websockets.connect(relay) as ws:
            # Query for model card
            query = {
                "kinds": [30100],
                "#d": [model_id],
                "limit": 1,
            }
            
            await ws.send(json.dumps(["REQ", "model_card", query]))
            
            # Read responses until we get an EVENT or EOSE
            while True:
                response = await ws.recv()
                data = json.loads(response)
                
                if data[0] == "EOSE":
                    return None
                elif data[0] == "EVENT":
                    event = data[2]
                    return self._parse_model_card(event)
                # Skip AUTH and other messages
    
    async def fetch_sites(
        self,
        model_id: str,
        relay_url: Optional[str] = None,
    ) -> list[Site]:
        """Fetch site contributions for a model.
        
        Args:
            model_id: Model identifier
            relay_url: Optional relay override
        
        Returns:
            List of Site objects
        """
        import websockets
        
        relay = relay_url or self.relay_url
        
        async with websockets.connect(relay) as ws:
            query = {
                "kinds": [30101],
                "#d": [model_id],
            }
            
            await ws.send(json.dumps(["REQ", "sites", query]))
            
            sites = []
            while True:
                response = await ws.recv()
                data = json.loads(response)
                
                if data[0] == "EOSE":
                    break
                elif data[0] == "EVENT":
                    event = data[2]
                    site = self._parse_site(event)
                    if site:
                        sites.append(site)
                # Skip AUTH and other messages
            
            return sites
    
    async def fetch_attestations(
        self,
        target_pubkey: str,
        relay_url: Optional[str] = None,
    ) -> list[Attestation]:
        """Fetch trust attestations for a peer.
        
        Args:
            target_pubkey: Target peer's pubkey (hex)
            relay_url: Optional relay override
        
        Returns:
            List of Attestation objects
        """
        import websockets
        
        relay = relay_url or self.relay_url
        
        async with websockets.connect(relay) as ws:
            query = {
                "kinds": [30102],
                "#p": [target_pubkey],
            }
            
            await ws.send(json.dumps(["REQ", "attestations", query]))
            
            attestations = []
            while True:
                response = await ws.recv()
                data = json.loads(response)
                
                if data[0] == "EOSE":
                    break
                elif data[0] == "EVENT":
                    event = data[2]
                    att = self._parse_attestation(event)
                    if att:
                        attestations.append(att)
                # Skip AUTH and other messages
            
            return attestations
    
    def compose_prior(
        self,
        eta0: Eta,
        sites: list[Site],
        p_values: Optional[list[float]] = None,
    ) -> Eta:
        """Build a local prior from trusted peers.
        
        Args:
            eta0: Base prior (from model card)
            sites: List of Site objects to include
            p_values: Optional p values (uses trust table if omitted)
        
        Returns:
            Composed prior Eta
        """
        if p_values is None:
            p_values = [
                self.trust_table.get_p(site.author, "default")
                for site in sites
            ]
        
        return compose_prior(eta0, sites, p_values)
    
    def compute_p(
        self,
        site: Site,
        prior_without_site: Eta,
        local_likelihood: Eta,
        beta: float = 0.5,
        family: str = "normal",
    ) -> float:
        """Compute inclusion probability for a peer.
        
        Args:
            site: Site to evaluate
            prior_without_site: Prior excluding this site
            local_likelihood: Client's local data contribution
            beta: Prior belief from attestations
            family: Distribution family
        
        Returns:
            Inclusion probability p in (0, 1)
        """
        # Build the four Eta configurations for BMR
        eta_p_minus = prior_without_site
        eta_p_plus = eta_p_minus + site.delta_eta
        eta_q_minus = eta_p_minus + local_likelihood
        eta_q_plus = eta_p_plus + local_likelihood
        
        # Compute ΔF
        delta_f = bmr_delta_f(eta_q_plus, eta_p_minus, eta_q_minus, eta_p_plus, family)
        
        # Resolve p
        return p_from(delta_f, beta)
    
    def apply_corroboration(
        self,
        site: Site,
        trusted_sites: list[Site],
    ) -> float:
        """Apply corroboration scoring to a site.
        
        Args:
            site: Site to evaluate
            trusted_sites: List of trusted sites
        
        Returns:
            Multiplicative factor in (0, 1] to apply to p
        """
        p_values = [
            self.trust_table.get_p(s.author, "default")
            for s in trusted_sites
        ]
        
        return self.corroboration.score(site, trusted_sites, p_values)
    
    async def publish_site(
        self,
        private_key: str,
        model_id: str,
        model_version: int,
        delta_eta: Eta,
        cavity_members: list[CavityMember],
        blob_data: bytes,
        samples: Optional[int] = None,
        free_energy: Optional[float] = None,
        duration_sec: Optional[float] = None,
        relay_url: Optional[str] = None,
    ) -> str:
        """Publish a site contribution.
        
        Args:
            private_key: Node's private key
            model_id: Model identifier
            model_version: Model card version
            delta_eta: Site contribution
            cavity_members: Members of the cavity
            blob_data: Raw binary of Δη
            samples: Training samples
            free_energy: ELBO value
            duration_sec: Training duration
            relay_url: Optional relay override
        
        Returns:
            Event ID
        """
        # Upload blob
        sha256 = upload_blob(blob_data, self.blossom_servers[0])
        
        # Create event
        event = create_site_contribution_event(
            private_key=private_key,
            model_id=model_id,
            model_version=model_version,
            delta_eta_blob=sha256,
            cavity_members=[(m.pubkey, m.event_id, m.p_vector) for m in cavity_members],
            samples=samples,
            free_energy=free_energy,
            duration_sec=duration_sec,
        )
        
        # Publish
        relay = relay_url or self.relay_url
        await publish_event(event, relay)
        
        return event.id
    
    async def publish_attestation(
        self,
        private_key: str,
        target_pubkey: str,
        scope: str,
        inclusion_probability: float,
        model_ref: Optional[str] = None,
        relay_url: Optional[str] = None,
    ) -> str:
        """Publish a trust attestation.
        
        Args:
            private_key: Attester's private key
            target_pubkey: Target peer's pubkey
            scope: "peer", "peer:model", or "peer:model:group"
            inclusion_probability: p value in (0,1)
            model_ref: Model card coordinate
            relay_url: Optional relay override
        
        Returns:
            Event ID
        """
        event = create_trust_attestation_event(
            private_key=private_key,
            target_pubkey=target_pubkey,
            scope=scope,
            inclusion_probability=inclusion_probability,
            model_ref=model_ref,
        )
        
        relay = relay_url or self.relay_url
        await publish_event(event, relay)
        
        return event.id
    
    def _parse_model_card(self, event: dict) -> Optional[ModelCard]:
        """Parse a Model Card event."""
        try:
            tags = {t[0]: t[1:] for t in event["tags"]}
            
            # Parse groups
            groups = []
            for tag in event["tags"]:
                if tag[0] == "g":
                    groups.append(Group(
                        name=tag[1],
                        family=tag[2],
                        initializers=tag[3:],
                    ))
            
            # Parse eta0 if present
            eta0 = Eta(
                h=[0.0],  # placeholder
                Lam=[-1.0],  # placeholder
            )
            
            return ModelCard(
                id=tags.get("d", [""])[0],
                title=tags.get("t", [""])[0],
                version=int(tags.get("v", ["0"])[0]),
                groups=groups,
                eta0=eta0,
                onnx_blob=tags.get("o", [""])[0],
                blossom_servers=tags.get("b", []),
                summary=tags.get("s", [None])[0],
            )
        except Exception:
            return None
    
    def _parse_site(self, event: dict) -> Optional[Site]:
        """Parse a Site Contribution event."""
        try:
            tags = {t[0]: t[1:] for t in event["tags"]}
            
            # Parse cavity members
            members = []
            for tag in event["tags"]:
                if tag[0] == "m":
                    members.append(CavityMember(
                        pubkey=tag[1],
                        event_id=tag[2],
                        p_vector=tag[3] if len(tag) > 3 else "",
                    ))
            
            # Parse content
            content = json.loads(event.get("content", "{}"))
            
            # Placeholder delta_eta - in real impl, fetch from blob
            delta_eta = Eta(h=[0.0], Lam=[-1.0])
            
            return Site(
                author=event["pubkey"],
                model_id=tags.get("d", [""])[0],
                model_version=int(tags.get("v", ["0"])[0]),
                delta_eta=delta_eta,
                cavity_members=members,
                event_id=event["id"],
                samples=content.get("samples"),
                free_energy=content.get("free_energy"),
                duration_sec=content.get("duration_sec"),
            )
        except Exception:
            return None
    
    def _parse_attestation(self, event: dict) -> Optional[Attestation]:
        """Parse a Trust Attestation event."""
        try:
            tags = {t[0]: t[1:] for t in event["tags"]}
            
            return Attestation(
                target=tags.get("p", [""])[0],
                scope=tags.get("d", [""])[0],
                p=float(tags.get("i", ["0.5"])[0]),
                event_id=event["id"],
            )
        except Exception:
            return None
