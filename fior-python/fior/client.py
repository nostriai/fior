"""High-level FIOR client API.

Implements interface.md: discovery, composition, the trust layer (BMR scoring,
the Loewner confidence bound, corroboration, novelty, attestation prior), and
publication.  Methods are synchronous; each drives a private event loop, which
is right for scripts and the CLI.  The model backend (ONNX shape extraction)
is a hook: pass `initializer_dims` to read_model_card, or set the parsed
card's group dims yourself.

Local policy knobs (kappa_r, spectral_cap, corroboration t0, novelty) never
leave the node and never change the wire format.
"""

import json
import time
from typing import Optional

import numpy as np

from . import bip340, nostr
from .blob import BlobError, compute_sha256, decode_eta_blob, encode_eta_blob, fetch_blob, upload_blob
from .math import bmr_delta_f, compose_prior, p_from
from .trust import Corroboration, TrustTable, loewner_clip, novelty_weight, resolve_attestation_prior
from .types import (
    Attestation,
    CavityMember,
    Eta,
    Group,
    ModelCard,
    Site,
    SiteDescriptor,
    FAMILY_MVNORMAL,
    ONE_BLOCK_FAMILIES,
)


def _run(coro):
    import asyncio

    return asyncio.run(coro)


class ClientError(Exception):
    pass


class DomainError(ValueError):
    """A composed group left its natural parameter domain."""


class Client:
    """A FIOR participant.

    Parameters:
        relay_url: default relay for discovery and publication
        blossom_servers: default Blossom server roots for upload and fallback
        kappa_r: reader-side attestation weight, SHOULD be in 1..4
        private_key: hex secret key used for signing publications
        trust_path: JSON file for the local trust table and first-seen order
        spectral_cap: Loewner confidence bound constant c (0 disables)
        corroborate_t0: corroboration tolerance in pool MADs (0 disables)
        novelty: scale p by the causal span residual (needs first-seen order;
                 never enable without a confidence bound)
    """

    def __init__(
        self,
        relay_url: str = "wss://relay.example.com",
        blossom_servers: Optional[list[str]] = None,
        private_key: Optional[str] = None,
        kappa_r: float = 4.0,
        trust_path: Optional[str] = None,
        spectral_cap: float = 3.0,
        corroborate_t0: float = 0.0,
        novelty: bool = False,
    ):
        self.relay_url = relay_url
        self.blossom_servers = list(blossom_servers or [])
        self.private_key = private_key
        self.kappa_r = kappa_r
        self.spectral_cap = spectral_cap
        self.corroborate_t0 = corroborate_t0
        self.novelty = novelty
        self.trust_path = trust_path
        self.trust = TrustTable(path=trust_path)
        self.models: dict[str, ModelCard] = {}
        self.sites: dict[str, dict[str, SiteDescriptor]] = {}
        self.params: dict[str, dict[str, dict[str, Eta]]] = {}
        self.attestations: list[Attestation] = []
        self.first_seen: dict[str, float] = {}

    # -- identity / persistence ---------------------------------------

    def my_pubkey(self) -> Optional[str]:
        if not self.private_key:
            return None
        return bip340.get_pubkey(bytes.fromhex(self.private_key)).hex()

    def save_state(self, path=None) -> None:
        self.trust.save(path or self.trust_path)

    # -- relay plumbing ------------------------------------------------

    def _fetch_events(self, filters: list[dict], timeout: float = 10.0) -> list[dict]:
        events = []

        async def go():
            async for _sub, ev in nostr.subscribe(self.relay_url, filters, timeout=timeout):
                events.append(ev)

        _run(go())
        return events

    def _publish(self, event_dict: dict) -> nostr.PubResult:
        return _run(nostr.publish_event(event_dict, self.relay_url))

    # -- tag helpers ---------------------------------------------------

    @staticmethod
    def _tags(ev: dict, key: str) -> list[list[str]]:
        return [t for t in ev.get("tags", []) if t and t[0] == key]

    @staticmethod
    def _tag(ev: dict, key: str, default=None):
        ts = Client._tags(ev, key)
        return ts[0][1] if ts else default

    @staticmethod
    def _blob_ref(ev: dict, key: str):
        ts = Client._tags(ev, key)
        if not ts:
            return None
        t = ts[0]
        if len(t) < 4:
            return None
        return (t[1], t[2], int(t[3]) if t[3].isdigit() else 0, t[4:])

    # -- parsing -------------------------------------------------------

    def _parse_model_card(self, ev: dict, dims: dict) -> Optional[ModelCard]:
        groups = []
        names = set()
        for t in self._tags(ev, "g"):
            if len(t) < 3:
                continue
            name, family, inits = t[1], t[2], t[3:]
            if family not in ("normal", "mvnormal", "gamma", "beta", "dirichlet", "cat"):
                continue
            groups.append(Group(name=name, family=family, initializers=inits,
                                dim=sum(dims.get(i, 0) for i in inits)))
            names.add(name)
        for t in self._tags(ev, "D"):
            if len(t) < 3 or t[1] in names:
                continue
            groups.append(Group(name=t[1], family=t[2], initializers=[t[1]], dim=dims.get(t[1], 0)))
        try:
            version = int(self._tag(ev, "v", "0"))
            ttl = int(self._tag(ev, "l", "0")) or None
        except ValueError:
            version, ttl = 0, None
        b_tags = self._tags(ev, "b")
        return ModelCard(
            id=self._tag(ev, "d", ""),
            title=self._tag(ev, "t", ""),
            version=version,
            groups=groups,
            onnx_blob=self._blob_ref(ev, "o"),
            eta0_ref=self._blob_ref(ev, "x"),
            blossom_servers=b_tags[0][1:] if b_tags else [],
            summary=self._tag(ev, "s"),
            creator=ev.get("pubkey"),
            event_id=ev.get("id"),
            ttl=ttl,
        )

    def _is_expired(self, ev: dict, card: Optional[ModelCard]) -> bool:
        e = self._tag(ev, "E")
        if e:
            try:
                if int(e) < int(time.time()):
                    return True
            except ValueError:
                pass
        if card and card.ttl:
            try:
                if int(ev.get("created_at") or 0) + card.ttl < int(time.time()):
                    return True
            except (TypeError, ValueError):
                pass
        return False

    def _parse_site_descriptor(self, ev: dict, card: Optional[ModelCard]) -> SiteDescriptor:
        members = []
        for t in self._tags(ev, "m"):
            if len(t) < 3:
                continue
            pv = []
            if len(t) > 3 and t[3]:
                try:
                    pv = [float(x) for x in t[3].split(",")]
                except ValueError:
                    pv = []
            members.append(CavityMember(pubkey=t[1], event_id=t[2], p_vector=pv))
        try:
            content = json.loads(ev.get("content", "{}"))
        except json.JSONDecodeError:
            content = {}
        try:
            mv = int(self._tag(ev, "v", "0"))
        except ValueError:
            mv = 0
        e_tag = self._tag(ev, "E")
        return SiteDescriptor(
            author=ev.get("pubkey", ""),
            model_id=self._tag(ev, "d", ""),
            model_version=mv,
            event_id=ev.get("id", ""),
            blob_ref=self._blob_ref(ev, "x") or ("", "", 0, []),
            cavity_members=members,
            created_at=ev.get("created_at"),
            expires_at=int(e_tag) if e_tag else None,
            samples=content.get("samples"),
            free_energy=content.get("free_energy"),
            duration_sec=content.get("duration_sec"),
        )

    def _parse_attestation(self, ev: dict) -> Optional[Attestation]:
        d = self._tag(ev, "d", "")
        target = self._tag(ev, "p", "")
        p_str = self._tag(ev, "i") or "0.5"
        try:
            p = float(p_str)
        except ValueError:
            return None
        model = group = None
        parts = d.split(":")
        if len(parts) >= 2 and target and d != target:
            model = parts[-2] if len(parts) >= 2 and not parts[-1].isdigit() else None
        # d = <target>[:<model>[:<group>]]
        if target and d.startswith(target):
            rest = d[len(target):]
            if rest:
                bits = rest.lstrip(":").split(":")
                if bits:
                    model = bits[0] or None
                if len(bits) > 1:
                    group = bits[1] or None
        return Attestation(
            attester=ev.get("pubkey", ""),
            target=target,
            p=p,
            model_id=model,
            group=group,
            event_id=ev.get("id"),
        )

    # -- zero base prior when the card has no x blob -------------------

    @staticmethod
    def _zero_group(group: Group) -> Eta:
        eps = 1e-4
        if group.family == FAMILY_MVNORMAL:
            return Eta(group.family, np.zeros(group.dim), -eps * np.eye(group.dim))
        if group.family in ONE_BLOCK_FAMILIES:
            return Eta(group.family, np.zeros(group.dim))
        return Eta(group.family, np.zeros(group.dim), -np.full(group.dim, eps))

    # -- discovery -----------------------------------------------------

    def read_model_card(self, model_id: str, initializer_dims: Optional[dict] = None,
                        fetch_eta0: bool = True) -> Optional[ModelCard]:
        """Latest 30100 for the model; fetch and decode η₀ when present."""
        evs = self._fetch_events([{"kinds": [30100], "#d": [model_id], "limit": 1}])
        if not evs:
            return None
        ev = max(evs, key=lambda e: e.get("created_at", 0))
        card = self._parse_model_card(ev, initializer_dims or {})
        if fetch_eta0 and card is not None and card.eta0_ref:
            sha, enc, size, servers = card.eta0_ref
            servers = [s for s in servers if s] or self.blossom_servers or card.blossom_servers
            if servers:
                try:
                    raw = fetch_blob(sha, servers)
                    pairs = decode_eta_blob(raw, card.groups, enc)
                    card.eta0 = {n: e for n, e in pairs}
                except (RuntimeError, BlobError):
                    card.eta0 = {}
        self.models[model_id] = card
        return card

    def list_sites(self, model_id: str) -> list[SiteDescriptor]:
        """Latest non-stale 30101 per author for the model."""
        card = self.models.get(model_id)
        evs = self._fetch_events([{"kinds": [30101], "#d": [model_id]}])
        by_author: dict[str, dict] = {}
        for ev in evs:
            if self._is_expired(ev, card):
                continue
            v = self._tag(ev, "v")
            if card is not None and v is not None and v.isdigit() and int(v) != card.version:
                continue
            author = ev["pubkey"]
            cur = by_author.get(author)
            if cur is None or int(ev.get("created_at", 0)) > int(cur.get("created_at", 0)):
                by_author[author] = ev
            if ev.get("id") not in self.first_seen:
                self.first_seen[ev["id"]] = float(ev.get("created_at", 0))
        descs = [self._parse_site_descriptor(ev, card) for ev in by_author.values()]
        self.sites[model_id] = {d.author: d for d in descs}
        return descs

    def fetch_site_params(self, descriptor: SiteDescriptor, model_id: str) -> dict[str, Eta]:
        """Fetch, verify, decode the site's Δη blob.  Unresolvable -> {}."""
        card = self.models.get(model_id)
        if card is None:
            raise LookupError(f"read the model card first: {model_id}")
        sha, enc, size, servers = descriptor.blob_ref
        if not sha:
            return {}
        servers = [s for s in servers if s] or self.blossom_servers or card.blossom_servers
        try:
            raw = fetch_blob(sha, servers)
            pairs = decode_eta_blob(raw, card.groups, enc)
        except (RuntimeError, BlobError):
            return {}
        delta = {name: e for name, e in pairs}
        self.params.setdefault(model_id, {})[descriptor.author] = delta
        return delta

    def fetch_attestations(self, target: Optional[str] = None) -> list[Attestation]:
        filters = [{"kinds": [30102]}]
        if target:
            filters[0]["#p"] = [target]
        out = []
        for ev in self._fetch_events(filters):
            a = self._parse_attestation(ev)
            if a is not None:
                out.append(a)
        self.attestations = [a for a in self.attestations + out
                             if a.event_id not in {other.event_id for other in out}] or out
        return out

    # -- composition -------------------------------------------------------

    def compose_prior(self, card: ModelCard, members: list[Site],
                      p_table: dict[str, dict[str, float]]) -> dict[str, Eta]:
        """Per-group prior η0 + Σ p·Δη over members, excluding the caller.

        Raises DomainError when any group leaves its family domain; the
        protocol defines no recovery.
        """
        mine = self.my_pubkey()
        out: dict[str, Eta] = {}
        for g in card.groups:
            etas, ps = [], []
            for s in members:
                if mine and s.author == mine:
                    continue
                if s.delta_eta is None or g.name not in (s.delta_eta or {}):
                    continue
                etas.append(s.delta_eta[g.name])
                ps.append(p_table.get(s.author, {}).get(g.name, 0.5))
            base = card.eta0.get(g.name) or self._zero_group(g)
            try:
                out[g.name] = compose_prior(base, etas, ps)
            except ValueError as e:
                raise DomainError(str(e)) from e
        return out

    @staticmethod
    def compute_site(posterior: dict[str, Eta], cavity: dict[str, Eta]) -> dict[str, Eta]:
        """Δη = η_post − η_cavity per group: the node's site contribution."""
        return {g: posterior[g] - cavity[g] for g in posterior}

    # -- trust -------------------------------------------------------------

    def _attestation_prior(self, target: str, model_id: Optional[str],
                           group: Optional[str]) -> tuple[float, float]:
        if not self.attestations:
            return 1.0, 1.0
        relevant = []
        for a in self.attestations:
            if a.target != target:
                continue
            if model_id is not None and a.model_id not in (None, model_id):
                continue
            if group is not None and a.group not in (None, group):
                continue
            relevant.append((a.attester, a.target, a.p))
        attester_p = {}
        for a in self.attestations:
            attester_p.setdefault(a.attester, self.trust.get_p(a.attester, "attestation", 0.5))
        return resolve_attestation_prior(relevant, attester_p, target, self.kappa_r)

    def score_peers(self, card: ModelCard, prior: dict[str, Eta],
                    posterior: dict[str, Eta], members: list[Site],
                    p_table: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """One trust round, per group per peer.  Returns the new p table.

        BMR at unit weight against the leave-one-out cavity, attestation
        prior at κ_r, then the Loewner cap and corroboration on the site
        before scoring, and novelty as a post-scoring multiplier when
        configured.  Mirrors test/fior_sim.py's round loop for the local
        client.
        """
        mine = self.my_pubkey()
        peers = [s for s in members if not (mine and s.author == mine)]
        if not peers:
            return p_table
        groups = card.groups
        local_lik = {g.name: posterior[g.name] - prior[g.name] for g in groups}

        # model-scope novelty weight (span test is at model scope)
        nov = None
        if self.novelty:
            order = np.array([self.first_seen.get(p.event_id, float(i)) for i, p in enumerate(peers)])
            flat = [self._flatten_site(p.delta_eta or {}, groups) for p in peers]
            nov = _novelty_from_flattened(flat, order)

        corr = {}
        if self.corroborate_t0 > 0:
            for g in groups:
                etas = [p.delta_eta.get(g.name) for p in peers]
                if all(e is not None for e in etas):
                    corr[g.name] = Corroboration([e for e in etas if e is not None])
        # p per peer index
        new_pt = {p.author: dict(p_table.get(p.author, {})) for p in peers}
        for gi, g in enumerate(groups):
            prior_g = prior[g.name]
            lik_g = local_lik[g.name]
            etas = [p.delta_eta.get(g.name) for p in peers]
            for i, p in enumerate(peers):
                site_g = p.delta_eta.get(g.name)
                if site_g is None:
                    new_pt[p.author][g.name] = 0.0
                    continue
                p_n = p_table.get(p.author, {}).get(g.name, 0.5)
                csite = site_g
                if self.corroborate_t0 > 0 and corr.get(g.name):
                    w = corr[g.name].weights(i, np.array([p_table.get(q.author, {}).get(g.name, 0.5) for q in peers]), -1, self.corroborate_t0)
                    csite = corr[g.name].apply(i, site_g, w)
                if self.spectral_cap > 0:
                    ref_lam = prior_g.Lam - p_n * site_g.Lam
                    ref_lam = (ref_lam + ref_lam.T) / 2.0 if ref_lam.ndim == 2 else ref_lam
                    ref_eta = Eta(site_g.family, prior_g.h, ref_lam)
                    try:
                        csite = loewner_clip(csite, ref_eta, self.spectral_cap)
                    except np.linalg.LinAlgError:
                        csite = site_g
                p_full = prior_g + (1.0 - p_n) * csite
                eta_q = p_full + lik_g
                dF = bmr_delta_f(eta_q, p_full, csite)
                a, b = self._attestation_prior(p.author, card.id, g.name)
                new_p = p_from(dF, a, b)
                if nov is not None:
                    new_p *= nov[i]
                new_pt[p.author][g.name] = float(np.clip(new_p, 1e-10, 1 - 1e-10))
        # persist to the local trust table
        for author, gr in new_pt.items():
            for gname, pv in gr.items():
                self.trust.set_p(author, gname, pv)
        if self.trust_path:
            self.trust.save(self.trust_path)
        return new_pt

    def _flatten_site(self, delta: dict[str, Eta], groups: list[Group]) -> np.ndarray:
        parts = []
        for g in groups:
            e = delta.get(g.name)
            if e is None:
                continue
            if e.family == FAMILY_MVNORMAL:
                iu = np.triu_indices(e.dim)
                scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
                parts.append(np.concatenate([e.h, e.Lam[iu] * scale]))
            elif e.family in ONE_BLOCK_FAMILIES:
                parts.append(e.h)
            else:
                parts.append(np.concatenate([e.h, e.Lam]))
        return np.concatenate(parts) if parts else np.zeros(1)

    def score_one(self, card: ModelCard, prior: dict[str, Eta],
                  posterior: dict[str, Eta], site: Site,
                  p_table: dict[str, dict[str, float]]) -> dict[str, float]:
        """ΔF and p for a single peer across groups (debug/CLI use)."""
        # reuse score_peers with a one-peer member list
        newp = self.score_peers(card, prior, posterior, [site], p_table)
        return newp.get(site.author, {})

    # -- publication --------------------------------------------------------

    def publish_site(self, card: ModelCard, delta: dict[str, Eta],
                     members: list[CavityMember], encoding: str = "f64le",
                     servers: Optional[list[str]] = None, expiration: Optional[int] = None,
                     samples: Optional[int] = None, free_energy: Optional[float] = None,
                     duration_sec: Optional[float] = None) -> tuple[str, nostr.PubResult]:
        """Upload the Δη blob (to every server) and publish the site event."""
        if not self.private_key:
            raise ClientError("private_key is required to publish")
        servers = list(servers or self.blossom_servers)
        if not servers:
            raise ClientError("no Blossom server configured for upload")
        data = encode_eta_blob(card.groups, [(g.name, delta[g.name]) for g in card.groups], encoding)
        sha = compute_sha256(data)
        for server in servers:
            upload_blob(data, server, sha)
        x_ref = [sha, encoding, str(len(data)), *servers]
        ev = nostr.create_site_contribution_event(
            self.private_key,
            model_id=card.id,
            model_version=card.version,
            model_coordinate=card.coordinate,
            delta_eta_ref=x_ref,
            cavity_members=members,
            expiration=expiration,
            samples=samples,
            free_energy=free_energy,
            duration_sec=duration_sec,
        )
        res = self._publish(ev)
        return ev["id"], res

    def withdraw_site(self, model_id: str) -> Optional[tuple[str, nostr.PubResult]]:
        """Publish a NIP-09 deletion for the caller's site on a model."""
        if not self.private_key or not self.my_pubkey():
            raise ClientError("private_key is set to withdraw")
        targets = [d.event_id for d in self.sites.get(model_id, {}).values()
                   if d.author == self.my_pubkey()]
        if not targets:
            return None
        ev = nostr.create_deletion_event(self.private_key, targets)
        return ev["id"], self._publish(ev)

    def publish_attestation(self, target: str, p: float, model_id: Optional[str] = None,
                            group: Optional[str] = None, expiration: Optional[int] = None) -> tuple[str, nostr.PubResult]:
        if not self.private_key:
            raise ClientError("private_key is set to publish")
        card = self.models.get(model_id) if model_id else None
        ev = nostr.create_trust_attestation_event(
            self.private_key,
            target_pubkey=target,
            inclusion_probability=p,
            model_id=model_id,
            group=group,
            model_coordinate=card.coordinate if card else None,
            expiration=expiration,
        )
        return ev["id"], self._publish(ev)


def _novelty_from_flattened(flat_sites: list[np.ndarray], order: np.ndarray) -> np.ndarray:
    """Causal span novelty over already-flattened site vectors."""
    from .trust import novelty_weight

    class _Vec:
        pass

    # novelty_weight operates on Eta objects; operate directly on the vectors
    m = len(flat_sites)
    if m == 0:
        return np.ones(1)
    out = np.zeros(m)
    V = np.array(flat_sites)
    for n in range(m):
        older = np.where(order < order[n])[0]
        if len(older) == 0:
            out[n] = 1.0
            continue
        b = V[n]
        nb = float(np.linalg.norm(b))
        if nb == 0:
            out[n] = 0.0
            continue
        c, *_ = np.linalg.lstsq(V[older].T, b, rcond=None)
        out[n] = float(np.linalg.norm(b - V[older].T @ c) / nb)
    ref = float(np.median(out))
    if ref <= 1e-9:
        return np.ones(m)
    return np.clip(out / ref, 0.0, 1.0)