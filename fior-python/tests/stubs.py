"""Minimal in-process Nostr relay + Blossom server for FIOR tests.

Implements just enough NIP-01 to exercise the client: addressable-replacement
semantics (one event per (kind, pubkey, d)), single-letter-index filters,
EOSE, OK with duplicate/rejected reasons, and NIP-09 deletion.  No
persistence, no auth, no pagination.
"""

import asyncio
import hashlib
import http.server
import json
import socket
import threading
import time

import websockets


def event_id(ev: dict) -> str:
    ser = json.dumps(
        [0, ev["pubkey"], ev["created_at"], ev["kind"], ev["tags"], ev["content"]],
        separators=(",", ":"),
    )
    return hashlib.sha256(ser.encode()).hexdigest()


def _d_value(ev: dict) -> str:
    for t in ev.get("tags", []):
        if t and t[0] == "d":
            return t[1] if len(t) > 1 else ""
    return ""


def _match(ev: dict, f: dict) -> bool:
    if "kinds" in f and ev["kind"] not in f["kinds"]:
        return False
    if "authors" in f and ev["pubkey"] not in f["authors"]:
        return False
    for k in ("#d", "#e", "#p", "#a", "#t", "#G", "#F"):
        if k in f:
            want = set(f[k])
            got = {t[1] for t in ev.get("tags", []) if t and t[0] == k[1:] and len(t) > 1}
            if not (got & want):
                return False
    if "since" in f and int(ev.get("created_at", 0)) < int(f["since"]):
        return False
    if "until" in f and int(ev.get("created_at", 0)) > int(f["until"]):
        return False
    return True


class StubRelay:
    """WebSocket relay with replaceable-address event semantics."""

    def __init__(self):
        self.events: list[dict] = []
        self.address: dict[tuple, dict] = {}   # (kind, pubkey, d) -> live event
        self.deleted: set[str] = set()
        self.port = None

    async def handler(self, ws):
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, list) or not msg:
                continue
            if msg[0] == "EVENT" and len(msg) > 1:
                ok, reason = self._accept(msg[1])
                await ws.send(json.dumps(["OK", msg[1].get("id", ""), ok, reason]))
            elif msg[0] == "REQ" and len(msg) > 2:
                sub = msg[1]
                filters = msg[2:]
                limit = None
                for f in filters:
                    if "limit" in f:
                        limit = int(f["limit"])
                matched = [ev for ev in self.events
                           if ev["id"] not in self.deleted and any(_match(ev, f) for f in filters)]
                matched.sort(key=lambda e: -e.get("created_at", 0))
                if limit is not None:
                    matched = matched[:limit]
                for ev in matched:
                    await ws.send(json.dumps(["EVENT", sub, ev]))
                await ws.send(json.dumps(["EOSE", sub]))
            elif msg[0] == "CLOSE":
                pass

    def _accept(self, ev: dict):
        if not isinstance(ev, dict) or "id" not in ev or "pubkey" not in ev:
            return False, "invalid: malformed event"
        if ev["id"] != event_id(ev):
            return False, "invalid: bad event id"
        if ev["kind"] == 5:
            for t in ev.get("tags", []):
                if t and t[0] == "e" and len(t) > 1:
                    self.deleted.add(t[1])
            self.events.append(ev)
            return True, ""
        addr = (ev["kind"], ev["pubkey"], _d_value(ev))
        prev = self.address.get(addr)
        if prev is not None and int(ev.get("created_at", 0)) <= int(prev.get("created_at", 0)):
            return False, "duplicate: not newer than the current replacement"
        if prev is not None:
            self.events = [e for e in self.events if e is not prev]
        self.address[addr] = ev
        self.events.append(ev)
        return True, ""

    async def _serve(self):
        async with websockets.serve(self.handler, "127.0.0.1", 0) as server:
            self.port = server.sockets[0].getsockname()[1]
            try:
                await asyncio.Future()
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass

    def start(self) -> str:
        t = threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True)
        t.start()
        for _ in range(200):
            if self.port:
                break
            time.sleep(0.01)
        if not self.port:
            raise RuntimeError("stub relay failed to bind")
        return f"ws://127.0.0.1:{self.port}"


class BlobHandler(http.server.BaseHTTPRequestHandler):
    store: dict[str, bytes] = {}

    def do_POST(self):
        if self.path != "/upload":
            return self.send_error(404)
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        sha = hashlib.sha256(data).hexdigest()
        self.store[sha] = data
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"sha256": sha}).encode())

    def do_GET(self):
        sha = self.path.lstrip("/")
        data = self.store.get(sha)
        if data is None:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


def start_blossom(out_store: dict) -> str:
    BlobHandler.store = out_store
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), BlobHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return f"http://127.0.0.1:{srv.server_port}"