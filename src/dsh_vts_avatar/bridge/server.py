"""Local WebSocket bridge consumed by the dsh Host plugin."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import orjson
import websockets
from websockets.asyncio.server import Server, ServerConnection, serve

from ..config import Config
from .engine import AvatarEngine

log = logging.getLogger(__name__)


class BridgeServer:
    def __init__(self, config: Config):
        self.config = config
        self.engine = AvatarEngine(config, on_event=self.broadcast)
        self.clients: set[ServerConnection] = set()
        self._server: Server | None = None
        self._send_lock = asyncio.Lock()

    async def start(self) -> None:
        await self.engine.start()
        self._server = await serve(
            self._handler,
            self.config.bridge.host,
            self.config.bridge.port,
            max_size=2 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )
        log.info("bridge listening on ws://%s:%d", self.config.bridge.host, self.config.bridge.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        await self.engine.close()

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self.clients:
            return
        raw = orjson.dumps(payload).decode("utf-8")
        async with self._send_lock:
            stale: list[ServerConnection] = []
            for client in list(self.clients):
                try:
                    await client.send(raw)
                except Exception:
                    stale.append(client)
            for client in stale:
                self.clients.discard(client)

    async def _handler(self, ws: ServerConnection) -> None:
        self.clients.add(ws)
        peer = getattr(ws, "remote_address", None)
        log.info("plugin connected: %s", peer)
        try:
            await ws.send(
                orjson.dumps(
                    {
                        "type": "hello.ack",
                        "protocol": 1,
                        "status": self.engine.status(),
                    }
                ).decode("utf-8")
            )
            async for raw in ws:
                try:
                    msg = orjson.loads(raw)
                except Exception:
                    await ws.send(orjson.dumps({"type": "error", "code": "bad_json"}).decode("utf-8"))
                    continue
                try:
                    await self._dispatch(ws, msg)
                except Exception as exc:
                    log.exception("bridge dispatch failed")
                    await ws.send(
                        orjson.dumps(
                            {
                                "type": "error",
                                "code": "dispatch_failed",
                                "message": str(exc),
                                "retryable": True,
                            }
                        ).decode("utf-8")
                    )
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            log.info("plugin disconnected: %s", peer)

    async def _dispatch(self, ws: ServerConnection, msg: dict[str, Any]) -> None:
        mtype = str(msg.get("type", ""))
        if mtype == "hello":
            await ws.send(
                orjson.dumps(
                    {
                        "type": "ready",
                        "protocol": 1,
                        "status": self.engine.status(),
                    }
                ).decode("utf-8")
            )
        elif mtype == "utterance.start":
            log.info("utterance.start %s", msg.get("utterance_id"))
            await self.engine.begin_utterance(str(msg.get("utterance_id") or ""))
            await ws.send(
                orjson.dumps(
                    {"type": "utterance.started", "utterance_id": self.engine.utterance_id}
                ).decode("utf-8")
            )
        elif mtype == "text.delta":
            text = str(msg.get("text") or "")
            log.info("text.delta %d chars", len(text))
            await self.engine.feed_text(text)
        elif mtype == "utterance.end":
            log.info("utterance.end %s", msg.get("utterance_id"))
            await self.engine.end_utterance()
        elif mtype == "cancel":
            await self.engine.cancel()
        elif mtype == "state":
            self.engine.set_state(str(msg.get("state") or "idle"))
        elif mtype == "action":
            self.engine.add_action(str(msg.get("kind") or "nod"), float(msg.get("intensity", 0.7)))
        elif mtype == "head_demo":
            ok = self.engine.demo_head(
                curve=str(msg.get("curve") or "minjerk"),
                intensity=float(msg.get("intensity", 0.5)),
                duration_ms=int(msg.get("duration_ms", 900)),
            )
            await ws.send(
                orjson.dumps({"type": "head_demo.result", "ok": ok}).decode("utf-8")
            )
        elif mtype == "reconnect":
            ok = await self.engine.reconnect()
            await ws.send(orjson.dumps({"type": "reconnected", "ok": ok}).decode("utf-8"))
        elif mtype == "snapshot":
            metrics = await self.engine.snapshot()
            await ws.send(
                orjson.dumps(
                    {
                        "type": "snapshot.result",
                        "request_id": msg.get("request_id"),
                        "metrics": metrics,
                    }
                ).decode("utf-8")
            )
        elif mtype == "status":
            await ws.send(orjson.dumps({"type": "status", "status": self.engine.status()}).decode("utf-8"))
        elif mtype == "ping":
            await ws.send(orjson.dumps({"type": "pong"}).decode("utf-8"))
        else:
            await ws.send(
                orjson.dumps({"type": "error", "code": "unknown_type", "message": mtype}).decode("utf-8")
            )


async def run_bridge(config: Config) -> None:
    server = BridgeServer(config)
    await server.start()
    try:
        await asyncio.Future()
    finally:
        await server.stop()
