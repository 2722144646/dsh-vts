"""Command line entry points."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys
import time

from .audio.engine import make_clock
from .bridge.engine import AvatarEngine
from .bridge.server import run_bridge
from .config import ensure_dirs, load_config
from .ndi.capture import NDICapture, list_sources
from .vts.client import VTSClient
from .vts.probe import run_probe

log = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _cmd_probe(args) -> int:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    client = VTSClient(cfg.vts)
    try:
        profile = await run_probe(cfg, client, request_token=args.request_token)
    except Exception as exc:
        log.error("probe failed: %s", exc)
        print(
            "\nVTS probe failed. Check:\n"
            "  1. VTube Studio is running.\n"
            "  2. Settings -> API -> Allow Plugin API access is ON.\n"
            "  3. The port matches config.toml (default 8001).\n"
            "  4. If this is the first run, approve the plugin popup in VTS.\n",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(profile.summary(), ensure_ascii=False, indent=2))
    return 0


async def _cmd_demo(args) -> int:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.no_audio:
        cfg.audio.enabled = False
    engine = AvatarEngine(cfg, on_event=_print_event)
    await engine.start()
    try:
        await engine.begin_utterance("demo")
        await engine.feed_text(args.text)
        await engine.end_utterance()
        timeout = time.monotonic() + args.seconds
        while time.monotonic() < timeout:
            await asyncio.sleep(0.1)
    finally:
        await engine.close()
    return 0


async def _print_event(payload: dict) -> None:
    if payload.get("type") in {"ready", "playback", "error", "cancelled"}:
        print(json.dumps(payload, ensure_ascii=False))


async def _cmd_serve(args) -> int:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.no_audio:
        cfg.audio.enabled = False
    await run_bridge(cfg)
    return 0


def _cmd_ndi(args) -> int:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.list:
        sources = list_sources(args.wait_ms)
        print(json.dumps(sources, ensure_ascii=False, indent=2))
        return 0
    capture = NDICapture(source_hint=args.source or cfg.ndi.source, snapshot_dir=cfg.snapshot_dir)
    try:
        if not capture.open(wait_ms=args.wait_ms):
            print("No NDI sources found. Enable NDI output in VTS first.", file=sys.stderr)
            return 2
        metrics = capture.snapshot()
        if metrics is None:
            print("NDI source found but no video frame arrived.", file=sys.stderr)
            return 3
        print(json.dumps(metrics.__dict__, ensure_ascii=False, indent=2))
        return 0
    finally:
        capture.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dsh-vts", description="DSH -> VTube Studio avatar bridge")
    parser.add_argument("--config", help="path to config.toml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="probe VTS parameters and write data/contract.json")
    probe.add_argument("--request-token", action="store_true", help="force a new VTS plugin token popup")

    serve = sub.add_parser("serve", help="run the local WebSocket bridge")
    serve.add_argument("--no-audio", action="store_true")

    demo = sub.add_parser("demo", help="run one line through TTS/VTS without DSH")
    demo.add_argument("text", nargs="?", default="你好，我是 dsh。现在，我正在用 VTube Studio 和你说话。")
    demo.add_argument("--seconds", type=float, default=6.0)
    demo.add_argument("--no-audio", action="store_true")

    ndi = sub.add_parser("ndi", help="list or capture the VTS NDI feed")
    ndi.add_argument("--list", action="store_true")
    ndi.add_argument("--source", default="")
    ndi.add_argument("--wait-ms", type=int, default=3000)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        if args.command == "probe":
            return asyncio.run(_cmd_probe(args))
        if args.command == "serve":
            return asyncio.run(_cmd_serve(args))
        if args.command == "demo":
            return asyncio.run(_cmd_demo(args))
        if args.command == "ndi":
            return _cmd_ndi(args)
    except KeyboardInterrupt:
        return 130
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
