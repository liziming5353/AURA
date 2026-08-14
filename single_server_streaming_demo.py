#!/usr/bin/env python3
"""
Single-machine streaming interaction demo (no frontend/ASR/TTS services).

This script talks directly to the main TCP server protocol:
  - Type 1: video payload (WebM/MP4 bytes)
  - Type 4: clear context
  - Type 6: start camera/reset
  - Type 11: text prompt (simulated ASR text)
It then consumes streaming token messages (Type 8) and prints them in real time.

It also simulates TTS behavior locally by:
  - chunking finalized model text by sentence
  - printing sentence playback events with configurable cps
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Client -> Server
VIDEO_TYPE = 1
CLEAR_CONTEXT_TYPE = 4
START_CAMERA_TYPE = 6
TEXT_PROMPT_TYPE = 11

# Server -> Client
STREAMING_TOKEN_TYPE = 8
ERROR_TYPE = 7
ASR_QUERY_ECHO_TYPE = 10


@dataclass
class ScenarioEvent:
    at_s: float
    kind: str
    payload: str | None = None


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        part = sock.recv(n - len(chunks))
        if not part:
            raise ConnectionError("socket closed while receiving data")
        chunks.extend(part)
    return bytes(chunks)


def send_message(sock: socket.socket, msg_type: int, payload: bytes) -> None:
    sock.sendall(struct.pack(">BQ", msg_type, len(payload)) + payload)


def split_sentences(text: str) -> list[str]:
    separators = "。！？!?；;\n"
    result: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in separators:
            sentence = "".join(buf).strip()
            if sentence:
                result.append(sentence)
            buf.clear()
    tail = "".join(buf).strip()
    if tail:
        result.append(tail)
    return result


def build_scenario(events_path: Path | None) -> list[ScenarioEvent]:
    if events_path is None:
        # Default timeline: ask at several points while sending video.
        return [
            ScenarioEvent(0.0, "start_camera"),
            ScenarioEvent(0.5, "send_video"),
            ScenarioEvent(2.0, "ask", "先描述你目前看到的场景。"),
            ScenarioEvent(5.0, "send_video"),
            ScenarioEvent(7.0, "ask", "和两秒前相比，画面里有什么变化？"),
            ScenarioEvent(11.0, "send_video"),
            ScenarioEvent(13.0, "ask", "请只用一句话给出当前最重要提醒。"),
        ]

    raw = json.loads(events_path.read_text(encoding="utf-8"))
    events: list[ScenarioEvent] = []
    for item in raw:
        events.append(
            ScenarioEvent(
                at_s=float(item["at_s"]),
                kind=str(item["kind"]),
                payload=item.get("payload"),
            )
        )
    events.sort(key=lambda x: x.at_s)
    return events


def run_tts_simulator(text: str, cps: float) -> None:
    sentences = split_sentences(text)
    if not sentences:
        return
    for idx, sent in enumerate(sentences, start=1):
        duration = max(len(sent) / max(cps, 1.0), 0.3)
        print(f"\n[TTS_SIM] sentence#{idx} start ({duration:.2f}s): {sent}")
        time.sleep(min(duration, 2.0))
        print(f"[TTS_SIM] sentence#{idx} end")


def schedule_events(
    sock: socket.socket,
    events: Iterable[ScenarioEvent],
    video_bytes: bytes,
    done_flag: threading.Event,
) -> None:
    t0 = time.monotonic()
    for ev in events:
        while not done_flag.is_set():
            now = time.monotonic() - t0
            remain = ev.at_s - now
            if remain <= 0:
                break
            time.sleep(min(remain, 0.05))
        if done_flag.is_set():
            return

        if ev.kind == "start_camera":
            print("\n[SCENARIO] start_camera/reset")
            send_message(sock, START_CAMERA_TYPE, b"start")
        elif ev.kind == "clear_context":
            print("\n[SCENARIO] clear_context")
            send_message(sock, CLEAR_CONTEXT_TYPE, b"clear")
        elif ev.kind == "send_video":
            print(f"\n[SCENARIO] send_video ({len(video_bytes)} bytes)")
            send_message(sock, VIDEO_TYPE, video_bytes)
        elif ev.kind == "ask":
            prompt = (ev.payload or "").strip()
            print(f"\n[SCENARIO] ask: {prompt}")
            send_message(sock, TEXT_PROMPT_TYPE, prompt.encode("utf-8"))
        else:
            print(f"\n[SCENARIO] ignore unknown event kind: {ev.kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-server streaming interaction demo")
    parser.add_argument("--host", default="127.0.0.1", help="main server host")
    parser.add_argument("--port", type=int, default=12345, help="main server port")
    parser.add_argument("--video", required=True, help="path to demo video file")
    parser.add_argument(
        "--scenario-json",
        default=None,
        help="optional JSON scenario file: [{\"at_s\":2.0,\"kind\":\"ask\",\"payload\":\"...\"}]",
    )
    parser.add_argument("--cps", type=float, default=7.0, help="TTS simulation chars/sec")
    parser.add_argument(
        "--auto-stop-after-final",
        action="store_true",
        help="stop when first final model answer completes",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    video_bytes = video_path.read_bytes()
    if len(video_bytes) < 1000:
        raise ValueError("video file too small; server requires >=1000 bytes for valid container")

    scenario_path = Path(args.scenario_json) if args.scenario_json else None
    events = build_scenario(scenario_path)

    done = threading.Event()
    answer_buffers: dict[str, list[str]] = {}

    with socket.create_connection((args.host, args.port), timeout=30) as sock:
        sock.settimeout(None)
        print(f"[CONNECT] connected to {args.host}:{args.port}")

        scheduler = threading.Thread(
            target=schedule_events,
            args=(sock, events, video_bytes, done),
            daemon=True,
        )
        scheduler.start()

        try:
            while True:
                header = recv_exact(sock, 9)
                msg_type, msg_len = struct.unpack(">BQ", header)
                payload = recv_exact(sock, msg_len)

                if msg_type == STREAMING_TOKEN_TYPE:
                    token_data = json.loads(payload.decode("utf-8", errors="ignore"))
                    response_id = token_data.get("response_id", "unknown")
                    token = token_data.get("token", "")
                    is_final = bool(token_data.get("is_final", False))
                    is_start = bool(token_data.get("is_start", False))
                    is_silent = bool(token_data.get("is_silent", False))

                    if is_start:
                        print(f"\n[MODEL] response_id={response_id} stream_start")

                    if token:
                        print(token, end="", flush=True)
                        answer_buffers.setdefault(response_id, []).append(token)

                    if is_final:
                        full_text = "".join(answer_buffers.get(response_id, []))
                        if is_silent:
                            print("\n[MODEL] final: <silent>")
                        else:
                            print("\n[MODEL] final complete")
                            run_tts_simulator(full_text, args.cps)
                        if args.auto_stop_after_final:
                            break

                elif msg_type == ASR_QUERY_ECHO_TYPE:
                    data = json.loads(payload.decode("utf-8", errors="ignore"))
                    query = data.get("query", "")
                    print(f"\n[ASR_SIM] query_echo: {query}")

                elif msg_type == ERROR_TYPE:
                    err = payload.decode("utf-8", errors="ignore")
                    print(f"\n[SERVER_ERROR] {err}")
                    break

                else:
                    print(f"\n[INFO] recv msg_type={msg_type}, bytes={msg_len}")

        finally:
            done.set()
            scheduler.join(timeout=1.0)
            print("\n[EXIT] demo finished")


if __name__ == "__main__":
    main()
