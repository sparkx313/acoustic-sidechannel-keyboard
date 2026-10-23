"""Offline WAV privacy scanner for keyboard-like acoustic transients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

FRAME_MS = 20.0
HOP_MS = 10.0
MERGE_GAP_MS = 50.0
MIN_HIGH_BAND_FRACTION = 0.25
MIN_SPECTRAL_FLATNESS = 0.10


def decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    """Decode little-endian integer PCM samples into normalized float64."""
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if sample_width == 3:
        triples = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float64) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise ValueError("Only 8, 16, 24, and 32-bit integer PCM WAV files are supported.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def frame_features(mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_length = max(1, round(sample_rate * FRAME_MS / 1000.0))
    hop_length = max(1, round(sample_rate * HOP_MS / 1000.0))
    starts = np.arange(0, len(mono), hop_length, dtype=np.int64)
    rms_values = np.empty(len(starts), dtype=np.float64)
    high_band_fractions = np.zeros(len(starts), dtype=np.float64)
    flatness_values = np.zeros(len(starts), dtype=np.float64)
    window = np.hanning(frame_length) if frame_length > 1 else np.ones(1)
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate)
    band = (frequencies >= 1500.0) & (frequencies <= min(8000.0, sample_rate / 2.0))

    for index, start in enumerate(starts):
        segment = mono[start : start + frame_length]
        rms_values[index] = math.sqrt(float(np.mean(segment * segment)))
        padded = np.zeros(frame_length, dtype=np.float64)
        padded[: len(segment)] = segment
        power = np.abs(np.fft.rfft(padded * window)) ** 2
        total_power = float(power.sum())
        if total_power > 0.0 and np.any(band):
            band_power = power[band] + 1e-15
            high_band_fractions[index] = float(power[band].sum() / total_power)
            flatness_values[index] = float(np.exp(np.mean(np.log(band_power))) / np.mean(band_power))

    ends = np.minimum(starts + frame_length, len(mono))
    return starts, ends, rms_values, np.column_stack((high_band_fractions, flatness_values))


def group_candidates(
    starts: np.ndarray,
    ends: np.ndarray,
    candidate_mask: np.ndarray,
    features: np.ndarray,
    sample_rate: int,
) -> list[dict[str, float]]:
    selected = np.flatnonzero(candidate_mask)
    if len(selected) == 0:
        return []
    max_gap_samples = round(sample_rate * MERGE_GAP_MS / 1000.0)
    groups: list[list[int]] = [[int(selected[0])]]
    for idx in selected[1:]:
        idx = int(idx)
        current = groups[-1]
        if starts[idx] - ends[current[-1]] <= max_gap_samples:
            current.append(idx)
        else:
            groups.append([idx])

    events = []
    for group in groups:
        first, last = group[0], group[-1]
        events.append({
            "start_sample": int(starts[first]),
            "end_sample": int(ends[last]),
            "start_seconds": round(float(starts[first] / sample_rate), 4),
            "end_seconds": round(float(ends[last] / sample_rate), 4),
            "peak_high_band_fraction": round(float(features[group, 0].max()), 4),
            "peak_spectral_flatness": round(float(features[group, 1].max()), 4),
        })
    return events


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find keyboard-like broadband transients in a PCM WAV and optionally silence their time windows."
    )
    parser.add_argument("input", type=Path, help="local PCM WAV file")
    parser.add_argument("--redacted-output", type=Path, help="write a WAV copy with detected windows silenced")
    parser.add_argument("--report", type=Path, help="also save the JSON report to this path")
    parser.add_argument("--threshold-dbfs", type=float, default=-36.0, help="minimum frame RMS threshold in dBFS (default: -36)")
    parser.add_argument("--sensitivity-mad", type=float, default=6.0, help="multiplier for robust per-file RMS variation (default: 6)")
    parser.add_argument("--padding-ms", type=float, default=40.0, help="silence this much audio around each detected window (default: 40)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output/report files")
    return parser


def ensure_distinct_paths(parser: argparse.ArgumentParser, source: Path, outputs: list[Path]) -> None:
    resolved_source = source.resolve()
    resolved_outputs = [path.expanduser().resolve() for path in outputs]
    if any(path == resolved_source for path in resolved_outputs):
        parser.error("output paths must not overwrite the source WAV")
    if len(set(resolved_outputs)) != len(resolved_outputs):
        parser.error("report and redacted-audio output paths must be different")


def write_redacted_wav(
    destination: Path,
    params: Any,
    original: bytes,
    events: list[dict[str, Any]],
    sample_rate: int,
    frame_width: int,
    padding_ms: float,
) -> list[dict[str, float]]:
    frame_count = params.nframes
    padding = round(sample_rate * padding_ms / 1000.0)
    silence_sample = b"\x80" if params.sampwidth == 1 else b"\x00"
    silence_frame = silence_sample * params.nchannels * params.sampwidth
    redacted = bytearray(original)
    redacted_windows = []
    for event in events:
        start = max(0, int(event["start_sample"]) - padding)
        end = min(frame_count, int(event["end_sample"]) + padding)
        redacted[start * frame_width : end * frame_width] = silence_frame * (end - start)
        redacted_windows.append({
            "start_seconds": round(start / sample_rate, 4),
            "end_seconds": round(end / sample_rate, 4),
        })
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as output:
        output.setparams(params)
        output.writeframes(bytes(redacted))
    return redacted_windows


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    source = args.input.expanduser()
    outputs = [path for path in (args.redacted_output, args.report) if path is not None]
    ensure_distinct_paths(parser, source, outputs)
    if args.padding_ms < 0:
        parser.error("--padding-ms must be zero or greater")
    if not math.isfinite(args.threshold_dbfs) or args.threshold_dbfs > 0 or args.threshold_dbfs < -120:
        parser.error("--threshold-dbfs must be between -120 and 0")
    if not math.isfinite(args.sensitivity_mad) or args.sensitivity_mad < 0:
        parser.error("--sensitivity-mad must be zero or greater")
    for destination in outputs:
        if destination.expanduser().exists() and not args.overwrite:
            parser.error(f"output already exists (use --overwrite): {destination}")

    try:
        with wave.open(str(source), "rb") as audio:
            params = audio.getparams()
            if params.comptype != "NONE":
                parser.error("compressed WAV audio is not supported; provide integer PCM")
            if params.nframes <= 0 or params.framerate <= 0:
                parser.error("input WAV must contain audio frames and a valid sample rate")
            raw = audio.readframes(params.nframes)
        samples = decode_pcm(raw, params.sampwidth)
        if params.nchannels < 1 or samples.size % params.nchannels or samples.size // params.nchannels != params.nframes:
            parser.error("WAV sample data does not match its channel count")
        channels = samples.reshape(-1, params.nchannels)
        mono = channels.mean(axis=1)
        starts, ends, rms, spectral = frame_features(mono, params.framerate)
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        parser.error(f"could not read PCM WAV: {exc}")

    median_rms = float(np.median(rms))
    mad_rms = float(np.median(np.abs(rms - median_rms)))
    adaptive_threshold = median_rms + args.sensitivity_mad * 1.4826 * mad_rms
    minimum_rms = 10.0 ** (args.threshold_dbfs / 20.0)
    threshold = max(minimum_rms, adaptive_threshold)
    candidate_mask = (
        (rms >= threshold)
        & (spectral[:, 0] >= MIN_HIGH_BAND_FRACTION)
        & (spectral[:, 1] >= MIN_SPECTRAL_FLATNESS)
    )
    events = group_candidates(starts, ends, candidate_mask, spectral, params.framerate)
    redacted_windows = []
    if args.redacted_output is not None:
        try:
            redacted_windows = write_redacted_wav(
                args.redacted_output.expanduser(), params, raw, events, params.framerate,
                params.nchannels * params.sampwidth, args.padding_ms
            )
        except (OSError, wave.Error) as exc:
            parser.error(f"could not write redacted WAV: {exc}")

    report: dict[str, Any] = {
        "tool": "offline-audio-privacy-scan",
        "source_file": source.name,
        "source_sha256": sha256_file(source),
        "sample_rate_hz": params.framerate,
        "channels": params.nchannels,
        "bits_per_sample": params.sampwidth * 8,
        "duration_seconds": round(params.nframes / params.framerate, 4),
        "frame_ms": FRAME_MS,
        "hop_ms": HOP_MS,
        "rms_threshold_dbfs": args.threshold_dbfs,
        "adaptive_rms_threshold": round(threshold, 8),
        "minimum_high_band_fraction": MIN_HIGH_BAND_FRACTION,
        "minimum_spectral_flatness": MIN_SPECTRAL_FLATNESS,
        "candidate_windows": events,
        "candidate_window_count": len(events),
        "redacted_windows": redacted_windows,
        "limitations": [
            "These are heuristic broadband transient candidates, not identified keystrokes.",
            "The tool never infers key identities or typed text and does not record or upload audio.",
            "False positives and missed sounds are expected; silence after scanning does not prove privacy.",
            "Redaction silences detected windows plus padding and may remove speech or other sounds nearby.",
        ],
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        try:
            report_path = args.report.expanduser()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(serialized, encoding="utf-8")
        except OSError as exc:
            parser.error(f"could not write JSON report: {exc}")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
