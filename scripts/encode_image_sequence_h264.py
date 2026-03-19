#!/usr/bin/env python3
"""Encode an Isaac Sim PNG image sequence into an H.264 MP4."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_INPUT_ROOT = Path(
    "/home/jonas/Drone/Drone/logs/skrl/"
    "drone_target_touch_vehicle_moving_stage4/"
    "2026-03-10_16-27-59_ppo_torch/image_sequences/play"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode Isaac Sim image-sequence frames into an H.264 MP4."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=(
            "Input frame directory or parent directory that contains timestamped image-sequence folders. "
            "Defaults to the Stage4 Taipei demo image-sequence play output root."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path. Defaults to <resolved_input_dir>.mp4",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Output video FPS. Defaults to 50.0 to match the environment step rate.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 CRF quality. Lower is higher quality. Defaults to 18.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="slow",
        help="libx264 preset. Defaults to slow.",
    )
    parser.add_argument(
        "--pix-fmt",
        type=str,
        default="yuv420p",
        help="Output pixel format. Defaults to yuv420p for wide compatibility.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="rgb_*.png",
        help="Frame glob pattern inside the resolved input directory. Defaults to rgb_*.png.",
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=None,
        help="Optional explicit ffmpeg binary path. If omitted, the script looks for ffmpeg in PATH.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the resolved paths and ffmpeg command without encoding.",
    )
    return parser.parse_args()


def resolve_input_dir(input_path: Path, pattern: str) -> Path:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_dir():
        direct_frames = sorted(input_path.glob(pattern))
        if direct_frames:
            return input_path

        candidate_dirs = sorted(
            [path for path in input_path.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidate_dirs:
            if sorted(candidate.glob(pattern)):
                return candidate

    raise FileNotFoundError(
        f"No frames matching '{pattern}' found in {input_path} or its immediate child directories."
    )


def collect_frames(frame_dir: Path, pattern: str) -> list[Path]:
    frames = sorted(frame_dir.glob(pattern))
    if not frames:
        raise FileNotFoundError(f"No frames matching '{pattern}' found in {frame_dir}")
    return frames


def ensure_ffmpeg_available(ffmpeg_override: Path | None) -> str:
    if ffmpeg_override is not None:
        ffmpeg_path = str(ffmpeg_override.expanduser().resolve())
        if not Path(ffmpeg_path).exists():
            raise FileNotFoundError(f"Explicit ffmpeg path does not exist: {ffmpeg_path}")
        return ffmpeg_path

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise FileNotFoundError("ffmpeg not found in PATH")
    return ffmpeg_path


def build_ffmpeg_command(
    ffmpeg_path: str,
    sequence_pattern: Path,
    output_path: Path,
    fps: float,
    preset: str,
    crf: int,
    pix_fmt: str,
) -> list[str]:
    return [
        ffmpeg_path,
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(sequence_pattern),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        pix_fmt,
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def main() -> int:
    args = parse_args()
    frame_dir = resolve_input_dir(args.input_dir, args.pattern)
    frames = collect_frames(frame_dir, args.pattern)
    output_path = args.output.expanduser().resolve() if args.output else frame_dir.with_suffix(".mp4")
    ffmpeg_path = "ffmpeg" if args.dry_run and args.ffmpeg is None else ensure_ffmpeg_available(args.ffmpeg)

    print(f"[INFO] Resolved input frame directory: {frame_dir}")
    print(f"[INFO] Found {len(frames)} frames matching {args.pattern}")
    print(f"[INFO] Output video path: {output_path}")

    with tempfile.TemporaryDirectory(prefix="isaac_frames_", dir="/tmp") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for index, source_path in enumerate(frames):
            link_path = temp_dir / f"frame_{index:06d}{source_path.suffix.lower()}"
            try:
                os.link(source_path, link_path)
            except OSError:
                os.symlink(source_path, link_path)

        frame_suffix = frames[0].suffix.lower()
        ffmpeg_input_pattern = temp_dir / f"frame_%06d{frame_suffix}"
        command = build_ffmpeg_command(
            ffmpeg_path=ffmpeg_path,
            sequence_pattern=ffmpeg_input_pattern,
            output_path=output_path,
            fps=args.fps,
            preset=args.preset,
            crf=args.crf,
            pix_fmt=args.pix_fmt,
        )

        print("[INFO] ffmpeg command:")
        print(" ".join(command))
        if args.dry_run:
            return 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)

    print(f"[INFO] H.264 video written to: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
