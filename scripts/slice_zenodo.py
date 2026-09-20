#!/usr/bin/env python3
"""Streaming slicer for Zenodo ATA RFI .sigmf tar archives.

Slices giant continuous recordings (~140M samples, 560MB) into standardized,
lightweight burst frames (40,960 samples = 160KB each, matching data0 exactly).

Features:
- Pure streaming read directly from tar (zero full-file disk unpacking).
- Peak memory footprint < 5 MB.
- Writes valid SigMF metadata pairs (.sigmf-data + .sigmf-meta) for each frame.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Optional


SAMPLES_PER_FRAME_DEFAULT = 40960  # Exactly matches data0 (MIT RF Challenge)
BYTES_PER_SAMPLE_CI16 = 4          # 2 bytes I (int16) + 2 bytes Q (int16)


def slice_archive(
    tar_path: Path,
    out_dir: Path,
    samples_per_frame: int = SAMPLES_PER_FRAME_DEFAULT,
    max_frames: Optional[int] = 50,
    stride_frames: int = 1,
    dry_run: bool = False,
) -> dict:
    """Slice a single Zenodo .sigmf tar archive into standard SigMF burst frames.

    Args:
        tar_path: Path to Zenodo .sigmf tar file.
        out_dir: Target directory for sliced frames.
        samples_per_frame: Number of I/Q samples per frame (default: 40960).
        max_frames: Max frames to extract (None for all).
        stride_frames: Extract every N-th frame (to sample across time).
        dry_run: If True, simulate without writing files.

    Returns:
        Summary dict with extraction statistics.
    """
    frame_bytes = samples_per_frame * BYTES_PER_SAMPLE_CI16
    capture_name = tar_path.stem

    target_dir = out_dir / capture_name
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r") as tf:
        members = tf.getmembers()
        meta_member = next((m for m in members if m.name.endswith(".sigmf-meta")), None)
        data_member = next((m for m in members if m.name.endswith(".sigmf-data")), None)

        if meta_member is None or data_member is None:
            raise ValueError(f"Invalid Zenodo archive {tar_path.name}: missing .sigmf-meta or .sigmf-data")

        # Read base metadata
        meta_file = tf.extractfile(meta_member)
        if meta_file is None:
            raise RuntimeError(f"Could not read metadata from {tar_path.name}")
        base_meta = json.load(meta_file)

        # Extract capture parameters
        global_meta = base_meta.get("global", {})
        sample_rate = float(global_meta.get("core:sample_rate", 61440000.0))
        captures_list = base_meta.get("captures", [{}])
        first_capture = captures_list[0] if captures_list else {}
        center_freq = first_capture.get("core:frequency") or global_meta.get("core:frequency", 0.0)
        start_datetime = first_capture.get("core:datetime") or first_capture.get("core.datetime", "")
        author = global_meta.get("core:author", "Daniel Estevez")
        recorder = global_meta.get("core:recorder", "Maia SDR")

        total_data_bytes = data_member.size
        total_possible_frames = total_data_bytes // frame_bytes

        print(f"[{capture_name}]")
        print(f"  Archive size: {tar_path.stat().st_size / 1e6:.1f} MB | Total samples: {total_data_bytes // BYTES_PER_SAMPLE_CI16:,}")
        print(f"  Sample rate: {sample_rate / 1e6:.2f} MHz | Center freq: {center_freq / 1e6:.2f} MHz")
        print(f"  Target frame size: {samples_per_frame:,} samples ({frame_bytes / 1024:.1f} KB)")
        print(f"  Available frames: {total_possible_frames:,} | Max to extract: {max_frames or 'ALL'} | Stride: {stride_frames}")

        data_stream = tf.extractfile(data_member)
        if data_stream is None:
            raise RuntimeError(f"Could not open data stream from {tar_path.name}")

        extracted_count = 0
        current_frame_idx = 0

        while True:
            if max_frames is not None and extracted_count >= max_frames:
                break

            chunk = data_stream.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break  # Reached end of stream

            # Check if this frame should be kept based on stride
            if current_frame_idx % stride_frames == 0:
                frame_label = f"frame_{extracted_count:04d}"
                data_filename = f"{frame_label}.sigmf-data"
                meta_filename = f"{frame_label}.sigmf-meta"

                # Calculate sample offset in continuous recording
                sample_offset = current_frame_idx * samples_per_frame
                time_offset_sec = sample_offset / sample_rate

                # Construct clean, data0-compatible metadata
                frame_meta = {
                    "global": {
                        "core:author": f"{author} (sliced for SIH 26147)",
                        "core:datatype": "ci16_le",
                        "core:description": f"ATA RFI capture {capture_name} slice #{extracted_count} (offset {sample_offset:,} samples, +{time_offset_sec:.3f}s)",
                        "core:recorder": recorder,
                        "core:sample_rate": sample_rate,
                        "core:version": "1.0.0",
                    },
                    "captures": [
                        {
                            "core:datetime": start_datetime,
                            "core:frequency": float(center_freq),
                            "core:sample_start": 0,  # Relative to frame file
                        }
                    ],
                    "annotations": [],
                    "_source_slice": {
                        "archive": tar_path.name,
                        "capture": capture_name,
                        "slice_index": extracted_count,
                        "original_frame_index": current_frame_idx,
                        "original_sample_offset": sample_offset,
                        "original_time_offset_seconds": time_offset_sec,
                        "samples": samples_per_frame,
                    },
                }

                if not dry_run:
                    # Write binary .sigmf-data
                    out_data_path = target_dir / data_filename
                    with open(out_data_path, "wb") as df:
                        df.write(chunk)

                    # Write JSON .sigmf-meta
                    out_meta_path = target_dir / meta_filename
                    with open(out_meta_path, "w", encoding="utf-8") as mf:
                        json.dump(frame_meta, mf, indent=2)

                extracted_count += 1

            current_frame_idx += 1

            # Skip ahead if stride > 1
            if stride_frames > 1:
                skip_bytes = (stride_frames - 1) * frame_bytes
                data_stream.seek(skip_bytes, os.SEEK_CUR)
                current_frame_idx += (stride_frames - 1)

    print(f"  Done: Extracted {extracted_count} frames -> {target_dir}")
    return {
        "capture": capture_name,
        "frames_extracted": extracted_count,
        "samples_per_frame": samples_per_frame,
        "bytes_per_frame": frame_bytes,
        "target_dir": str(target_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Slice Zenodo .sigmf tar archives into standard SigMF burst frames")
    parser.add_argument("--input", "-i", default=r"D:\dataset\zenodo", help="Zenodo tar file or directory of archives")
    parser.add_argument("--output", "-o", default=r"D:\dataset\zenodo_frames", help="Output directory for sliced frames")
    parser.add_argument("--samples-per-frame", "-s", type=int, default=SAMPLES_PER_FRAME_DEFAULT, help=f"Samples per frame (default: {SAMPLES_PER_FRAME_DEFAULT})")
    parser.add_argument("--max-frames", "-m", type=int, default=20, help="Max frames per capture (default: 20, use 0 for all)")
    parser.add_argument("--stride", type=int, default=50, help="Frame stride (default: 50, samples across capture)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing files")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output)
    max_frames = None if args.max_frames == 0 else args.max_frames

    if input_path.is_file():
        archives = [input_path]
    elif input_path.is_dir():
        archives = sorted(input_path.glob("*.sigmf"))
        if not archives:
            archives = sorted(input_path.rglob("*.sigmf"))
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"Found {len(archives)} Zenodo archive(s) to process.")
    results = []
    for arch in archives:
        res = slice_archive(
            arch,
            out_dir=out_dir,
            samples_per_frame=args.samples_per_frame,
            max_frames=max_frames,
            stride_frames=args.stride,
            dry_run=args.dry_run,
        )
        results.append(res)

    total_frames = sum(r["frames_extracted"] for r in results)
    print(f"\n==================================================")
    print(f"SUCCESS: Generated {total_frames} standardized burst frames across {len(results)} capture(s)")
    print(f"Output location: {out_dir}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
