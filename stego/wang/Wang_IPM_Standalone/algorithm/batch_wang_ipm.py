#!/usr/bin/env python3
"""Batch Wang IPM embedding: same folder conventions as Tar5Stego.

Outputs: <output_root>/payload_0.1/<segment>/<segment>_stego_IPM.txt etc.
Does NOT overwrite the original Tar5 STC results by default.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cover-root", required=True, help="Segment folders containing <segment>_cover_ipm.txt")
    p.add_argument("--output-root", default="Wangstegresults/480p/32_steg")
    p.add_argument("--payload-ratios", nargs="+", type=float, default=[0.1,0.3,0.5])
    p.add_argument("--ddg-root", help="Optional Tar5 DDG root; expected <segment>/<segment>/frame_*_ddg.json")
    p.add_argument("--intra-period", type=int, default=4)
    p.add_argument("--first-i-frame", type=int, default=1)
    p.add_argument("--all-frames", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict-rd", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cover_root = Path(args.cover_root)
    if not cover_root.is_dir():
        p.error(f"cover-root does not exist: {cover_root}")
    segments = sorted(d for d in cover_root.iterdir() if d.is_dir())
    if not segments:
        p.error("No segment subdirectories found")
    algorithm = Path(__file__).with_name("wang_ipm.py")
    if not algorithm.is_file():
        p.error(f"Missing {algorithm}; keep both .py files in the same folder")

    done=skipped=failed=0
    for ratio in args.payload_ratios:
        if not 0 <= ratio <= 1:
            p.error(f"Invalid payload ratio: {ratio}")
        for seg in segments:
            cover_file = seg / f"{seg.name}_cover_ipm.txt"
            if not cover_file.is_file():
                cover_file = seg / f"{seg.name}_cover_IPM.txt"  # case-sensitive FS support
            if not cover_file.is_file():
                print(f"[SKIP] cover IPM not found: {seg.name}")
                skipped+=1
                continue
            output = Path(args.output_root) / f"payload_{ratio}" / seg.name / f"{seg.name}_stego_IPM.txt"
            if output.is_file() and output.stat().st_size > 0 and not args.overwrite:
                print(f"[SKIP] exists: {output}")
                skipped+=1
                continue
            cmd=[sys.executable,str(algorithm),"embed","--cover",str(cover_file),
                 "--output",str(output),"--payload-ratio",str(ratio),
                 "--intra-period",str(args.intra_period),"--first-i-frame",str(args.first_i_frame),
                 "--seed",str(args.seed)]
            if args.all_frames:
                cmd.append("--all-frames")
            if args.strict_rd:
                cmd.append("--strict-rd")
            if args.ddg_root:
                ddir = Path(args.ddg_root) / seg.name / seg.name
                if ddir.is_dir():
                    cmd += ["--ddg-dir",str(ddir)]
                else:
                    print(f"[WARN] no DDG for {seg.name}; will use nearest fallback")
            print(f"[RUN] {seg.name}, payload={ratio}")
            code=subprocess.run(cmd).returncode
            if code == 0:
                done+=1
            else:
                failed+=1
                print(f"[FAIL] {seg.name}, payload={ratio}: exit={code}")
    print(f"Complete. done={done} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
