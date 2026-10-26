# -*- coding: utf-8 -*-
"""Batch Yang HEVC stego encoder using the bundled executable and configuration.

Input YUV420 files are named <name>_<width>x<height>_<fps>.yuv. Pass --input,
--out and --msg to choose the data directory, output directory and bit message.
The default payload is 0.2 bits per coefficient; override it with --payload.
"""
import csv
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

STEGO = Path(__file__).resolve().parent
# Release encoder carries the cfg-driven StegoEPFPayload / StegoEPFTargetBits options.
ENCODER = STEGO / "TAppEncoder.exe"
BASE_CFG = STEGO / "encoder_lowdelay_P_main.cfg"
DEFAULT_MSG = STEGO / "secret_msg_100000.txt"     # Optional local message file.

INPUT_DIR = STEGO / "input"
OUT_DIR = None          # None => auto: <STEGO>/payload<p>/qp<qp>/<input dir name>

QP = 32
PAYLOAD = 0.2           # <<< FIXED STC EMBEDDING RATE (bpc)
FRAMES = 0              # 0 => auto-detect all available frames of each sequence
MSG_BITS = 100000       # (kept for reference; the shared file is already 100000 bits)
SECRET_SEED = 0

_RE_FRAME = re.compile(
    r"frame embedded=(\d+) flips=(\d+) compFlips=(\d+) \(covers (\d+)/(\d+)/(\d+)\)")
_RE_FINAL = re.compile(
    r"message bits: (\d+) \| embedded: (\d+) \| coefficient flips: (\d+) \| compensation flips: (\d+)")


def parse_seq_name(yuv_path):
    """'PartyScene_832x480_50.yuv' -> ('PartyScene_832x480_50', 832, 480, 50)."""
    stem = yuv_path.stem
    parts = stem.rsplit("_", 2)          # name, WxH, fps
    w, h = (int(x) for x in parts[1].lower().split("x"))
    return stem, w, h, int(parts[2])


def auto_frames(yuv_path, w, h):
    """Total frames implied by the raw YUV420 file size."""
    return yuv_path.stat().st_size // (w * h * 3 // 2)


def set_cfg_value(lines, key, value):
    """Set (or un-comment, or append) a 'key: value' entry in the cfg lines."""
    klen = len(key)
    for i, line in enumerate(lines):
        stripped = line.strip()
        body = stripped[1:].strip() if stripped.startswith("#") else stripped
        if body.startswith(key) and body[klen:].strip().startswith(":"):
            lines[i] = key.ljust(30) + ": " + value + chr(10)
            return
    lines.append(key.ljust(30) + ": " + value + chr(10))


def build_cfg(base_text, values):
    lines = base_text.splitlines(True)
    for k, v in values.items():
        set_cfg_value(lines, k, v)
    return "".join(lines)


def write_message(path, nbits, seed):
    rng = random.Random(seed)
    path.write_text("".join(rng.choice("01") for _ in range(nbits)), encoding="ascii")


def run_one(name, w, h, fps, frames, msg_path):
    outdir = OUT_DIR / name
    outdir.mkdir(parents=True, exist_ok=True)
    yuv = INPUT_DIR / (name + ".yuv")
    if not yuv.exists():
        raise FileNotFoundError(str(yuv))
    bit = outdir / (name + ".bin")
    rec = outdir / (name + "_rec.yuv")
    cfgp = outdir / "run.cfg"

    for p in (bit, rec, cfgp):
        if p.exists():
            p.unlink()

    base = BASE_CFG.read_text(encoding="utf-8", errors="ignore")
    cfg_text = build_cfg(base, {
        "InputFile": str(yuv),
        "BitstreamFile": str(bit),
        "ReconFile": str(rec),
        "SourceWidth": str(w),
        "SourceHeight": str(h),
        "FrameRate": str(fps),
        "FramesToBeEncoded": str(frames),
        "FrameSkip": "0",
        "QP": str(QP),
        "IntraPeriod": "4",
        "GOPSize": "4",
        "UseStegoEPF": "1",
        "StegoEPFMessageFile": str(msg_path),
        "StegoEPFPayload": ("%.6f" % PAYLOAD),      # <<< fixed payload (runtime, cfg-driven)
        "StegoEPFTargetBits": "0",                  # <<< 0 = proportional / fixed-payload mode
    })
    cfgp.write_text(cfg_text, encoding="utf-8", newline="")

    print("[%s] encode %dx%d@%d frames=%d QP=%d payload=%.4f ..."
          % (name, w, h, fps, frames, QP, PAYLOAD), flush=True)
    fd, log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    rc = -1
    stats = None
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            rc = subprocess.run([str(ENCODER), "-c", str(cfgp)],
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        stego_lines = []
        total_covers = 0
        intra_frames = 0
        for line in text.splitlines():
            if "StegoEPF" in line or "StegoDCO" in line:
                print("  " + line.strip(), flush=True)
                stego_lines.append(line.strip())
                m = _RE_FRAME.search(line)
                if m:
                    total_covers += int(m.group(4)) + int(m.group(5)) + int(m.group(6))
                    intra_frames += 1
        # persist the per-sequence embedding summary (message bits / embedded / flips)
        if stego_lines:
            (outdir / "summary.txt").write_text("\n".join(stego_lines) + "\n",
                                                 encoding="utf-8")
        # capacity statistics from the final summary line
        for line in stego_lines:
            m = _RE_FINAL.search(line)
            if m:
                embedded = int(m.group(2))
                stats = {
                    "sequence": name,
                    "frames": frames,
                    "intra_frames": intra_frames,
                    "total_covers": total_covers,
                    "embedded_bits": embedded,
                    "capacity_bpc": (embedded / total_covers) if total_covers else 0.0,
                    "coefficient_flips": int(m.group(3)),
                    "compensation_flips": int(m.group(4)),
                }
                break
    finally:
        for tmp in (log_path,):
            try:
                os.unlink(tmp)
            except OSError:
                pass

    nb = bit.stat().st_size if bit.exists() else 0
    nr = rec.stat().st_size if rec.exists() else 0
    print("[%s] rc=%d  bin=%d bytes  rec=%d bytes" % (name, rc, nb, nr), flush=True)
    ok = rc == 0 and bit.exists() and rec.exists()
    if stats is not None:
        stats["bin_bytes"] = nb
        stats["rec_bytes"] = nr
        stats["ok"] = ok
        return ok, stats
    return ok, None


def write_capacity_csv(rows):
    """Save the capacity report (one row per sequence)."""
    if not rows:
        return
    path = OUT_DIR / "capacity.csv"
    cols = ["sequence", "frames", "intra_frames", "total_covers", "embedded_bits",
            "capacity_bpc", "coefficient_flips", "compensation_flips",
            "bin_bytes", "rec_bytes", "ok"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})
    print()
    print("=" * 90)
    print("CAPACITY REPORT -> %s" % path)
    print("=" * 90)
    print("%-32s %8s %6s %12s %8s %10s" %
          ("sequence", "frames", "intra", "covers", "bits", "bpc"))
    for row in rows:
        print("%-32s %8d %6d %12d %8d %10.4f" %
              (row["sequence"], row["frames"], row["intra_frames"],
               row["total_covers"], row["embedded_bits"], row["capacity_bpc"]))
    print("=" * 90)


def main():
    global PAYLOAD, QP, INPUT_DIR, OUT_DIR
    frames = FRAMES
    only = None
    msg_path = None
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--frames" and i + 1 < len(argv):
            frames = int(argv[i + 1]); i += 2
        elif a == "--only" and i + 1 < len(argv):
            only = argv[i + 1]; i += 2
        elif a == "--msg" and i + 1 < len(argv):
            msg_path = Path(argv[i + 1]); i += 2
        elif a == "--payload" and i + 1 < len(argv):
            PAYLOAD = float(argv[i + 1]); i += 2
        elif a == "--qp" and i + 1 < len(argv):
            QP = int(argv[i + 1]); i += 2
        elif a == "--input" and i + 1 < len(argv):
            INPUT_DIR = Path(argv[i + 1]); i += 2
        elif a == "--out" and i + 1 < len(argv):
            OUT_DIR = Path(argv[i + 1]); i += 2
        else:
            i += 1

    if OUT_DIR is None:
        OUT_DIR = STEGO / ("payload%g" % PAYLOAD) / ("qp%d" % QP) / INPUT_DIR.name

    # loud environment checks
    if not ENCODER.exists():
        sys.exit("ERROR: encoder not found: %s" % ENCODER)
    if not BASE_CFG.exists():
        sys.exit("ERROR: base config not found: %s" % BASE_CFG)
    if not INPUT_DIR.exists():
        sys.exit("ERROR: input dir not found: %s" % INPUT_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yuvs = sorted(INPUT_DIR.glob("*.yuv"))
    if only:
        yuvs = [y for y in yuvs if y.stem == only or y.stem.startswith(only)]
    if not yuvs:
        print("No matching sequence found in %s" % INPUT_DIR, flush=True)
        return

    # Use a local message file by default, or override it with --msg.
    if msg_path is None and not DEFAULT_MSG.exists():
        print("ERROR: shared secret message file not found: %s" % DEFAULT_MSG, flush=True)
        print("Put secret_msg_100000.txt in the shared folder, or pass --msg <path>.", flush=True)
        sys.exit(1)

    rows = []
    try:
        print("=" * 90)
        print("Batch HEVC stego-encode (embed-only, UseStegoEPF) | FIXED PAYLOAD=%.4f | QP%d IPPP | Release"
              % (PAYLOAD, QP))
        print("input : %s" % INPUT_DIR)
        print("output: %s" % OUT_DIR)
        print("=" * 90)
        for yuv in yuvs:
            name, w, h, fps = parse_seq_name(yuv)
            nf = frames if frames > 0 else auto_frames(yuv, w, h)
            src_msg = msg_path if msg_path is not None else DEFAULT_MSG
            print()
            try:
                ok, stats = run_one(name, w, h, fps, nf, src_msg)
                if not ok:
                    print("[%s] WARN: encode may have failed" % name, flush=True)
                if stats is not None:
                    rows.append(stats)
            except Exception as exc:
                print("[%s] FAILED: %s" % (name, exc), flush=True)
        print()
        write_capacity_csv(rows)
        print("ALL DONE", flush=True)
    finally:
        pass


if __name__ == "__main__":
    main()
