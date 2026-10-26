#!/usr/bin/env python3
"""Wang et al. (2015) angle-difference IPM embedding, Tar5Stego TXT adapter.

Compatibility target: Tar5Stego/stc_embed_ipm.py output format, consumed by
Tar5Stego/Cencoder.py via the custom x265 --InputStego option.

NOTE: the original paper's minimum-RD-cost mode search requires RD costs for
ALL eligible modes. The existing Tar5 DDG usually holds only a few alternatives.
This tool labels the nearest-IPM fallback and counts its use; it should not be
presented as an exact RD-optimal reproduction when full RD costs are unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

LINE_RE = re.compile(
    r"CTU Address:\s*(?P<ctu>\d+)\s+x=(?P<x>\d+),y=(?P<y>\d+)\s+"
    r"paritionSize=(?P<part>\d+)\s+depth=(?P<depth>\d+)\s+"
    r"size=(?P<size>\d+x\d+)\s+"
    r"(?:(?:oriIPM=(?P<ori>\d+)\s+stegIPM=(?P<steg>\d+))|IPM=(?P<ipm>\d+))"
)
FRAME_BEGIN = re.compile(r"^Frame\s+(\d+)\s+begin\s*$")
FRAME_END = re.compile(r"^Frame\s+(\d+)\s+end\s*$")


@dataclass
class Block:
    frame: int
    ctu: int
    x: int
    y: int
    part: int
    depth: int
    size: str
    ori: int
    stego: int

    @property
    def is_4x4(self) -> bool:
        # Tar5's encoder emits 'size=8x8' for 4x4 PUs under NxN PU split.
        return self.part == 3 and self.depth == 3 and self.size == "8x8"


def load_ipm(path: Path, *, use_stego: bool = False) -> dict[int, list[Block]]:
    frames = {}
    current = None
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            m = FRAME_BEGIN.fullmatch(line)
            if m:
                current = int(m.group(1))
                if current in frames:
                    raise ValueError(f"Duplicate frame {current} at line {n}")
                frames[current] = []
                continue
            m = FRAME_END.fullmatch(line)
            if m:
                if current != int(m.group(1)):
                    raise ValueError(f"Mismatched Frame end at line {n}")
                current = None
                continue
            m = LINE_RE.search(line)
            if not m:
                raise ValueError(f"Unrecognized nonempty line {n}: {line[:160]}")
            if current is None:
                raise ValueError(f"IPM entry outside Frame begin/end at line {n}")
            original = m.group("ori") or m.group("ipm")
            stego = m.group("steg") or m.group("ipm")
            if use_stego and stego is None:
                raise ValueError(f"Missing stego or decoded IPM at line {n}")
            b = Block(
                frame=current,
                ctu=int(m.group("ctu")),
                x=int(m.group("x")),
                y=int(m.group("y")),
                part=int(m.group("part")),
                depth=int(m.group("depth")),
                size=m.group("size"),
                ori=int(original),
                stego=int(stego),
            )
            if not (0 <= b.ori <= 34 and 0 <= b.stego <= 34):
                raise ValueError(f"Invalid IPM at line {n}")
            frames[current].append(b)
    if current is not None:
        raise ValueError(f"Unterminated Frame {current}")
    if not frames:
        raise ValueError(f"No frames in {path}")
    return frames


def valid_pair(a: int, b: int) -> str | None:
    if 2 <= a <= 34 and 2 <= b <= 34:
        return "directional"
    if a in (0, 1) and b in (0, 1):
        return "planar_dc"
    return None


def angle_bits(a: int, b: int) -> tuple[int, int]:
    """Both directional. Angular difference is |a-b| * pi/32.

    Table 2 uses 2*pi/32 wide bins repeated 00,01,10,11. The
    d=32 == pi endpoint is assigned to last bin (11). Table 1's IPM=28
    duplicated -9*pi/32 is treated as a typo; regular sequence gives -18*pi/32.
    """
    if not (2 <= a <= 34 and 2 <= b <= 34):
        raise ValueError("Direction-only IPMs must be in [2,34]")
    d = min(abs(a - b), 31)
    symbol = (d // 2) % 4
    return (symbol >> 1) & 1, symbol & 1


def pair_symbols(a: int, b: int) -> tuple[int, int] | None:
    typ = valid_pair(a, b)
    if typ == "directional":
        return angle_bits(a, b)
    if typ == "planar_dc":
        return a, b
    return None


def get_pairs(blocks: list[Block]) -> list[tuple[Block, Block, str | None]]:
    eligible = [b for b in blocks if b.is_4x4]
    # Pair sequential 4x4 PUs in their stream/decoder log order, not spatial sorting.
    return [
        (
            eligible[i],
            eligible[i + 1],
            valid_pair(eligible[i].stego, eligible[i + 1].stego),
        )
        for i in range(0, len(eligible) - 1, 2)
    ]


def load_ddg_costs(
    ddg_dir: Path | None,
) -> dict[int, dict[tuple[int, int, int], dict[int, float]]]:
    result = defaultdict(dict)
    if ddg_dir is None:
        return result
    files = list(ddg_dir.rglob("frame_*_ddg.json"))
    if not files:
        raise ValueError(f"No frame_*_ddg.json found under {ddg_dir}")
    for path in files:
        m = re.search(r"frame_(\d+)_ddg\.json$", path.name)
        if not m:
            continue
        frame = int(m.group(1))
        data = json.loads(path.read_text(encoding="utf-8"))
        for node in data.get("nodes", []):
            if node.get("w") != 4 or node.get("h") != 4:
                continue
            key = (int(node.get("ctu_addr", -1)), int(node["x"]), int(node["y"]))
            modes = {}
            for k, v in node.get("candidates", {}).items():
                try:
                    mode, cost = int(k), float(v)
                except (ValueError, TypeError):
                    continue
                if 0 <= mode <= 34 and math.isfinite(cost) and cost >= 0:
                    modes[mode] = cost
            result[frame][key] = modes
    return result


def choose_second_ipm(
    a: Block,
    b: Block,
    desired: tuple[int, int],
    costs: dict[int, float],
    strict_rd: bool,
) -> tuple[int, str]:
    if angle_bits(a.stego, b.stego) == desired:
        return b.stego, "already_matched"
    alternatives = [m for m in range(2, 35) if angle_bits(a.stego, m) == desired]
    recorded = [m for m in alternatives if m in costs]
    if recorded:
        # Only optimal *within recorded candidates* (not necessarily among all 33 modes).
        return min(recorded, key=lambda m: (costs[m], abs(m - b.ori), m)), "recorded_rd"
    if strict_rd:
        raise ValueError(
            f"No matching RD candidate frame={b.frame} ctu={b.ctu} x={b.x} y={b.y}"
        )
    return min(alternatives, key=lambda m: (abs(m - b.ori), m)), "nearest_fallback"


def write_tar5_txt(frames: dict[int, list[Block]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for frame, blocks in frames.items():
            f.write(f"Frame {frame} begin\n")
            for b in blocks:
                f.write(
                    f"CTU Address: {b.ctu} x={b.x},y={b.y} paritionSize={b.part} "
                    f"depth={b.depth} size={b.size} oriIPM={b.ori} stegIPM={b.stego}\n"
                )
            f.write(f"Frame {frame} end\n")


def load_message(path: Path | None, seed: int):
    if path is None:
        rng = random.Random(seed)
        return lambda n: [rng.randint(0, 1) for _ in range(n)]
    data = path.read_bytes()
    bits = [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]
    cursor = [0]

    def take(n):
        if cursor[0] + n > len(bits):
            raise ValueError(
                "Message file does not contain enough bits for target payload"
            )
        x = bits[cursor[0] : cursor[0] + n]
        cursor[0] += n
        return x

    return take


def embed(args):
    frames = load_ipm(Path(args.cover))
    costs = load_ddg_costs(Path(args.ddg_dir) if args.ddg_dir else None)
    next_bits = load_message(
        Path(args.message_file) if args.message_file else None, args.seed
    )
    output_bits = []
    info = {}
    stats = defaultdict(int)
    for frame, blocks in frames.items():
        candidates = [b for b in blocks if b.is_4x4]
        active = args.all_frames or (
            frame >= args.first_i_frame
            and (frame - args.first_i_frame) % args.intra_period == 0
        )
        # Non-I frames preserved unchanged so x265 receives full IPM structure.
        pair_list = get_pairs(blocks)
        valid = [p for p in pair_list if p[2] is not None]
        cap = 2 * len(valid)
        requested = (
            (int(len(candidates) * args.payload_ratio) // 2) * 2 if active else 0
        )
        count = min(requested, cap)
        if active and requested > cap:
            stats["capacity_limited_frames"] += 1
        msg = next_bits(count)
        modified = 0
        stats_frame = defaultdict(int)
        for j, (a, b, kind) in enumerate(valid[: count // 2]):
            desired = (msg[2 * j], msg[2 * j + 1])
            if kind == "planar_dc":
                a.stego, b.stego = desired
                stats_frame["special_pairs"] += 1
            else:
                key = (b.ctu, b.x, b.y)
                b.stego, how = choose_second_ipm(
                    a, b, desired, costs.get(frame, {}).get(key, {}), args.strict_rd
                )
                stats_frame[how] += 1
            modified += int(a.ori != a.stego) + int(b.ori != b.stego)
            if pair_symbols(a.stego, b.stego) != desired:
                raise AssertionError(f"Encoding mismatch on frame {frame} pair {j}")
        output_bits.extend(msg)
        info[str(frame)] = {
            "active": active,
            "four_by_four": len(candidates),
            "pairs": len(pair_list),
            "valid_pairs": len(valid),
            "capacity_bits": cap,
            "requested_bits": requested,
            "embedded_bits": count,
            "modified_ipms": modified,
            "selection": dict(stats_frame),
        }
        stats["total_embedded_bits"] += count
        stats["total_modified_ipms"] += modified
        for k, v in stats_frame.items():
            stats[k] += v
    output = Path(args.output)
    write_tar5_txt(frames, output)
    bits_path = Path(str(output) + ".bits")
    bits_path.write_bytes(bytes(output_bits))
    metadata = {
        "algorithm": "Wang2015-angle-difference",
        "approximation": "partial DDG RDcost, nearest fallback where missing",
        "source": str(args.cover),
        "stego_txt": str(output),
        "seed": args.seed,
        "payload_ratio": args.payload_ratio,
        "intra_period": args.intra_period,
        "first_i_frame": args.first_i_frame,
        "all_frames": args.all_frames,
        "frame_stats": info,
        "totals": dict(stats),
        "bits_file": str(bits_path),
        "bits_sha256": hashlib.sha256(bytes(output_bits)).hexdigest(),
    }
    meta_path = Path(str(output) + ".json")
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Tar5 TXT: {output}")
    print(f"Metadata: {meta_path}; message bits: {bits_path}")
    print("Summary:", json.dumps(dict(stats), ensure_ascii=False))


def extract(args):
    frames = load_ipm(Path(args.input), use_stego=True)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    all_bits = []
    for frame_string, summary in metadata["frame_stats"].items():
        count = summary["embedded_bits"]
        if count == 0:
            continue
        frame = int(frame_string)
        if frame not in frames:
            raise ValueError(f"Decoded output is missing Frame {frame}")
        pairs = [pair for pair in get_pairs(frames[frame]) if pair[2] is not None]
        if 2 * len(pairs) < count:
            raise ValueError(
                f"Frame {frame}: decoded capacity {2*len(pairs)} less than {count}"
            )
        for a, b, kind in pairs[: count // 2]:
            bits = pair_symbols(a.stego, b.stego)
            if bits is None:
                raise AssertionError("Unexpected invalid pair")
            all_bits.extend(bits)
    out = Path(args.output_bits)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(all_bits))
    calculated = hashlib.sha256(bytes(all_bits)).hexdigest()
    expected = metadata["bits_sha256"]
    print(f"Extracted {len(all_bits)} bits -> {out}")
    print(f"SHA256 equals original: {calculated == expected}")
    if calculated != expected:
        orig_path = Path(metadata.get("bits_file", ""))
        if orig_path.is_file():
            reference = orig_path.read_bytes()
            n = min(len(reference), len(all_bits))
            errors = sum(x != y for x, y in zip(reference, all_bits))
            errors += abs(len(reference) - len(all_bits))
            print(
                f"Bit errors: {errors}; compared length: {n}; BER: {errors/max(len(reference), 1):.6f}"
            )
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="command", required=True)
    p = cmds.add_parser("embed", help="Cover IPM TXT -> Wang stego_IPM TXT")
    p.add_argument("--cover", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--payload-ratio",
        type=float,
        default=0.3,
        help="Target secret bits per 4x4 PU in an active frame (default 0.3)",
    )
    p.add_argument("--intra-period", type=int, default=4)
    p.add_argument("--first-i-frame", type=int, default=1)
    p.add_argument("--all-frames", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--message-file",
        help="Optional binary message file; otherwise seeded random bits",
    )
    p.add_argument(
        "--ddg-dir",
        help="Optional existing frame_N_ddg.json folder with candidate RD costs",
    )
    p.add_argument(
        "--strict-rd",
        action="store_true",
        help="Abort rather than nearest fallback if no recorded RD candidate",
    )
    q = cmds.add_parser(
        "extract", help="Extract using decoded or pre-encoding stego IPM TXT"
    )
    q.add_argument("--input", required=True)
    q.add_argument("--metadata", required=True)
    q.add_argument("--output-bits", required=True)
    args = parser.parse_args()
    if args.command == "embed":
        if not 0 <= args.payload_ratio <= 1:
            parser.error("--payload-ratio must be between 0 and 1")
        if args.intra_period < 1:
            parser.error("--intra-period must be >= 1")
        embed(args)
    else:
        extract(args)


if __name__ == "__main__":
    main()
