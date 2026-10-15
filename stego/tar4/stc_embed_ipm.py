import os
import re
import numpy as np
import subprocess
import argparse
from collections import defaultdict
from tqdm import tqdm

# === 参数 ===
parser = argparse.ArgumentParser()
parser.add_argument(
    '--cover_ipm',
    type=str,
    required=True,
    help='A文件，全量cover IPM')
parser.add_argument('--input_txt', type=str, required=True, help='B文件，候选嵌入块')
parser.add_argument('--output_txt', type=str, required=True, help='输出C文件')
parser.add_argument('--payload', type=int, default=None)
parser.add_argument('--embed_ratio', type=float, default=1)
parser.add_argument('--constraint_height', type=int, default=7)
parser.add_argument(
    '--stc_exe',
    type=str,
    default=os.path.join(
        "stc",
        "example1_ml_stc.exe"))
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--T_STC', type=int, default=16)
args = parser.parse_args()

COVER_BIN, MESSAGE_BIN, RHO_BIN, STEGO_BIN = "cover.dat", "message.dat", "rho.dat", "stego.dat"


def write_bin(
    fname,
    array,
    dtype=np.uint8): np.array(
        array,
    dtype=dtype).tofile(fname)


# === Step1: 解析 A文件 (_cover_ipm.txt)，作为输出基底 ===
pattern_cover_line = re.compile(
    r"CTU Address:\s*(\d+)\s+x=(\d+),y=(\d+)\s+paritionSize=(\d+)\s+depth=(\d+)\s+size=([0-9]+x[0-9]+)\s+IPM=(\d+)"
)
cover_blocks = []  # (frame, ctu, x, y, partsize, depth, size, oriIPM)
ctu_4x4_count = defaultdict(int)
current_frame = None
with open(args.cover_ipm, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith("Frame") and "begin" in line:
            current_frame = int(line.split()[1])
            continue
        if "end" in line:
            current_frame = None
            continue
        m = pattern_cover_line.search(line)
        if m and current_frame is not None:
            ctu, x, y, partsize, depth, size, ipm = m.groups()
            frame, ctu, x, y, partsize, depth, ipm = int(current_frame), int(
                ctu), int(x), int(y), int(partsize), int(depth), int(ipm)
            cover_blocks.append((frame, ctu, x, y, partsize, depth, size, ipm))
            if partsize == 3 and depth == 3 and size == "8x8":
                ctu_4x4_count[(frame, ctu)] += 1
qualified_ctus = {k for k, v in ctu_4x4_count.items() if v >= args.T_STC}
print(
    f"总CTU数: {len(ctu_4x4_count)}, 达标CTU数: {len(qualified_ctus)} (T_STC={args.T_STC})")

# === Step2: 解析 B文件 (_ipm_cover_for_stc.txt)，并构建候选 ===
pattern_stc = re.compile(
    r"Frame=(\d+)\s+CTU Address:\s*(\d+)\s+x=(\d+),y=(\d+)\s+paritionSize=(\d+)\s+depth=(\d+)"
    r"\s+size=(\d+x\d+)\s+coverBit=(\d+)\s+D0=([\d\.eE\-\+]+)\s+IPM_0=(\d+)\s+D1=([\d\.eE\-\+]+)\s+IPM_1=(\d+)"
)
frame_groups = defaultdict(lambda: defaultdict(list))
with open(args.input_txt, 'r') as f:
    for line in f:
        m = pattern_stc.search(line.strip())
        if not m:
            continue
        frame, ctu, x, y, partsize, depth, size, coverBit, d0, ipm0, d1, ipm1 = m.groups()
        frame, ctu, x, y, partsize, depth, coverBit, ipm0, ipm1 = map(
            int, (frame, ctu, x, y, partsize, depth, coverBit, ipm0, ipm1))
        d0, d1 = float(d0), float(d1)
        if (frame, ctu) not in qualified_ctus:
            continue
        cost = d1 if coverBit == 0 else d0
        frame_groups[frame][ctu].append({
            'ctu': ctu, 'x': x, 'y': y, 'bit': coverBit, 'cost': cost, 'ipm_pair': (ipm0, ipm1)
        })

# === Step3: STC嵌入 ===
np.random.seed(args.seed)
modified_map = {}  # key:(frame,ctu,x,y) -> stegIPM
total_embedded, modified_count = 0, 0

for frame, ctu_dict in tqdm(
        sorted(frame_groups.items()), desc="Embedding frames"):
    stc_cover, stc_rho, stc_pairs, stc_index_map, start = [], [], [], {}, 0
    for ctu, candidates in sorted(ctu_dict.items()):
        num_candidates = len(candidates)
        if args.payload is not None:
            remaining = args.payload - total_embedded
            if remaining <= 0:
                break
            message_len = min(remaining, num_candidates)
        else:
            message_len = int(num_candidates * args.embed_ratio)
        if message_len == 0:
            continue
        stc_index_map[ctu] = (start, start + message_len, candidates)
        start += message_len
        stc_cover.extend([c['bit'] for c in candidates[:message_len]])
        stc_rho.extend([c['cost'] for c in candidates[:message_len]])
        stc_pairs.extend([c['ipm_pair'] for c in candidates[:message_len]])

    if not stc_cover:
        continue
    message_bits = np.random.randint(0, 2, len(stc_cover)).astype(np.uint8)
    write_bin(COVER_BIN, stc_cover)
    write_bin(MESSAGE_BIN, message_bits)
    write_bin(RHO_BIN, stc_rho)
    subprocess.run([args.stc_exe, COVER_BIN, MESSAGE_BIN, RHO_BIN,
                   STEGO_BIN, str(args.constraint_height)], check=True)
    stego_bits = np.fromfile(STEGO_BIN, dtype=np.uint8)[:len(stc_cover)]

    for ctu, (start_idx, end_idx, candidates) in stc_index_map.items():
        bits_ctu = stego_bits[start_idx:end_idx]
        for b, c in zip(bits_ctu, candidates):
            steg = c['ipm_pair'][0] if b == 0 else c['ipm_pair'][1]
            ori = c['ipm_pair'][c['bit']]
            if steg != ori:
                modified_count += 1
            modified_map[(frame, c['ctu'], c['x'], c['y'])] = steg
            total_embedded += 1


# === Step4: 生成C文件（只更新A中的4x4块） ===
# === Step4: 生成C文件（保留A结构，只更新4×4块IPM） ===
with open(args.output_txt, 'w') as f:
    current_frame = None
    for block in cover_blocks:
        frame, ctu, x, y, partsize, depth, size, ori = block
        if frame != current_frame:
            if current_frame is not None:
                f.write(f"Frame {current_frame} end\n")
            current_frame = frame
            f.write(f"Frame {frame} begin\n")
        steg = ori
        # 只更新4×4块
        if partsize == 3 and depth == 3 and size == "8x8":
            steg = modified_map.get((frame, ctu, x, y), ori)
        # 大块直接保留，不改块类型和IPM
        f.write(
            f"CTU Address: {ctu} x={x},y={y} paritionSize={partsize} depth={depth} size={size} oriIPM={ori} stegIPM={steg}\n")
    if current_frame is not None:
        f.write(f"Frame {current_frame} end\n")


print(
    f"\n✅ 嵌入完成: {total_embedded} bits, 修改IPM: {modified_count}, 输出文件: {args.output_txt}")
