import os
import re
import json
import math
import subprocess
import numpy as np
from collections import defaultdict

# ================================================================
# 参数配置
# ================================================================
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cover_ipm", type=str, required=True)
parser.add_argument(
    "--input_txt",
    type=str,
    required=True,
    help="输入候选 IPM 文件路径")
parser.add_argument(
    "--output_txt",
    type=str,
    required=True,
    help="输出 stego IPM 文件路径")
parser.add_argument("--stc_exe", type=str, default="./stc/example1_ml_stc.exe")
parser.add_argument("--payload_ratio", type=float, default=0.3)
parser.add_argument("--joint_ratio", type=float, default=0.15)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--height", type=int, default=7)
args = parser.parse_args()

input_txt = args.input_txt
output_txt = args.output_txt
stc_exe = args.stc_exe
payload_ratio = args.payload_ratio
joint_ratio = args.joint_ratio
alpha = args.alpha
height = args.height
tmp_dir = "./tmp_stc_ipm"
os.makedirs(tmp_dir, exist_ok=True)

# ================================================================
# Step1. 解析 cand±1 / ori / RDcost 信息
# ================================================================
pattern = re.compile(
    r"Frame=(\d+)\s+CTU Address:\s*(\d+)\s+x=(\d+),y=(\d+)\s+paritionSize=(\d+)\s+depth=(\d+)\s+size=([0-9]+x[0-9]+)\s+"
    r"cand1IPM=(-?\d+)\s+cand1RDcost=([-]?\d+\.\d+),oriIPM=(\d+)\s+oriRDcost=([\d\.]+),cand2IPM=(-?\d+)\s+cand2RDcost=([\d\.]+)"
)

records = []
with open(input_txt, "r", encoding="utf-8") as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue
        frame, ctu, x, y, part, depth, size, cand1, rd1, ori, rd_ori, cand2, rd2 = m.groups()
        part, depth = int(part), int(depth)
        frame, ctu, x, y = int(frame), int(ctu), int(x), int(y)
        cand1, cand2, ori = int(cand1), int(cand2), int(ori)
        rd1, rd2, rd_ori = float(rd1), float(rd2), float(rd_ori)

        # ==== 边界合法性 ====
        if cand1 < 0 or rd1 < 0:
            cand1, rd1 = None, None
        if cand2 > 34 or rd2 < 0:
            cand2, rd2 = None, None

        # ==== θ 判断 ====
        if rd1 is None and rd2 is not None:
            theta = 1
        elif rd2 is None and rd1 is not None:
            theta = -1
        elif rd1 is None and rd2 is None:
            theta = 0
        else:
            theta = -1 if rd1 < rd2 else (1 if rd1 > rd2 else 0)

        # ==== 代价修正 ====
        rho_minus = rd1 if rd1 is not None else np.inf
        rho_ori = rd_ori
        rho_plus = rd2 if rd2 is not None else np.inf
        if theta < 0:
            if np.isfinite(rho_minus):
                rho_minus *= (1 - alpha)
            if np.isfinite(rho_plus):
                rho_plus *= (1 + alpha)
        elif theta > 0:
            if np.isfinite(rho_minus):
                rho_minus *= (1 + alpha)
            if np.isfinite(rho_plus):
                rho_plus *= (1 - alpha)
        rho_eff = min(rho_minus, rho_plus)

        records.append({
            "frame": frame, "ctu": ctu, "x": x, "y": y,
            "part": part, "depth": depth, "size": size,
            "ori": ori, "cand1": cand1, "cand2": cand2,
            "rho": rho_eff, "theta": theta,
            "rd_ori": rd_ori, "rd1": rd1, "rd2": rd2,
            "ipm_minus": cand1, "ipm_plus": cand2

        })

print(f"✅ 加载 {len(records)} 个 PU")

# ================================================================
# Step2. 划分 S/J 集合
# ================================================================
plan_S = []
plan_J = defaultdict(list)
pair_id = 0
for frame_id, frame_recs in defaultdict(list, {}).items():
    pass  # 兼容旧接口

# 简单成对配对（相邻 PU）
sorted_recs = sorted(records, key=lambda r: (r["frame"], r["y"], r["x"]))
i = 0
while i < len(sorted_recs) - 1:
    r1, r2 = sorted_recs[i], sorted_recs[i + 1]
    # 仅在同一帧内配对，且方向相关（θ 同号）
    if r1["frame"] == r2["frame"] and r1["theta"] * r2["theta"] > 0:
        plan_J[pair_id].append(r1)
        plan_J[pair_id].append(r2)
        pair_id += 1
        i += 2
    else:
        plan_S.append(r1)
        i += 1
if i == len(sorted_recs) - 1:
    plan_S.append(sorted_recs[-1])

num_S = len(plan_S)
num_J = sum(len(v) for v in plan_J.values())
print(f"✅ 集合划分完成: S={num_S}, J_pairs={len(plan_J)} (共 {num_J} PU)")

# ================================================================
# Step3. 计算 τ(l,r) 和联合代价
# ================================================================
joint_list = []
for pid, pair in plan_J.items():
    if len(pair) != 2:
        continue
    p1, p2 = pair
    tau = math.log(p1["rd_ori"] + 1e-6) * math.log(p2["rd_ori"] + 1e-6)
    rho_star = tau * (p1["rho"] + p2["rho"])
    joint_list.append({
        "pair_id": pid,
        "frame": p1["frame"],
        "p1": p1,
        "p2": p2,
        "rho_star": rho_star,
        "tau": tau
    })

print(f"✅ τ(l,r) 联合代价计算完成，共 {len(joint_list)} 对")

# ================================================================
# Step4. 第一层 STC (J 集合)
# ================================================================


def run_stc(embed_items, rho_key, label="S"):
    """运行 STC 嵌入并返回修改映射"""
    if len(embed_items) == 0:
        return {}
    num_embed = len(embed_items)
    cover_alt = np.zeros(num_embed * 2, dtype=np.uint8)
    rho_vals = np.zeros(num_embed, np.float32)
    msg_bits = np.random.randint(0, 2, size=num_embed).astype(np.uint8)

    for i, r in enumerate(embed_items):
        ori = int(r["ori"])
        if r["theta"] >= 0 and r["ipm_plus"] is not None:
            alt = int(r["ipm_plus"])
        elif r["theta"] <= 0 and r["ipm_minus"] is not None:
            alt = int(r["ipm_minus"])
        elif r["ipm_plus"] is not None:
            alt = int(r["ipm_plus"])
        elif r["ipm_minus"] is not None:
            alt = int(r["ipm_minus"])
        else:
            alt = ori
        cover_alt[2 * i] = ori
        cover_alt[2 * i + 1] = alt
        rho_vals[i] = r[rho_key]

    tmp_cov = os.path.join(tmp_dir, f"cover_{label}.dat")
    tmp_rho = os.path.join(tmp_dir, f"rho_{label}.dat")
    tmp_msg = os.path.join(tmp_dir, f"msg_{label}.dat")
    tmp_stg = os.path.join(tmp_dir, f"stego_{label}.dat")

    cover_alt.tofile(tmp_cov)
    rho_vals.tofile(tmp_rho)
    msg_bits.tofile(tmp_msg)

    cmd = [stc_exe, tmp_cov, tmp_msg, tmp_rho, tmp_stg, str(height)]
    subprocess.run(cmd, check=True)
    stego_bits = np.fromfile(tmp_stg, dtype=np.uint8)[:num_embed]

    change_map = {}
    changed = 0
    for bit, r in zip(stego_bits, embed_items):
        steg = r["ori"]
        if bit == 1:
            if r["theta"] >= 0 and r["ipm_plus"] is not None:
                steg = int(r["ipm_plus"])
            elif r["theta"] <= 0 and r["ipm_minus"] is not None:
                steg = int(r["ipm_minus"])
            elif r["ipm_plus"] is not None:
                steg = int(r["ipm_plus"])
            elif r["ipm_minus"] is not None:
                steg = int(r["ipm_minus"])
            changed += 1
        change_map[(r["frame"], r["ctu"], r["x"], r["y"])] = steg

    print(
        f"✅ STC 层 {label}: 嵌入 {num_embed} PU，修改 {changed} ({changed/num_embed:.2%})")
    return change_map


# === 第一层：J 集合 ===
num_J_embed = int(len(joint_list) * joint_ratio)
embed_J = sorted(joint_list, key=lambda z: z["rho_star"])[:num_J_embed]
flat_J = [p["p1"] for p in embed_J] + [p["p2"] for p in embed_J]
change_J = run_stc(flat_J, rho_key="rho", label="J")

# === 第二层：S 集合 ===
remaining_payload = payload_ratio - joint_ratio
num_S_embed = int(len(plan_S) * remaining_payload)
embed_S = sorted(plan_S, key=lambda z: z["rho"])[:num_S_embed]
change_S = run_stc(embed_S, rho_key="rho", label="S")

# ================================================================
# Step5. 按 cover_ipm.txt 结构完整写回
# ================================================================
cover_file = args.cover_ipm
if not os.path.exists(cover_file):
    raise FileNotFoundError(f"❌ 未找到 cover_ipm 文件: {cover_file}")

# === 合并 S / J 集合嵌入结果 ===
change_map = {**change_J, **change_S}
modified = sum(
    1 for r in records
    if (r["frame"], r["ctu"], r["x"], r["y"]) in change_map
    and change_map[(r["frame"], r["ctu"], r["x"], r["y"])] != r["ori"]
)
print(f"\n✅ 总修改 PU: {modified}/{len(records)} ({modified/len(records):.2%})")
print(f"📄 正在生成完整结构 stego 文件（对齐 cover_ipm）...")

# === 正则匹配 cover 文件行 ===
pattern_cover = re.compile(
    r"CTU Address:\s*(\d+)\s+x=(\d+),y=(\d+)\s+paritionSize=(\d+)\s+depth=(\d+)\s+size=([0-9]+x[0-9]+)\s+(?:IPM|oriIPM)=(\d+)"
)

modified_count = 0
with open(cover_file, "r", encoding="utf-8") as fin, open(output_txt, "w", encoding="utf-8") as fout:
    current_frame = None
    for line in fin:
        line_stripped = line.strip()

        # === 帧头帧尾 ===
        if line_stripped.startswith("Frame") and "begin" in line_stripped:
            current_frame = int(line_stripped.split()[1])
            fout.write(line)
            continue
        if "end" in line_stripped:
            fout.write(line)
            current_frame = None
            continue

        # === 匹配 IPM 行 ===
        m = pattern_cover.search(line_stripped)
        if not m or current_frame is None:
            fout.write(line)
            continue

        ctu, x, y, part, depth, size, ori = m.groups()
        frame, ctu, x, y, ori = int(current_frame), int(
            ctu), int(x), int(y), int(ori)

        # === 替换为 stego IPM（如果有修改） ===
        key = (frame, ctu, x, y)
        steg = change_map.get(key, ori)
        if steg != ori:
            modified_count += 1

        fout.write(
            f"CTU Address: {ctu} x={x},y={y} paritionSize={part} depth={depth} size={size} oriIPM={ori} stegIPM={steg}\n"
        )

print(f"✅ 写入完成，共修改 {modified_count} 个 PU")
print(f"📝 输出文件: {output_txt}")
