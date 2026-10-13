import os, re, pandas as pd
from pathlib import Path


# 解析 cover 文件（全量 IPM）
def parse_cover_file(path):
    pattern = re.compile(
        r"CTU Address:\s*(\d+)\s+x=(\d+),y=(\d+)\s+paritionSize=(\d+)\s+depth=(\d+)\s+size=([0-9]+x[0-9]+)\s+IPM=(\d+)"
    )
    blocks = []
    frame = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Frame") and "begin" in line:
                frame = int(line.split()[1])
                continue
            if "end" in line:
                frame = None
                continue
            m = pattern.search(line)
            if m and frame is not None:
                ctu, x, y, partsize, depth, size, ipm = m.groups()
                blocks.append(
                    (
                        frame,
                        int(ctu),
                        int(x),
                        int(y),
                        int(partsize),
                        int(depth),
                        size,
                        int(ipm),
                    )
                )
    return blocks


# 解析候选 B 文件
def parse_ipm_file(path):
    records = []
    for line in open(path, "r"):
        line = line.strip()
        if not line:
            continue
        kv_pairs = re.findall(r"([A-Za-z0-9_]+)\s*[=:]\s*([\w\[\],x\.]+)", line)
        record = {}
        for k, v in kv_pairs:
            if k == "Address":
                k = "CTU"
            elif k == "size":
                wh = v.split("x")
                record["width"], record["height"] = int(wh[0]), int(wh[1])
                continue
            try:
                record[k] = int(v)
            except ValueError:
                try:
                    record[k] = float(v)
                except ValueError:
                    record[k] = v
        records.append(record)
    return pd.DataFrame(records)


# 生成嵌入计划（Matrix Coding）
def generate_embedding_plan(df, payload=1.0, embed_bits=0b11):
    if len(df) == 0:
        return []
    # 8x8 左上角
    df["cu_x"] = df["x"] // 8 * 8
    df["cu_y"] = df["y"] // 8 * 8
    # uniform过滤
    group_keys = ["Frame", "CTU", "cu_x", "cu_y", "depth", "paritionSize"]
    filtered = [
        g
        for _, g in df.groupby(group_keys)
        if len(g) == 4 and len(set(g["coverIPM"])) != 1
    ]
    if not filtered:
        return []
    df_filtered = pd.concat(filtered).reset_index(drop=True)
    df_filtered["min_cost"] = df_filtered[["D0", "D1"]].min(axis=1)
    # 每帧选择 payload 比例
    selected = []
    for frame, g in df_filtered.groupby("Frame"):
        n_keep = int(len(g) * payload)
        selected.append(g.sort_values("min_cost").iloc[:n_keep])
    df_selected = pd.concat(selected).reset_index(drop=True)
    # 3个PU一组 + Matrix coding
    groups = [df_selected.iloc[i : i + 3] for i in range(0, len(df_selected), 3)]
    embedding = []
    for group in groups:
        if len(group) < 3:
            continue
        cover_bits = list(group["coverBit"])
        w = embed_bits
        idx = w ^ (cover_bits[0] * 1 ^ cover_bits[1] * 2 ^ cover_bits[2] * 3)
        for j, (_, row) in enumerate(group.iterrows(), start=1):
            stego_ipm = row["coverIPM"]
            if idx == j:
                target_bit = 1 - int(row["coverBit"])
                stego_ipm = row["IPM_0"] if target_bit == 0 else row["IPM_1"]
            embedding.append(
                {
                    "frame": int(row["Frame"]),
                    "ctu": int(row["CTU"]),
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "stegIPM": int(stego_ipm),
                }
            )
    return embedding


# 生成最终 stego 文件
def write_stego_file(cover_blocks, embedding_plan, output_path):
    modified_map = {
        (e["frame"], e["ctu"], e["x"], e["y"]): e["stegIPM"] for e in embedding_plan
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        current_frame = None
        for frame, ctu, x, y, partsize, depth, size, ori in cover_blocks:
            if frame != current_frame:
                if current_frame is not None:
                    f.write(f"Frame {current_frame} end\n")
                current_frame = frame
                f.write(f"Frame {frame} begin\n")
            steg = modified_map.get((frame, ctu, x, y), ori)
            f.write(
                f"CTU Address: {ctu} x={x},y={y} paritionSize={partsize} depth={depth} size={size} oriIPM={ori} stegIPM={steg}\n"
            )
        if current_frame is not None:
            f.write(f"Frame {current_frame} end\n")


# === 批处理入口 ===
if __name__ == "__main__":
    cover_root = r"cover_dir"
    cand_root = r"results_cand_dir"
    stego_root = r"stego_out_dir"
    payload = 0.1
    for cand_path in Path(cand_root).rglob("*_ipm_cover_for_filter.txt"):
        rel = cand_path.relative_to(cand_root)
        cover_path = (
            Path(cover_root)
            / rel.parent
            / (cand_path.stem.replace("_ipm_cover_for_filter", "_cover_ipm") + ".txt")
        )
        out_path = (
            Path(stego_root)
            / rel.parent
            / (cand_path.stem.replace("_ipm_cover_for_filter", "_stego_IPM") + ".txt")
        )
        if not cover_path.exists():
            print(f"未找到对应cover: {cover_path}, 跳过")
            continue
        cover_blocks = parse_cover_file(str(cover_path))
        df = parse_ipm_file(str(cand_path))
        embedding_plan = generate_embedding_plan(df, payload=payload, embed_bits=0b11)
        write_stego_file(cover_blocks, embedding_plan, str(out_path))
        print(f"生成完成: {out_path} ({len(embedding_plan)} 个嵌入块)")
