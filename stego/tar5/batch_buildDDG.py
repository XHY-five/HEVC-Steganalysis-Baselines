# import os
# import re
# import json
# import glob
# from pathlib import Path
# from collections import defaultdict
# from tqdm import tqdm
#
# def parse_ipm_file_with_rdcost(ipm_path, rdcost_path):
#     rdcost_map = defaultdict(lambda: defaultdict(dict))
#
#     with open(rdcost_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             frame_m = re.search(r"Frame=(\d+)", line)
#             pos_m = re.search(r"CTU Address: (\d+).*?x=(\d+),y=(\d+).*?paritionSize=(\d+).*?depth=(\d+).*?size=(\d+)x(\d+)", line)
#             if not (frame_m and pos_m):
#                 continue
#
#             fid = int(frame_m.group(1))
#             ctu = int(pos_m.group(1))
#             x, y = int(pos_m.group(2)), int(pos_m.group(3))
#             part, depth = int(pos_m.group(4)), int(pos_m.group(5))
#             w, h = int(pos_m.group(6)), int(pos_m.group(7))
#             if part == 3 and depth == 3:
#                 w = h = 4
#
#             entries = re.findall(r"(cand1|cand2|ori)IPM=([-\d]+)\s+(?:cand1|cand2|ori)RDcost=([-\d.]+)", line)
#             for tag, ipm_str, cost_str in entries:
#                 ipm = int(ipm_str)
#                 cost = float(cost_str)
#                 if ipm < 0 or ipm > 34 or cost < 0:
#                     continue
#                 rdcost_map[fid][(x, y, w, h)][(ipm, ctu)] = cost
#                 # 增加字段提取（方向性特征）
#                 oz_m = re.search(r"oz=([\d.]+)", line)
#                 or_m = re.search(r"or=([\d.]+)", line)
#                 sum_e2_m = re.search(r"sum_e2=([\d.]+)", line)
#                 rho_dz_m = re.search(r"rho_dz=([\d.]+)", line)
#                 rho_dr_m = re.search(r"rho_dr=([\d.]+)", line)
#                 rho_d_m = re.search(r"rho_d=([\d.]+)", line)
#
#                 oz = float(oz_m.group(1)) if oz_m else 0.0
#                 or_ratio = float(or_m.group(1)) if or_m else 0.0
#                 sum_e2 = float(sum_e2_m.group(1)) if sum_e2_m else 0.0
#                 rho_dz = float(rho_dz_m.group(1)) if rho_dz_m else 0.0
#                 rho_dr = float(rho_dr_m.group(1)) if rho_dr_m else 0.0
#                 rho_d = float(rho_d_m.group(1)) if rho_d_m else 0.0
#     frames = []
#     current_frame = []
#     frame_id = 0
#
#     with open(ipm_path, 'r') as f:
#         for line in f:
#             if "Frame" in line and "begin" in line:
#                 current_frame = []
#             elif "Frame" in line and "end" in line:
#                 if current_frame:
#                     frames.append(current_frame)
#                 frame_id += 1
#             elif "paritionSize=8" in line:
#                 continue
#             else:
#                 m = re.search(r"CTU Address: (\d+).*?x=(\d+),y=(\d+).*?paritionSize=(\d+).*?depth=(\d+).*?size=(\d+)x(\d+).*?IPM=(\d+)", line)
#
#                 if m:
#                     ctu, x, y, part, depth, w, h, ipm = map(int, m.groups())
#                     if part == 3 and depth == 3:
#                         w = h = 4
#                     key = (x, y, w, h)
#                     cand_dict = rdcost_map[frame_id+1].get(key, {})
#
#                     match = [(k[0], v) for k, v in cand_dict.items() if k[1] == ctu]
#                     distortion = 0.0
#                     candidates = {}
#                     for ipm_cand, cost in match:
#                         candidates[str(ipm_cand)] = cost
#                         if ipm_cand == ipm:
#                             distortion = cost
#
#                     current_frame.append({
#                         "x": x, "y": y, "w": w, "h": h,
#                         "ipm": ipm,
#                         "distortion": distortion,
#                         "candidates": candidates,
#                         "ctu_addr": ctu,
#                         "oz": oz,
#                         "or": or_ratio,
#                         "sum_e2": sum_e2,
#                         "rho_dz": rho_dz,
#                         "rho_dr": rho_dr,
#                         "rho_d": rho_d
#                     })
#     return frames
#
# def build_ref_zone(x, y, w, h):
#     zone = set()
#     zone.update((x - 1, y + dy) for dy in range(h))
#     zone.update((x + dx, y - 1) for dx in range(w))
#     zone.add((x - 1, y - 1))
#     return zone
#
# def build_ddg_for_frame_fast(pus, grid_size=32):
#     from collections import defaultdict
#
#     for idx, pu in enumerate(pus):
#         pu["id"] = idx
#         pu["zone"] = set((pu["x"] + dx, pu["y"] + dy) for dx in range(pu["w"]) for dy in range(pu["h"]))
#         pu["ref_zone"] = build_ref_zone(pu["x"], pu["y"], pu["w"], pu["h"])
#         pu["bbox"] = (pu["x"], pu["y"], pu["x"] + pu["w"] - 1, pu["y"] + pu["h"] - 1)
#
#     grid = defaultdict(list)
#     def get_cells(x0, y0, x1, y1):
#         gx0, gy0 = x0 // grid_size, y0 // grid_size
#         gx1, gy1 = x1 // grid_size, y1 // grid_size
#         return [(gx, gy) for gx in range(gx0, gx1+1) for gy in range(gy0, gy1+1)]
#
#     for pu in pus:
#         x0, y0, x1, y1 = pu["bbox"]
#         for cell in get_cells(x0, y0, x1, y1):
#             grid[cell].append(pu["id"])
#
#     edges = []
#     for pu_b in pus:
#         ref_zone = pu_b["ref_zone"]
#         if not ref_zone:
#             continue
#         rx = min(p[0] for p in ref_zone)
#         ry = min(p[1] for p in ref_zone)
#         rw = max(p[0] for p in ref_zone) - rx + 1
#         rh = max(p[1] for p in ref_zone) - ry + 1
#         ref_cells = get_cells(rx, ry, rx+rw-1, ry+rh-1)
#
#         checked_ids = set()
#         for cell in ref_cells:
#             for aid in grid.get(cell, []):
#                 if aid == pu_b["id"] or aid in checked_ids:
#                     continue
#                 checked_ids.add(aid)
#                 pu_a = pus[aid]
#                 ax0, ay0, ax1, ay1 = pu_a["bbox"]
#                 if rx > ax1 or rx + rw - 1 < ax0 or ry > ay1 or ry + rh - 1 < ay0:
#                     continue
#                 if pu_a["zone"] & ref_zone:
#                     edges.append((aid, pu_b["id"]))
#
#     nodes = [{
#         "id": pu["id"], "x": pu["x"], "y": pu["y"],
#         "w": pu["w"], "h": pu["h"], "ipm": pu["ipm"],
#         "distortion": pu.get("distortion", 0.0),
#         "candidates": pu.get("candidates", {}),
#         "ctu_addr": pu.get("ctu_addr", -1),
#         "oz": pu.get("oz", 0.0),
#         "or": pu.get("or", 0.0),
#         "sum_e2": pu.get("sum_e2", 0.0),
#         "rho_dz": pu.get("rho_dz", 0.0),
#         "rho_dr": pu.get("rho_dr", 0.0),
#         "rho_d": pu.get("rho_d", 0.0)
#     } for pu in pus]
#
#     return {"nodes": nodes, "edges": edges}
#
# def build_ddg_all_frames(ipm_file, rdcost_file, output_dir, skip_existing=True):
#     frames = parse_ipm_file_with_rdcost(ipm_file, rdcost_file)
#     os.makedirs(output_dir, exist_ok=True)
#
#     for i, pus in enumerate(frames):
#         out_path = os.path.join(output_dir, f"frame_{i+1}_ddg.json")
#         if skip_existing and os.path.exists(out_path):
#             continue
#         ddg = build_ddg_for_frame_fast(pus)
#         with open(out_path, "w") as f:
#             json.dump(ddg, f, indent=2)
#
# def batch_build_ddg(ipm_root, rdcost_root, output_root):
#     txt_files = glob.glob(f"{rdcost_root}/**/*_ipm_cover_rdcost.txt", recursive=True)
#
#     for rdcost_path in tqdm(txt_files, desc="Building DDGs"):
#         try:
#             seg_name = Path(rdcost_path).stem.replace("_ipm_cover_rdcost", "")
#             rel_dir = os.path.relpath(os.path.dirname(rdcost_path), rdcost_root)
#             ipm_path = os.path.join(ipm_root, rel_dir, f"{seg_name}_cover_ipm.txt")
#             out_dir = os.path.join(output_root, rel_dir, seg_name)
#             os.makedirs(out_dir, exist_ok=True)
#
#             if not os.path.exists(ipm_path):
#                 print(f"[WARN] IPM 文件未找到：{ipm_path}")
#                 continue
#
#             print(f"[INFO] Processing: {seg_name}")
#             build_ddg_all_frames(ipm_path, rdcost_path, out_dir)
#
#         except Exception as e:
#             print(f"[ERROR] 处理失败：{rdcost_path}，原因：{e}")
#
# if __name__ == "__main__":
#     ipm_root = r"D:\Users\13321\PycharmProjects\Tar4Project\dec_ori_0.3_32"
#     rdcost_root = r"D:\Users\13321\PycharmProjects\Tar5Stego\results_0.3_32_rdcost"
#     output_root = "utils/0.3_32_ddg"
#     batch_build_ddg(ipm_root, rdcost_root, output_root)
#     print("✅ Batch DDG build completed.")


import os
import re
import json
import glob
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def parse_ipm_file_with_rdcost(ipm_path, rdcost_path):
    rdcost_map = defaultdict(lambda: defaultdict(dict))

    with open(rdcost_path, 'r') as f:
        for line in f:
            line = line.strip()
            frame_m = re.search(r"Frame=(\d+)", line)
            pos_m = re.search(
                r"CTU Address: (\d+).*?x=(\d+),y=(\d+).*?paritionSize=(\d+).*?depth=(\d+).*?size=(\d+)x(\d+)",
                line)
            if not (frame_m and pos_m):
                continue

            fid = int(frame_m.group(1))
            ctu = int(pos_m.group(1))
            x, y = int(pos_m.group(2)), int(pos_m.group(3))
            part, depth = int(pos_m.group(4)), int(pos_m.group(5))
            w, h = int(pos_m.group(6)), int(pos_m.group(7))
            if part == 3 and depth == 3:
                w = h = 4

            entries = re.findall(
                r"(cand1|cand2|ori)IPM=([-\d]+)\s+(?:cand1|cand2|ori)RDcost=([-\d.]+)", line)
            for tag, ipm_str, cost_str in entries:
                ipm = int(ipm_str)
                cost = float(cost_str)
                if ipm < 0 or ipm > 34 or cost < 0:
                    continue
                rdcost_map[fid][(x, y, w, h)][(ipm, ctu)] = cost
                # 增加字段提取（方向性特征）
                oz_m = re.search(r"oz=([\d.]+)", line)
                or_m = re.search(r"or=([\d.]+)", line)
                sum_e2_m = re.search(r"sum_e2=([\d.]+)", line)
                rho_dz_m = re.search(r"rho_dz=([\d.]+)", line)
                rho_dr_m = re.search(r"rho_dr=([\d.]+)", line)
                rho_d_m = re.search(r"rho_d=([\d.]+)", line)

                oz = float(oz_m.group(1)) if oz_m else 0.0
                or_ratio = float(or_m.group(1)) if or_m else 0.0
                sum_e2 = float(sum_e2_m.group(1)) if sum_e2_m else 0.0
                rho_dz = float(rho_dz_m.group(1)) if rho_dz_m else 0.0
                rho_dr = float(rho_dr_m.group(1)) if rho_dr_m else 0.0
                rho_d = float(rho_d_m.group(1)) if rho_d_m else 0.0
            mpm_m = re.search(r"MPM=\[([\d,]+)\]", line)
            mpm = list(map(int, mpm_m.group(1).split(','))) if mpm_m else []
    frames = []
    current_frame = []
    frame_id = 0

    with open(ipm_path, 'r') as f:
        for line in f:
            if "Frame" in line and "begin" in line:
                current_frame = []
            elif "Frame" in line and "end" in line:
                if current_frame:
                    frames.append(current_frame)
                frame_id += 1
            elif "paritionSize=8" in line:
                continue
            else:
                m = re.search(
                    r"CTU Address: (\d+).*?x=(\d+),y=(\d+).*?paritionSize=(\d+).*?depth=(\d+).*?size=(\d+)x(\d+).*?IPM=(\d+)",
                    line)

                if m:
                    ctu, x, y, part, depth, w, h, ipm = map(int, m.groups())
                    if part == 3 and depth == 3:
                        w = h = 4
                    key = (x, y, w, h)
                    cand_dict = rdcost_map[frame_id + 1].get(key, {})

                    match = [(k[0], v)
                             for k, v in cand_dict.items() if k[1] == ctu]
                    distortion = 0.0
                    candidates = {}
                    for ipm_cand, cost in match:
                        candidates[str(ipm_cand)] = cost
                        if ipm_cand == ipm:
                            distortion = cost

                    current_frame.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "ipm": ipm,
                        "distortion": distortion,
                        "candidates": candidates,
                        "ctu_addr": ctu,
                        "oz": oz,
                        "or": or_ratio,
                        "sum_e2": sum_e2,
                        "rho_dz": rho_dz,
                        "rho_dr": rho_dr,
                        "rho_d": rho_d,
                        "mpm": mpm  # <=== 新增
                    })
    return frames


def build_ref_zone(x, y, w, h):
    zone = set()
    zone.update((x - 1, y + dy) for dy in range(h))
    zone.update((x + dx, y - 1) for dx in range(w))
    zone.add((x - 1, y - 1))
    return zone


def build_ddg_for_frame_fast(pus, grid_size=32):
    from collections import defaultdict

    for idx, pu in enumerate(pus):
        pu["id"] = idx
        pu["zone"] = set((pu["x"] + dx, pu["y"] + dy)
                         for dx in range(pu["w"]) for dy in range(pu["h"]))
        pu["ref_zone"] = build_ref_zone(pu["x"], pu["y"], pu["w"], pu["h"])
        pu["bbox"] = (
            pu["x"],
            pu["y"],
            pu["x"] +
            pu["w"] -
            1,
            pu["y"] +
            pu["h"] -
            1)

    grid = defaultdict(list)

    def get_cells(x0, y0, x1, y1):
        gx0, gy0 = x0 // grid_size, y0 // grid_size
        gx1, gy1 = x1 // grid_size, y1 // grid_size
        return [(gx, gy) for gx in range(gx0, gx1 + 1)
                for gy in range(gy0, gy1 + 1)]

    for pu in pus:
        x0, y0, x1, y1 = pu["bbox"]
        for cell in get_cells(x0, y0, x1, y1):
            grid[cell].append(pu["id"])

    edges = []
    for pu_b in pus:
        ref_zone = pu_b["ref_zone"]
        if not ref_zone:
            continue
        rx = min(p[0] for p in ref_zone)
        ry = min(p[1] for p in ref_zone)
        rw = max(p[0] for p in ref_zone) - rx + 1
        rh = max(p[1] for p in ref_zone) - ry + 1
        ref_cells = get_cells(rx, ry, rx + rw - 1, ry + rh - 1)

        checked_ids = set()
        for cell in ref_cells:
            for aid in grid.get(cell, []):
                if aid == pu_b["id"] or aid in checked_ids:
                    continue
                checked_ids.add(aid)
                pu_a = pus[aid]
                ax0, ay0, ax1, ay1 = pu_a["bbox"]
                if rx > ax1 or rx + rw - 1 < ax0 or ry > ay1 or ry + rh - 1 < ay0:
                    continue
                if pu_a["zone"] & ref_zone:
                    edges.append((aid, pu_b["id"]))

    nodes = [{
        "id": pu["id"], "x": pu["x"], "y": pu["y"],
        "w": pu["w"], "h": pu["h"], "ipm": pu["ipm"],
        "distortion": pu.get("distortion", 0.0),
        "candidates": pu.get("candidates", {}),
        "ctu_addr": pu.get("ctu_addr", -1),
        "oz": pu.get("oz", 0.0),
        "or": pu.get("or", 0.0),
        "sum_e2": pu.get("sum_e2", 0.0),
        "rho_dz": pu.get("rho_dz", 0.0),
        "rho_dr": pu.get("rho_dr", 0.0),
        "rho_d": pu.get("rho_d", 0.0),
        "mpm": pu.get("mpm", [])  # <=== 新增
    } for pu in pus]

    return {"nodes": nodes, "edges": edges}


def build_ddg_all_frames(ipm_file, rdcost_file,
                         output_dir, skip_existing=True):
    frames = parse_ipm_file_with_rdcost(ipm_file, rdcost_file)
    os.makedirs(output_dir, exist_ok=True)

    for i, pus in enumerate(frames):
        out_path = os.path.join(output_dir, f"frame_{i+1}_ddg.json")
        if skip_existing and os.path.exists(out_path):
            continue
        ddg = build_ddg_for_frame_fast(pus)
        with open(out_path, "w") as f:
            json.dump(ddg, f, indent=2)


def batch_build_ddg(ipm_root, rdcost_root, output_root):
    txt_files = glob.glob(
        f"{rdcost_root}/**/*_ipm_cover_rdcost.txt",
        recursive=True)

    for rdcost_path in tqdm(txt_files, desc="Building DDGs"):
        try:
            seg_name = Path(rdcost_path).stem.replace("_ipm_cover_rdcost", "")
            rel_dir = os.path.relpath(
                os.path.dirname(rdcost_path), rdcost_root)
            ipm_path = os.path.join(
                ipm_root, rel_dir, f"{seg_name}_cover_ipm.txt")
            out_dir = os.path.join(output_root, rel_dir, seg_name)
            os.makedirs(out_dir, exist_ok=True)

            if not os.path.exists(ipm_path):
                print(f"[WARN] IPM 文件未找到：{ipm_path}")
                continue

            print(f"[INFO] Processing: {seg_name}")
            build_ddg_all_frames(ipm_path, rdcost_path, out_dir)

        except Exception as e:
            print(f"[ERROR] 处理失败：{rdcost_path}，原因：{e}")


if __name__ == "__main__":
    ipm_root = r"D:\Users\13321\PycharmProjects\Tar4Project\dec_ori_0.3_32"
    rdcost_root = r".\results_32_rdcost"
    output_root = "utils/32_ddg"
    batch_build_ddg(ipm_root, rdcost_root, output_root)
    print("✅ Batch DDG build completed.")
