# import os
# import json
# import glob
# from collections import defaultdict, deque
# from tqdm import tqdm
# from pathlib import Path
# import numpy as np
#
# def build_topo_layers(nodes, edges):
#     graph = defaultdict(list)
#     indeg = defaultdict(int)
#     for u, v in edges:
#         graph[u].append(v)
#         indeg[v] += 1
#     layers, queue, visited = [], deque([n["id"] for n in nodes if indeg[n["id"]] == 0]), set()
#     while queue:
#         layer = []
#         for _ in range(len(queue)):
#             nid = queue.popleft()
#             if nid in visited: continue
#             visited.add(nid)
#             layer.append(nid)
#             for succ in graph[nid]:
#                 indeg[succ] -= 1
#                 if indeg[succ] == 0:
#                     queue.append(succ)
#         if layer: layers.append(layer)
#     return layers
#
# def get_valid_pm1_candidates(node, gamma=1.0, layer_id=0):
#     ipm_ori = node["ipm"]
#     rho_d = node.get("rho_d", 0.0)  # ✅ 从 DDG 中读取
#     candidates = node.get("candidates", {})
#     result = []
#
#     for delta in [-1, 1]:
#         ipm_new = ipm_ori + delta
#         if 0 <= ipm_new <= 34 and str(ipm_new) in candidates:
#             dist_ori = candidates.get(str(ipm_ori), rho_d)
#             dist_new = candidates[str(ipm_new)]
#             r_s = abs(dist_new - dist_ori)  # 只是辅助输出
#             rho =r_s + rho_d + gamma * layer_id  # ✅ 最终驱动项
#
#             result.append({
#                 "id": node["id"],
#                 "x": node["x"],
#                 "y": node["y"],
#                 "w": node["w"],
#                 "h": node["h"],
#                 "ori": ipm_ori,
#                 "alt": ipm_new,
#                 "rho": rho,
#                 "rho_d": rho_d,
#                 "rho_s": r_s
#             })
#
#     if result:
#         return [min(result, key=lambda c: c["rho"])]
#     return []
#
# def apply_updist_strategy(nodes, modified_ids, eta=5.0):
#     id2node = {n["id"]: n for n in nodes}
#     modified_pus = [id2node[mid] for mid in modified_ids]
#     modified_8x8_cu_set = set()
#     modified_positions = set()
#
#     for pu in modified_pus:
#         if pu["w"] == 4 and pu["h"] == 4:
#             cu_x = pu["x"] // 8
#             cu_y = pu["y"] // 8
#             modified_8x8_cu_set.add((cu_x, cu_y))
#         for dx in range(pu["w"]):
#             for dy in range(pu["h"]):
#                 modified_positions.add((pu["x"] + dx, pu["y"] + dy))
#
#     for node in nodes:
#         if node["id"] in modified_ids:
#             continue
#
#         # 提取必要参数
#         ipm_ori = node["ipm"]
#         candidates = node.get("candidates", {})
#         dist_ori = candidates.get(str(ipm_ori), 0.0)
#         rho_d = node.get("rho_d", 0.0)
#
#         # 计算 rho_s
#         alt_diffs = [abs(candidates[str(ipm)] - dist_ori)
#                      for ipm in range(35)
#                      if str(ipm) in candidates and ipm != ipm_ori]
#         rho_s = min(alt_diffs) if alt_diffs else 0.0
#
#         # 检查是否满足 ψ=0 且 Nc=0
#         cu_id = (node["x"] // 8, node["y"] // 8)
#         in_modified_cu = cu_id in modified_8x8_cu_set
#
#         ref_zone = set()
#         ref_zone.update((node["x"] + dx, node["y"] - 1) for dx in range(node["w"]))
#         ref_zone.update((node["x"] - 1, node["y"] + dy) for dy in range(node["h"]))
#         ref_zone.add((node["x"] - 1, node["y"] - 1))
#         ref_hit = any(pos in modified_positions for pos in ref_zone)
#
#         if not in_modified_cu and not ref_hit:
#             rho = rho_s + rho_d  # 原始值
#         else:
#             rho = rho_s + (rho_d / eta)
#
#         node["rho"] = rho
#         node["rho_s"] = rho_s
#         node["rho_d"] = rho_d
#
# def layered_embedding_plan(ddg_path, output_path, gamma=1.0, eta=5.0, step=6, frame_payload_ratio=None):
#     with open(ddg_path, 'r') as f:
#         ddg = json.load(f)
#
#     nodes = ddg["nodes"]
#     edges = ddg["edges"]
#     id2node = {n["id"]: n for n in nodes}
#
#     # === Topo 分层 ===
#     layers = build_topo_layers(nodes, edges)
#     final_plan = []
#     embedded_ids = set()
#     embedded_xy = set()
#
#     # === 如果设置了 frame_payload_ratio，先确定总可嵌入数量 ===
#     all_4x4 = [n for n in nodes if n["w"] == 4 and n["h"] == 4 and 2 <= n["ipm"] <= 33]
#     if frame_payload_ratio is not None:
#         total_candidates = len(all_4x4)
#         target_payload = int(total_candidates * frame_payload_ratio)
#         # 按 rho 排序选出前 target_payload
#         sorted_candidates = sorted(all_4x4, key=lambda n: n.get("rho_d", 1e9))
#         selected_ids = {n["id"] for n in sorted_candidates[:target_payload]}
#     else:
#         selected_ids = None  # 使用原 step 分层策略
#
#     for start in range(0, len(layers), step):
#         batch_layers = layers[start:start+step]
#         batch_plan = []
#
#         for layer_id, layer in enumerate(batch_layers, start=start):
#             for nid in layer:
#                 node = id2node[nid]
#                 if not (node["w"] == 4 and node["h"] == 4): continue
#                 if not (2 <= node["ipm"] <= 33): continue
#                 if node["id"] in embedded_ids: continue
#                 if (node["x"], node["y"]) in embedded_xy: continue
#                 if selected_ids is not None and node["id"] not in selected_ids: continue
#
#                 cands = get_valid_pm1_candidates(node, gamma, layer_id)
#                 if cands:
#                     best = cands[0]
#                     best["ctu"] = node.get("ctu_addr", -1)
#                     batch_plan.append(best)
#                     embedded_ids.add(node["id"])
#                     embedded_xy.add((node["x"], node["y"]))
#
#         final_plan.extend(batch_plan)
#         new_ids = [p["id"] for p in batch_plan]
#         apply_updist_strategy(nodes, modified_ids=new_ids, eta=eta)
#
#     with open(output_path, 'w') as f:
#         json.dump(final_plan, f, indent=2)
#     print(f"[OK] {os.path.basename(output_path)}: {len(final_plan)} embeddings saved "
#           f"(target={frame_payload_ratio})")
#
# def batch_generate_plans(ddg_root, output_root, gamma=1.0, eta=5.0, step=6):
#     ddg_files = glob.glob(f"{ddg_root}/**/frame_*_ddg.json", recursive=True)
#     os.makedirs(output_root, exist_ok=True)
#
#     for path in tqdm(ddg_files, desc="Generating plans"):
#         try:
#             rel = os.path.relpath(path, ddg_root)
#             subname = Path(rel).with_suffix("").name.replace("_ddg", "")
#             out_dir = os.path.join(output_root, os.path.dirname(rel))
#             os.makedirs(out_dir, exist_ok=True)
#             output_path = os.path.join(out_dir, f"{subname}_embedding_plan.json")
#             layered_embedding_plan(path, output_path, gamma=gamma, eta=eta, step=step)
#         except Exception as e:
#             print(f"[ERROR] {path}: {e}")
#
# def save_stc_input(embedding_plan_path, out_cover_path, out_rho_path):
#     with open(embedding_plan_path, 'r') as f:
#         plan = json.load(f)
#
#     n = len(plan)
#     cover = np.zeros(n, dtype=np.uint8)
#     rho = np.full((n, 3), np.inf, dtype=np.float32)
#     for i, p in enumerate(plan):
#         ori = p["ori"]
#         alt = p["alt"]
#         r = p["rho"]
#         cover[i] = ori
#         rho[i, 1] = 0
#         if alt == ori - 1:
#             rho[i, 0] = r
#         if alt == ori + 1:
#             rho[i, 2] = r
#
#     cover.tofile(out_cover_path)
#     rho.tofile(out_rho_path)
#     print(f"[STC] saved: cover={out_cover_path}, rho={out_rho_path}, n={n}")
#
# if __name__ == "__main__":
#     ddg_root = "utils/0.3_32_ddg"
#     output_root = "utils/32_embedding_plan"
#     os.makedirs(output_root, exist_ok=True)
#     batch_generate_plans(ddg_root, output_root, gamma=1.0, eta=5.0, step=6)
#
#     # save_stc_input("utils/0.3_8_embedding_plan/aspensegment000/frame_1_embedding_plan.json",
#     #                "stc_cover.bin", "stc_rho.bin")
# import os
# import json
# import glob
# from collections import defaultdict, deque
# from tqdm import tqdm
# from pathlib import Path
# import numpy as np
# import csv
#
# def build_topo_layers(nodes, edges):
#     graph = defaultdict(list)
#     indeg = defaultdict(int)
#     for u, v in edges:
#         graph[u].append(v)
#         indeg[v] += 1
#     layers, queue, visited = [], deque([n["id"] for n in nodes if indeg[n["id"]] == 0]), set()
#     while queue:
#         layer = []
#         for _ in range(len(queue)):
#             nid = queue.popleft()
#             if nid in visited: continue
#             visited.add(nid)
#             layer.append(nid)
#             for succ in graph[nid]:
#                 indeg[succ] -= 1
#                 if indeg[succ] == 0:
#                     queue.append(succ)
#         if layer: layers.append(layer)
#     return layers
#
# def get_valid_pm1_candidates(node, gamma=1.0, layer_id=0):
#     ipm_ori = node["ipm"]
#     rho_d = node.get("rho_d", 0.0)
#     candidates = node.get("candidates", {})
#     result = []
#     for delta in [-1, 1]:
#         ipm_new = ipm_ori + delta
#         if 0 <= ipm_new <= 34 and str(ipm_new) in candidates:
#             dist_ori = candidates.get(str(ipm_ori), rho_d)
#             dist_new = candidates[str(ipm_new)]
#             r_s = abs(dist_new - dist_ori)
#             rho = r_s + rho_d + gamma * layer_id
#             result.append({
#                 "id": node["id"], "x": node["x"], "y": node["y"], "w": node["w"], "h": node["h"],
#                 "ori": ipm_ori, "alt": ipm_new, "rho": rho, "rho_d": rho_d, "rho_s": r_s
#             })
#     if result:
#         return [min(result, key=lambda c: c["rho"])]
#     return []
#
# def apply_updist_strategy(nodes, modified_ids, eta=8.0):
#     id2node = {n["id"]: n for n in nodes}
#     modified_pus = [id2node[mid] for mid in modified_ids]
#     modified_8x8_cu_set = set()
#     modified_positions = set()
#
#     for pu in modified_pus:
#         if pu["w"] == 4 and pu["h"] == 4:
#             cu_x, cu_y = pu["x"] // 8, pu["y"] // 8
#             modified_8x8_cu_set.add((cu_x, cu_y))
#         for dx in range(pu["w"]):
#             for dy in range(pu["h"]):
#                 modified_positions.add((pu["x"] + dx, pu["y"] + dy))
#
#     for node in nodes:
#         if node["id"] in modified_ids:
#             continue
#         ipm_ori = node["ipm"]
#         candidates = node.get("candidates", {})
#         dist_ori = candidates.get(str(ipm_ori), 0.0)
#         rho_d = node.get("rho_d", 0.0)
#         alt_diffs = [abs(candidates[str(ipm)] - dist_ori)
#                      for ipm in range(35)
#                      if str(ipm) in candidates and ipm != ipm_ori]
#         rho_s = min(alt_diffs) if alt_diffs else 0.0
#         cu_id = (node["x"] // 8, node["y"] // 8)
#         in_modified_cu = cu_id in modified_8x8_cu_set
#         ref_zone = set()
#         ref_zone.update((node["x"] + dx, node["y"] - 1) for dx in range(node["w"]))
#         ref_zone.update((node["x"] - 1, node["y"] + dy) for dy in range(node["h"]))
#         ref_zone.add((node["x"] - 1, node["y"] - 1))
#         ref_hit = any(pos in modified_positions for pos in ref_zone)
#         if not in_modified_cu and not ref_hit:
#             rho = rho_s + rho_d
#         else:
#             rho = rho_s + (rho_d / eta)
#         node["rho"] = rho
#         node["rho_s"] = rho_s
#         node["rho_d"] = rho_d
#
# def layered_embedding_plan(ddg_path, output_path, gamma=1.0, eta=8.0, step=4, frame_payload_ratio=None):
#     with open(ddg_path, 'r') as f:
#         ddg = json.load(f)
#     nodes = ddg["nodes"]
#     edges = ddg["edges"]
#     id2node = {n["id"]: n for n in nodes}
#     layers = build_topo_layers(nodes, edges)
#     final_plan, embedded_ids, embedded_xy = [], set(), set()
#
#     # === payload 预估 ===
#     all_4x4 = [n for n in nodes if n["w"] == 4 and n["h"] == 4 and 2 <= n["ipm"] <= 33]
#     if frame_payload_ratio is not None:
#         total_candidates = len(all_4x4)
#         target_payload = int(total_candidates * frame_payload_ratio / 0.85)
#     else:
#         target_payload = None
#
#     for start in range(0, len(layers), step):
#         batch_layers = layers[start:start+step]
#         batch_plan = []
#         for layer_id, layer in enumerate(batch_layers, start=start):
#             for nid in layer:
#                 node = id2node[nid]
#                 if not (node["w"] == 4 and node["h"] == 4): continue
#                 if not (2 <= node["ipm"] <= 33): continue
#                 if node["id"] in embedded_ids or (node["x"], node["y"]) in embedded_xy: continue
#                 cands = get_valid_pm1_candidates(node, gamma, layer_id)
#                 if cands:
#                     best = cands[0]
#                     best["ctu"] = node.get("ctu_addr", -1)
#                     batch_plan.append(best)
#                     embedded_ids.add(node["id"])
#                     embedded_xy.add((node["x"], node["y"]))
#         final_plan.extend(batch_plan)
#         # === 动态截断 ===
#         if target_payload and len(final_plan) >= target_payload:
#             final_plan = final_plan[:target_payload]
#             break
#         new_ids = [p["id"] for p in batch_plan]
#         apply_updist_strategy(nodes, modified_ids=new_ids, eta=eta)
#
#     # === 统计信息 ===
#     alt_minus = sum(1 for p in final_plan if p["alt"] < p["ori"])
#     alt_plus = len(final_plan) - alt_minus
#     rho_vals = [p["rho"] for p in final_plan]
#     stats = {
#         "frame": Path(output_path).stem,
#         "target": target_payload,
#         "actual": len(final_plan),
#         "bias(%)": (len(final_plan) - (target_payload or len(final_plan))) / (target_payload or 1) * 100,
#         "alt_-1": alt_minus,
#         "alt_+1": alt_plus,
#         "rho_mean": np.mean(rho_vals) if rho_vals else 0,
#         "rho_max": np.max(rho_vals) if rho_vals else 0
#     }
#
#     with open(output_path, 'w') as f:
#         json.dump(final_plan, f, indent=2)
#     return stats
#
# def batch_generate_plans(ddg_root, output_root, gamma=1.0, eta=8.0, step=4, frame_payload_ratio=None):
#     ddg_files = glob.glob(f"{ddg_root}/**/frame_*_ddg.json", recursive=True)
#     os.makedirs(output_root, exist_ok=True)
#     stats_list = []
#     for path in tqdm(ddg_files, desc="Generating plans"):
#         try:
#             rel = os.path.relpath(path, ddg_root)
#             subname = Path(rel).with_suffix("").name.replace("_ddg", "")
#             out_dir = os.path.join(output_root, os.path.dirname(rel))
#             os.makedirs(out_dir, exist_ok=True)
#             output_path = os.path.join(out_dir, f"{subname}_embedding_plan.json")
#             stats = layered_embedding_plan(path, output_path, gamma=gamma, eta=eta, step=step, frame_payload_ratio=frame_payload_ratio)
#             stats_list.append(stats)
#         except Exception as e:
#             print(f"[ERROR] {path}: {e}")
#
#     # # === 保存统计表 ===
#     # keys = ["frame","target","actual","bias(%)","alt_-1","alt_+1","rho_mean","rho_max"]
#     # with open(csv_path, 'w', newline='') as csvfile:
#     #     writer = csv.DictWriter(csvfile, fieldnames=keys)
#     #     writer.writeheader()
#     #     for s in stats_list:
#     #         writer.writerow(s)
#     # print(f"[INFO] Stats saved to {csv_path}")
#
# def save_stc_input(embedding_plan_path, out_cover_path, out_rho_path):
#     with open(embedding_plan_path, 'r') as f:
#         plan = json.load(f)
#     n = len(plan)
#     cover = np.zeros(n, dtype=np.uint8)
#     rho = np.full((n, 3), np.inf, dtype=np.float32)
#     for i, p in enumerate(plan):
#         ori, alt, r = p["ori"], p["alt"], p["rho"]
#         cover[i] = ori
#         rho[i, 1] = 0
#         if alt == ori - 1:
#             rho[i, 0] = r
#         if alt == ori + 1:
#             rho[i, 2] = r
#     cover.tofile(out_cover_path)
#     rho.tofile(out_rho_path)
#     print(f"[STC] saved: cover={out_cover_path}, rho={out_rho_path}, n={n}")
#
# if __name__ == "__main__":
#     ddg_root = "utils/0.3_32_ddg"
#     output_root = "utils/32_embedding_plan"
#     os.makedirs(output_root, exist_ok=True)
#     batch_generate_plans(ddg_root, output_root, gamma=1.0, eta=5.0, step=6, frame_payload_ratio=0.3)
import os
import json
import glob
from collections import defaultdict, deque
from tqdm import tqdm
from pathlib import Path


def build_topo_layers(nodes, edges):
    graph = defaultdict(list)
    indeg = defaultdict(int)
    for u, v in edges:
        graph[u].append(v)
        indeg[v] += 1
    layers, queue, visited = [], deque(
        [n["id"] for n in nodes if indeg[n["id"]] == 0]), set()
    while queue:
        layer = []
        for _ in range(len(queue)):
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            layer.append(nid)
            for succ in graph[nid]:
                indeg[succ] -= 1
                if indeg[succ] == 0:
                    queue.append(succ)
        if layer:
            layers.append(layer)
    return layers


def filter_uniform_8x8(pus):
    """去掉 8×8 CU 中所有 4×4 PU 的 IPM 都相同的块"""
    grid = defaultdict(list)
    for pu in pus:
        if pu["w"] == 4 and pu["h"] == 4:
            cu_x, cu_y = pu["x"] // 8, pu["y"] // 8
            grid[(cu_x, cu_y)].append(pu)
    skip_ids = set()
    for cu, blocks in grid.items():
        ipms = [b["ipm"] for b in blocks]
        if len(blocks) == 4 and len(set(ipms)) == 1:  # 4块都存在且IPM相同
            skip_ids.update([b["id"] for b in blocks])
    return skip_ids


def get_valid_pm1_candidates(node, gamma=1.0, layer_id=0):
    ipm_ori = node["ipm"]
    rho_d = node.get("rho_d", 0.0)
    candidates = node.get("candidates", {})
    result = []
    for delta in [-1, 1]:
        ipm_new = ipm_ori + delta
        if 0 <= ipm_new <= 34 and str(ipm_new) in candidates:
            dist_ori = candidates.get(str(ipm_ori), rho_d)
            dist_new = candidates[str(ipm_new)]
            r_s = abs(dist_new - dist_ori)
            rho = r_s + rho_d + gamma * layer_id
            result.append({
                "id": node["id"], "x": node["x"], "y": node["y"], "w": node["w"], "h": node["h"],
                "ori": ipm_ori, "alt": ipm_new, "rho": rho, "rho_d": rho_d, "rho_s": r_s
            })
    return [min(result, key=lambda c: c["rho"])] if result else []


def layered_embedding_plan(ddg_path, output_path, gamma=1.0):
    with open(ddg_path, 'r') as f:
        ddg = json.load(f)
    nodes, edges = ddg["nodes"], ddg["edges"]
    id2node = {n["id"]: n for n in nodes}

    # === 过滤 8×8 全相等块 ===
    skip_ids = filter_uniform_8x8(nodes)
    print(
        f"[INFO] {os.path.basename(ddg_path)}: Skipped {len(skip_ids)} uniform 8x8 PUs")

    layers = build_topo_layers(nodes, edges)
    final_plan, embedded_ids, embedded_xy = [], set(), set()

    for layer_id, layer in enumerate(layers):
        for nid in layer:
            if nid in skip_ids:  # 跳过不合格块
                continue
            node = id2node[nid]
            if not (node["w"] == 4 and node["h"] == 4):
                continue
            if not (2 <= node["ipm"] <= 33):
                continue
            if node["ipm"] in node.get("mpm", []):
                continue  # ← 加在这里
            if node["id"] in embedded_ids or (
                    node["x"], node["y"]) in embedded_xy:
                continue
            cands = get_valid_pm1_candidates(node, gamma, layer_id)
            if cands:
                best = cands[0]
                best["ctu"] = node.get("ctu_addr", -1)
                best["layer_id"] = layer_id
                final_plan.append(best)
                embedded_ids.add(node["id"])
                embedded_xy.add((node["x"], node["y"]))

    with open(output_path, 'w') as f:
        json.dump(final_plan, f, indent=2)
    print(
        f"[OK] {os.path.basename(output_path)}: {len(final_plan)} candidates saved (full set)")


def batch_generate_plans(ddg_root, output_root, gamma=1.0):
    ddg_files = glob.glob(f"{ddg_root}/**/frame_*_ddg.json", recursive=True)
    os.makedirs(output_root, exist_ok=True)
    for path in tqdm(ddg_files, desc="Generating plans"):
        rel = os.path.relpath(path, ddg_root)
        subname = Path(rel).with_suffix("").name.replace("_ddg", "")
        out_dir = os.path.join(output_root, os.path.dirname(rel))
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{subname}_embedding_plan.json")
        layered_embedding_plan(path, output_path, gamma=gamma)


if __name__ == "__main__":
    ddg_root = "utils/32_ddg"
    output_root = "utils/32_embedding_plan"
    os.makedirs(output_root, exist_ok=True)
    batch_generate_plans(ddg_root, output_root, gamma=1.0)
