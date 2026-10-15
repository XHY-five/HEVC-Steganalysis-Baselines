import os
import re
import json
import random
from collections import defaultdict

data_root = './480p/data_0.3_QP'   # 包含所有480P视频的映射图子文件夹的根目录
# data_root = './data_0.3_QP'  # 包含所有1080P视频子文件夹的根目录
videos = [
    'BasketballDrill', 'BasketballDrillText', 'BQMall', 'PartyScene', 'RaceHorses'
]  # 仅包含视频名称（不含后缀）

# ===== 配置 =====
CLIP_SIZE = 5
frame_pat = re.compile(r'frame_(\d+)')


videos_sorted = sorted(videos, key=len, reverse=True)


def match_video_by_segment(seg_name: str):
    s = seg_name.lower()
    for v in videos_sorted:
        if s.startswith(v.lower()):
            return v
    return None


def extract_frame_id(filename, segment_folder):
    """保持原样：返回 segmentName_frameID"""
    match = re.search(r'frame_(\d+)', filename)
    return f"{segment_folder}_frame_{match.group(1)}" if match else None


def extract_frame_idx(filename):
    m = frame_pat.search(filename)
    return int(m.group(1)) if m else None


for split_id in range(1, 2):
    random.seed(split_id)

    # 用 clip 分组收集：key = (segment_folder, clip_id)
    cover_by_clip = defaultdict(list)
    stego_by_clip = defaultdict(list)

    frame_to_clip = {}

    for video in videos:

        for segment_folder in os.listdir(data_root):

            vmatch = match_video_by_segment(segment_folder)
            if vmatch != video:
                continue

            segment_path = os.path.join(data_root, segment_folder)
            if not os.path.isdir(segment_path):
                continue

            for fname in os.listdir(segment_path):
                frame_id = extract_frame_id(fname, segment_folder)
                if not frame_id:
                    continue

                fidx = extract_frame_idx(fname)
                if fidx is None:
                    continue
                clip_id = fidx // CLIP_SIZE
                clip_key = (segment_folder, clip_id)

                if 'stego' in fname.lower():
                    stego_by_clip[clip_key].append(
                        frame_id)  # 注意：仍然存原始 frame_id
                    frame_to_clip[frame_id + '_stego'] = clip_key
                else:
                    cover_by_clip[clip_key].append(frame_id)
                    frame_to_clip[frame_id] = clip_key

    # ===== 基于 clip 的 8:2 train/test 划分（不改 frame_id）=====
    train_label_map = {}
    test_label_map = {}

    # 一个 clip 可能只在一个 video 下，因此直接对全部 clip_keys 划分即可
    all_clip_keys = sorted(
        set(list(cover_by_clip.keys()) + list(stego_by_clip.keys())))
    random.shuffle(all_clip_keys)
    n_train = int(len(all_clip_keys) * 0.8)
    train_clips = set(all_clip_keys[:n_train])
    test_clips = set(all_clip_keys[n_train:])

    for ck in train_clips:
        for fid in cover_by_clip.get(ck, []):
            train_label_map[fid] = 0
        for fid in stego_by_clip.get(ck, []):
            train_label_map[fid + '_stego'] = 1

    for ck in test_clips:
        for fid in cover_by_clip.get(ck, []):
            test_label_map[fid] = 0
        for fid in stego_by_clip.get(ck, []):
            test_label_map[fid + '_stego'] = 1

    # ===== 验证集：从“训练 clips”里再按 clip 抽 10%（仍然不改 frame_id）=====
    clip_to_train_frames = defaultdict(list)
    for fid in train_label_map:
        ck = frame_to_clip[fid]  # 这里的 fid 就是上面放进去的 key（含 _stego 的也能映射）
        clip_to_train_frames[ck].append(fid)

    clip_ids = list(clip_to_train_frames.keys())
    random.shuffle(clip_ids)
    val_clip_count = max(1, int(len(clip_ids) * 0.1)) if clip_ids else 0
    val_clips = set(clip_ids[:val_clip_count])

    val_label_map = {}
    new_train_label_map = {}

    for ck, fids in clip_to_train_frames.items():
        target = val_label_map if ck in val_clips else new_train_label_map
        for fid in fids:
            target[fid] = train_label_map[fid]

    # ===== 输出 =====
    output_dir = './480p/splits_0.3_QP'  # 输出目录
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f'label_map_train_split{split_id}.json'), 'w') as f:
        json.dump(new_train_label_map, f)
    with open(os.path.join(output_dir, f'label_map_test_split{split_id}.json'), 'w') as f:
        json.dump(test_label_map, f)
    with open(os.path.join(output_dir, f'label_map_val_split{split_id}.json'), 'w') as f:
        json.dump(val_label_map, f)

    # ===== 统计 & 交叉检查 =====
    train_cover = sum(1 for v in new_train_label_map.values() if v == 0)
    train_stego = sum(1 for v in new_train_label_map.values() if v == 1)
    val_cover = sum(1 for v in val_label_map.values() if v == 0)
    val_stego = sum(1 for v in val_label_map.values() if v == 1)
    test_cover = sum(1 for v in test_label_map.values() if v == 0)
    test_stego = sum(1 for v in test_label_map.values() if v == 1)

    print(f"Split{split_id}: Train - Cover: {train_cover}, Stego: {train_stego} | "
          f"Val - Cover: {val_cover}, Stego: {val_stego} | "
          f"Test - Cover: {test_cover}, Stego: {test_stego}")

    overlap = (set(new_train_label_map) & set(val_label_map)) | \
              (set(new_train_label_map) & set(test_label_map)) | \
              (set(val_label_map) & set(test_label_map))
    print(f"Overlap count: {len(overlap)}")
    print(f"Split{split_id} label maps generated.\n")

print("Clip(5) split done without renaming or changing frame_id format.")
