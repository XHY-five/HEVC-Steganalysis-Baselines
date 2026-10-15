import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def encode_coeff_hevc_vectorized(coeff, block_map, use_attention=False):
    """HEVC系数编码: one-hot + block类型 + 4×4掩码"""
    coeff = np.clip(coeff, -32, 32).astype(np.int32)
    coeff_shifted = coeff + 32  # [-32,32] → [0,64]
    H, W = coeff.shape
    if use_attention:
        feature_map = np.zeros((66, H, W), dtype=np.float32)  # 33→34，新增1通道
    else:
        feature_map = np.zeros((65, H, W), dtype=np.float32)  # 33→34，新增1通道
    # one-hot 编码
    valid_mask = (coeff_shifted >= 0) & (coeff_shifted < 64)
    feature_map[coeff_shifted[valid_mask],
                np.where(valid_mask)[0],
                np.where(valid_mask)[1]] = 1.0

    # 块类型编码 (1~5 → 4x4~64x64)
    feature_map[64] = block_map

    # === 4×4 掩码通道 ===
    if use_attention:
        mask_4x4 = (block_map == 1).astype(np.float32)
        feature_map[65] = mask_4x4
    # print("encode_coeff_hevc_vectorized -> feature_map:", feature_map.shape)
    # # <-- 调试
    return feature_map


class IPMCoeffDualDataset(Dataset):
    def __init__(self, data_root, label_map_path, use_attention=False):
        self.data_root = data_root
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.samples = list(self.label_map.items())

        self.use_attention = use_attention
        # 构建路径索引
        self.path_map = {}
        for root, _, files in os.walk(self.data_root):
            for fname in files:
                self.path_map[fname] = os.path.join(root, fname)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_id, label = self.samples[index]

        # 路径解析
        if 'stego' in frame_id:
            base_id = frame_id.replace('_stego', '')
            segment_name, frame_idx = base_id.split('_frame_')
            ipm_file = f"{segment_name}/frame_{frame_idx}_stego_ipm.npy"
            coeff_file = f"{segment_name}/frame_{frame_idx}_stego_coeff.npy"
            block_file = f"{segment_name}/frame_{frame_idx}_stego_block.npy"
        else:
            segment_name, frame_idx = frame_id.split('_frame_')
            ipm_file = f"{segment_name}/frame_{frame_idx}_ipm.npy"
            coeff_file = f"{segment_name}/frame_{frame_idx}_coeff.npy"
            block_file = f"{segment_name}/frame_{frame_idx}_block.npy"

        # 加载数据
        ipm = np.load(self.path_map[os.path.basename(ipm_file)])
        coeff = np.load(self.path_map[os.path.basename(coeff_file)])
        block_map = np.load(self.path_map[os.path.basename(block_file)])

        # === IPM 分支 ===
        ipm_onehot = np.eye(35)[ipm.astype(np.uint8)]  # [H, W, 35]
        ipm_onehot = ipm_onehot.transpose(2, 0, 1)  # [35, H, W]
        H, W = ipm.shape
        pos_x = np.tile(np.linspace(0, 1, W), (H, 1))
        pos_y = np.tile(np.linspace(0, 1, H).reshape(-1, 1), (1, W))
        ipm_tensor = np.concatenate(
            [ipm_onehot, pos_x[None], pos_y[None]], axis=0)  # [37, H, W]

        # === Coeff 分支 ===
        coeff_tensor = encode_coeff_hevc_vectorized(
            coeff, block_map, use_attention=self.use_attention)  # [33, H, W]
        pos_xc = np.tile(np.linspace(0, 1, W), (H, 1))
        pos_yc = np.tile(np.linspace(0, 1, H).reshape(-1, 1), (1, W))
        coeff_tensor = np.concatenate(
            [coeff_tensor, pos_xc[None], pos_yc[None]], axis=0)  # [35, H, W]
        # print("encode after concat:", coeff_tensor.shape)
        return (
            torch.tensor(ipm_tensor, dtype=torch.float32),
            torch.tensor(coeff_tensor, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )
