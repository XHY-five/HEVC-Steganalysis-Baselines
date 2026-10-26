import numpy as np
import re

# 通用 CU 信息解析器


def parse_cu_info(line):
    x = int(re.search(r'x=(\d+)', line).group(1))
    y = int(re.search(r'y=(\d+)', line).group(1))
    depth = int(re.search(r'depth=(\d+)', line).group(1))
    size_match = re.search(r'size=(\d+)x(\d+)', line)
    h, w = (int(size_match.group(1)), int(
        size_match.group(2))) if size_match else (0, 0)
    part_size = int(re.search(r'paritionSize=(\d+)', line).group(1))
    return x, y, h, w, depth, part_size

# 解析 IPM 信息


def parse_ipm_multiframe(file_content):
    frames = {}
    lines = file_content.strip().splitlines()
    i = 0
    current_frame = None
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Frame") and "begin" in line:
            current_frame = int(line.split()[1])
            frames[current_frame] = []
            i += 1
        elif line.startswith("CTU Address"):
            if 'paritionSize=8' in line:
                i += 1
                continue
            try:
                x, y, h, w, depth, part_size = parse_cu_info(line)
                if depth == 3 and part_size == 3:
                    h, w = 4, 4
                ipm = int(re.search(r'IPM=(\d+)', line).group(1))
                frames[current_frame].append(
                    {'x': x, 'y': y, 'size': (h, w), 'ipm': ipm})
            except Exception:
                pass
            i += 1
        else:
            i += 1
    return frames

# 构建 IPM map


def build_ipm_map(blocks, frame_width, frame_height):
    ipm_map = np.zeros((frame_height, frame_width), dtype=np.float32)
    for blk in blocks:
        x, y = blk['x'], blk['y']
        h, w = blk['size']
        ipm = blk['ipm']
        valid_h = min(h, frame_height - y)
        valid_w = min(w, frame_width - x)
        if valid_h > 0 and valid_w > 0:
            ipm_map[y:y + valid_h, x:x + valid_w] = ipm
    return ipm_map

# 解析 Coeff 信息


def parse_transform_coeff_multiframe(file_content):
    frames = {}
    lines = file_content.strip().splitlines()
    i = 0
    current_frame = None
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Frame") and "begin" in line:
            current_frame = int(line.split()[1])
            frames[current_frame] = []
            i += 1
        elif line.startswith("CTU Address"):
            if 'paritionSize=8' in line:
                i += 1
                continue
            try:
                x, y, h, w, depth, part_size = parse_cu_info(line)
                if depth == 3 and part_size == 3:
                    h, w = 4, 4
            except Exception:
                i += 1
                continue
            i += 1
            coeff_lines = []
            while i < len(lines) and lines[i].strip().startswith('['):
                row = eval(lines[i].strip().rstrip(','))
                coeff_lines.append(row)
                i += 1
            coeff = np.array(coeff_lines)
            frames[current_frame].append(
                {'x': x, 'y': y, 'size': (h, w), 'coeff': coeff})
        else:
            i += 1
    return frames


# parse_ipm_coeff.py

# 修改 build_coeff_map 函数
def build_coeff_map(blocks, frame_width, frame_height):
    # 初始化系数图（保留原始值，包括符号）
    block_size = 0
    coeff_map = np.zeros((frame_height, frame_width), dtype=np.float32)
    block_map = np.zeros((frame_height, frame_width),
                         dtype=np.int32)  # 存储块大小信息

    for blk in blocks:
        x, y = blk['x'], blk['y']
        h, w = blk['size']
        coeff = blk['coeff']
        valid_h = min(h, frame_height - y)
        valid_w = min(w, frame_width - x)

        if valid_h <= 0 or valid_w <= 0:
            continue

        # 保存原始系数值（包括符号）
            # 保存原始系数值（包括符号）
        if coeff.shape == (h, w):
            coeff_map[y:y + valid_h, x:x + valid_w] = coeff[:valid_h, :valid_w]
            # 标记块大小（1=4x4, 2=8x8, 3=16x16, 4=32x32）
            if h == 4:
                block_size = 1  # 4x4
            elif h == 8:
                block_size = 2  # 8x8
            elif h == 16:
                block_size = 3  # 16x16
            elif h == 32:
                block_size = 4  # 32x32
            elif h == 64:
                block_size = 5  # 64x64
            block_map[y:y + valid_h, x:x + valid_w] = block_size
        else:
            # 如果尺寸不匹配，填充平均值（保留符号）
            avg_coeff = np.mean(coeff)
            coeff_map[y:y + valid_h, x:x + valid_w] = avg_coeff

    # 返回系数图和块大小图
    return coeff_map, block_map
