import json

import numpy as np
import os


def save_ipm_and_coeff(frame_num, ipm_map, coeff_map, block_map, output_dir):
    np.save(os.path.join(output_dir, f'frame_{frame_num}_ipm.npy'), ipm_map)
    np.save(
        os.path.join(
            output_dir,
            f'frame_{frame_num}_coeff.npy'),
        coeff_map)
    np.save(
        os.path.join(
            output_dir,
            f'frame_{frame_num}_block.npy'),
        block_map)  # 新增块大小图


def save_steg_ipm_and_coeff(
        frame_num, ipm_map, coeff_map, block_map, output_dir):
    np.save(
        os.path.join(
            output_dir,
            f'frame_{frame_num}_stego_ipm.npy'),
        ipm_map)
    np.save(
        os.path.join(
            output_dir,
            f'frame_{frame_num}_stego_coeff.npy'),
        coeff_map)
    np.save(
        os.path.join(
            output_dir,
            f'frame_{frame_num}_stego_block.npy'),
        block_map)  # 新增块大小图


def generate_label_map(data_dir, output_json='label_map.json'):
    label_map = {}
    for fname in os.listdir(data_dir):
        if fname.endswith('_ipm.npy'):
            frame_id = fname.split('_')[1]
            if 'stego' in fname.lower():
                label_map[frame_id] = 1
            else:
                label_map[frame_id] = 0
    with open(output_json, 'w') as f:
        json.dump(label_map, f)
    print(f"Label map saved to {output_json}")


if __name__ == "__main__":
    generate_label_map('./data/train')
