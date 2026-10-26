"""Contract tests for the compact proposed model runner."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


RUN_PATH = Path(__file__).resolve().parents[1] / "proposed" / "run.py"
SPEC = importlib.util.spec_from_file_location("proposed_run", RUN_PATH)
proposed_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proposed_run)


def make_sample(root, segment, frame, stego, ipm):
    folder = root / segment
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "_stego" if stego else ""
    stem = f"frame_{frame}{suffix}"
    np.save(folder / f"{stem}_ipm.npy", np.full((2, 2), ipm, dtype=np.int32))
    np.save(folder / f"{stem}_coeff.npy", np.array([[-32, 31], [0, 1]]))
    np.save(folder / f"{stem}_block.npy", np.ones((2, 2), dtype=np.float32))


def test_feature_contract_has_model_channel_counts():
    ipm = np.array([[0, 34], [1, 2]])
    coeff = np.array([[-32, 31], [0, 1]])
    block = np.ones((2, 2))
    ipm_features, coeff_features = proposed_run.make_features(
        ipm, coeff, block)
    assert ipm_features.shape == (37, 2, 2)
    assert coeff_features.shape == (67, 2, 2)
    assert coeff_features[0, 0, 0] == 1
    assert coeff_features[63, 0, 1] == 1
    assert np.all(coeff_features[64] == 1)


def test_dataset_uses_segment_specific_cover_and_stego_files(tmp_path):
    make_sample(tmp_path, "clipA", 0, False, 5)
    make_sample(tmp_path, "clipA", 0, True, 7)
    make_sample(tmp_path, "clipB", 0, False, 9)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"clipA_frame_0": 0, "clipA_frame_0_stego": 1,
                                  "clipB_frame_0": 0}), encoding="utf-8")
    dataset = proposed_run.IPMCoeffDataset(tmp_path, labels)
    assert len(dataset) == 3
    assert dataset[0][0][5, 0, 0] == 1
    assert dataset[1][0][7, 0, 0] == 1
    assert dataset[2][0][9, 0, 0] == 1
    assert dataset[1][2].item() == 1


def test_invalid_ipm_is_rejected():
    with pytest.raises(ValueError, match="IPM"):
        proposed_run.make_features(
            np.array([[35]]), np.array([[0]]), np.array([[1]]))


def test_mismatched_feature_maps_are_rejected():
    with pytest.raises(ValueError, match="same 2D shape"):
        proposed_run.make_features(np.zeros((2, 2), dtype=int),
                                   np.zeros((1, 2)), np.zeros((2, 2)))


def test_invalid_frame_id_is_rejected(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text('{"bad-id": 0}', encoding="utf-8")
    dataset = proposed_run.IPMCoeffDataset(tmp_path, labels)
    with pytest.raises(ValueError, match="Invalid frame ID"):
        dataset[0]


def test_split_paths_match_existing_label_maps(tmp_path):
    paths = proposed_run.split_paths(tmp_path, 3)
    assert paths[0].name == "label_map_train_split3.json"
    assert paths[1].name == "label_map_val_split3.json"
    assert paths[2].name == "label_map_test_split3.json"


def test_nonpositive_epochs_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        proposed_run.train_fold(tmp_path, tmp_path, epochs=0)


def test_config_paths_are_relative_to_config_file(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"data_root": "data", "split_dir": "splits",
                                  "output_dir": "results", "fold": 2}), encoding="utf-8")
    values = proposed_run.load_config(config)
    assert values["data_root"] == tmp_path / "data"
    assert values["split_dir"] == tmp_path / "splits"
    assert values["output_dir"] == tmp_path / "results"
    assert values["fold"] == 2


def test_main_uses_config_without_command_line_arguments(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"data_root": "data", "split_dir": "splits",
                                  "epochs": 3}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(proposed_run, "train_fold", lambda **kwargs: calls.append(kwargs))
    proposed_run.main(config)
    assert calls[0]["epochs"] == 3
    assert calls[0]["data_root"] == tmp_path / "data"


def test_config_requires_input_paths(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="data_root"):
        proposed_run.load_config(config)


def test_one_epoch_training_smoke(tmp_path):
    data = tmp_path / "data"
    split = tmp_path / "split"
    split.mkdir()
    for frame, stego, ipm in [(0, False, 5), (1, True, 7)]:
        make_sample(data, "clip", frame, stego, ipm)
    labels = {"clip_frame_0": 0, "clip_frame_1_stego": 1}
    for kind in ("train", "val", "test"):
        (split / f"label_map_{kind}_split1.json").write_text(
            json.dumps(labels), encoding="utf-8")

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(37 + 67, 2)

        def forward(self, ipm, coeff):
            features = torch.cat((ipm.mean((2, 3)), coeff.mean((2, 3))), dim=1)
            return self.fc(features)

    result = proposed_run.train_fold(data, split, fold=1, output_dir=tmp_path / "out",
                                     epochs=1, batch_size=2, device="cpu",
                                     model_factory=TinyModel)
    assert 0 <= result["accuracy"] <= 1
    assert (tmp_path / "out" / "best_model_fold1.pth").is_file()
