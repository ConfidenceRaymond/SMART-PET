from __future__ import annotations

import json

import pytest

from smartpet.config import load_train_config


def _base(tmp_path):
    return {
        "train_csv": str(tmp_path / "train.csv"),
        "val_csv": str(tmp_path / "val.csv"),
        "mni_reference": str(tmp_path / "ref.nii.gz"),
        "out_dir": str(tmp_path / "run"),
        "attention_levels": [2, 3],
    }


def test_config_loads_attention_and_finetune_path(tmp_path):
    payload = _base(tmp_path)
    payload["init_checkpoint"] = str(tmp_path / "best.pt")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    config = load_train_config(path)
    assert config.attention_levels == (2, 3)
    assert config.init_checkpoint.endswith("best.pt")


def test_config_rejects_unknown_fields(tmp_path):
    payload = _base(tmp_path)
    payload["typo_field"] = 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown"):
        load_train_config(path)
