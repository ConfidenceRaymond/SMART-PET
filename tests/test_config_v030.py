from __future__ import annotations

import json
from pathlib import Path

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


def test_config_rejects_wrong_scalar_types_before_training(tmp_path):
    payload = _base(tmp_path)
    payload["epochs"] = 3.7
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match="epochs"):
        load_train_config(path)


def test_config_rejects_wrong_patch_arity(tmp_path):
    payload = _base(tmp_path)
    payload["patch_size"] = [128, 128]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exactly 3"):
        load_train_config(path)


def test_config_rejects_fractional_patch_size(tmp_path):
    payload = _base(tmp_path)
    payload["patch_size"] = [128.9, 128, 128]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match="patch_size"):
        load_train_config(path)


def test_config_accepts_batch_size_alias_in_overrides(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_base(tmp_path)))
    config = load_train_config(path, overrides={"batch_size_per_rank": 2})
    assert config.batch_size == 2


def test_corrected_s4_config_fields_are_strictly_loaded(tmp_path):
    payload = _base(tmp_path)
    payload.update(
        {
            "similarity_mode": "scale_consistent",
            "encoder_convs_per_level": 2,
            "channel_spatial_input_projection": True,
            "generator_spectral_norm": False,
            "discriminator_spectral_norm": True,
        }
    )
    path = tmp_path / "corrected.json"
    path.write_text(json.dumps(payload))
    config = load_train_config(path)
    assert config.similarity_mode == "scale_consistent"
    assert config.encoder_convs_per_level == 2
    assert config.channel_spatial_input_projection is True
    assert config.discriminator_spectral_norm is True


def test_architecture_boolean_fields_reject_integer_aliases(tmp_path):
    payload = _base(tmp_path)
    payload["discriminator_spectral_norm"] = 1
    path = tmp_path / "invalid_bool.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match="discriminator_spectral_norm"):
        load_train_config(path)


def test_repository_phase2b_corrected_s4_candidate_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_train_config(root / "configs" / "phase2b_corrected_s4_candidate.json")
    assert config.similarity_mode == "scale_consistent"
    assert config.encoder_convs_per_level == 2
    assert config.channel_spatial_input_projection is True
    assert config.generator_spectral_norm is False
    assert config.discriminator_spectral_norm is True


def test_maintained_configs_default_to_nondeterministic_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("train_from_scratch.json", "phase2b_corrected_s4_candidate.json", "finetune.json"):
        payload = json.loads((root / "configs" / name).read_text())
        assert payload["deterministic"] is False
