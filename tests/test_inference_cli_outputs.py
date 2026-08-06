from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smartpet.inference.domains import prepare_shared_inference_outputs
from smartpet.inference.outputs import prediction_identifier, resolve_output_plan


def test_shared_outputs_use_one_clipped_prediction() -> None:
    raw = np.array([-0.5, 0.0, np.arcsinh(2.0)], dtype=np.float32)
    normalized, suv, clipped = prepare_shared_inference_outputs(raw, asinh_scale=1.0)

    assert clipped == 1
    np.testing.assert_array_equal(
        normalized,
        np.array([0.0, 0.0, np.arcsinh(2.0)], dtype=np.float32),
    )
    np.testing.assert_allclose(
        suv,
        np.sinh(normalized).astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_prediction_identifier_is_content_based_and_deterministic() -> None:
    first = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    second = first.copy()
    changed = np.array([0.0, 1.0, 3.0], dtype=np.float32)

    assert prediction_identifier(first) == prediction_identifier(second)
    assert prediction_identifier(first) != prediction_identifier(changed)
    assert len(prediction_identifier(first)) == 64


def test_legacy_output_defaults_to_suv() -> None:
    plan = resolve_output_plan(
        output="prediction.nii.gz",
        output_domain=None,
        normalized_output=None,
        suv_output=None,
        metadata_json=None,
    )

    assert plan.legacy_output == Path("prediction.nii.gz")
    assert plan.legacy_domain == "suv"
    assert not plan.is_shared_output_mode
    assert plan.metadata_json is None


def test_dual_output_creates_default_metadata_json() -> None:
    plan = resolve_output_plan(
        output=None,
        output_domain=None,
        normalized_output="results/prediction_normalized.nii.gz",
        suv_output="results/prediction_suv.nii.gz",
        metadata_json=None,
    )

    assert plan.is_shared_output_mode
    assert plan.normalized_output == Path("results/prediction_normalized.nii.gz")
    assert plan.suv_output == Path("results/prediction_suv.nii.gz")
    assert plan.metadata_json == Path("results/prediction_normalized_prediction.json")


def test_output_modes_cannot_be_mixed() -> None:
    with pytest.raises(ValueError, match="either legacy"):
        resolve_output_plan(
            output="legacy.nii.gz",
            output_domain="suv",
            normalized_output="normalized.nii.gz",
            suv_output=None,
            metadata_json=None,
        )


def test_shared_output_rejects_output_domain() -> None:
    with pytest.raises(ValueError, match="only valid with legacy"):
        resolve_output_plan(
            output=None,
            output_domain="suv",
            normalized_output="normalized.nii.gz",
            suv_output="suv.nii.gz",
            metadata_json=None,
        )


def test_at_least_one_output_is_required() -> None:
    with pytest.raises(ValueError, match="Provide --output"):
        resolve_output_plan(
            output=None,
            output_domain=None,
            normalized_output=None,
            suv_output=None,
            metadata_json=None,
        )


def test_shared_output_paths_must_be_different() -> None:
    with pytest.raises(ValueError, match="must be different"):
        resolve_output_plan(
            output=None,
            output_domain=None,
            normalized_output="same.nii.gz",
            suv_output="same.nii.gz",
            metadata_json=None,
        )
