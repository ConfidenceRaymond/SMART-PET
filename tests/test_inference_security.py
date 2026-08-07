from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("nibabel")

from smartpet.inference.engine import InferenceEngine


class _WriteMarker:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        statement = f"open({str(self.marker)!r}, 'w', encoding='utf-8').write('executed')"
        return exec, (statement,)


def test_inference_refuses_pickle_code_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    checkpoint = tmp_path / "hostile.pt"
    reference = tmp_path / "reference.nii.gz"
    reference.touch()
    torch.save({"payload": _WriteMarker(marker)}, checkpoint)

    with pytest.raises(RuntimeError, match="weights_only=True"):
        InferenceEngine(
            checkpoint=checkpoint,
            mni_reference=reference,
            amp=False,
            device="cpu",
        )

    assert not marker.exists(), "checkpoint payload executed during inference load"
