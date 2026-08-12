from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_has_dedicated_work_directory_and_scripts_use_it() -> None:
    assert (ROOT / "work" / "README.md").is_file()
    bootstrap = (ROOT / "scripts" / "bootstrap_narval.sh").read_text()
    launcher = (
        ROOT / "scripts" / "slurm" / "narval_train_ddp.sh"
    ).read_text()
    external = (ROOT / "scripts" / "preprocessing" / "prepare_external_dataset.sh").read_text()
    assert "SMARTPET_WORK_ROOT" in bootstrap
    assert "TMPDIR" in bootstrap
    assert "SMARTPET_ROOT" in launcher
    assert "SLURM_SUBMIT_DIR" in launcher
    assert 'SMARTPET_VENV:-$ROOT/.venv' in launcher
    assert "unset PYTHONPATH" in launcher
    assert "unset EBPYTHONPREFIXES" in launcher
    assert "work-dir" in external


def test_external_preprocessing_launcher_isolates_pythonpath() -> None:
    script = (
        ROOT / "scripts" / "preprocessing" / "prepare_external_dataset.sh"
    ).read_text()

    assert 'PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"' not in script
    assert 'PYTHONPATH="${ROOT}/src"' not in script
    assert "unset PYTHONPATH" in script
    assert "PYTHONNOUSERSITE=1" in script
    assert "unset EBPYTHONPREFIXES" in script
    assert "smartpet-prepare-external" in script


def test_external_preprocessing_launcher_isolates_python_environment() -> None:
    script = (
        ROOT
        / "scripts"
        / "preprocessing"
        / "prepare_external_dataset.sh"
    ).read_text()

    assert 'PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"' not in script
    assert 'PYTHONPATH="${ROOT}/src"' not in script
    assert "unset PYTHONPATH" in script
    assert "PYTHONNOUSERSITE=1" in script
    assert "unset EBPYTHONPREFIXES" in script
    assert "smartpet-prepare-external" in script
