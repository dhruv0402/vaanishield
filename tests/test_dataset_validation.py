"""Missing audio must never be reported as a completed physical audit."""
import json
from pathlib import Path
import pytest
from scripts.validate_dataset import validate_manifest_integrity
from scripts.verify import run_verification


@pytest.fixture
def absent_audio_manifest(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "utterance_id": "test-only", "speaker_id": "s1", "split": "train",
        "audio_path": str(tmp_path / "absent.wav"), "duration_sec": 2.0,
        "loudness_lufs_or_rms": 0.2, "label": "bona_fide",
    }) + "\n")
    registry = tmp_path / "registry.csv"
    registry.write_text("speaker_id,source_license\n" +
                        "".join(f"s{i},test-only\n" for i in range(1, 5)))
    return str(manifest), str(registry)


def test_dataset_validation_requires_physical_audio(absent_audio_manifest):
    with pytest.raises(FileNotFoundError, match="Audio file"):
        validate_manifest_integrity(*absent_audio_manifest)


def test_acceptance_gate_requires_physical_audio(absent_audio_manifest, capsys):
    with pytest.raises(SystemExit) as error:
        run_verification(manifest_path=absent_audio_manifest[0])
    assert error.value.code == 1
    assert "Audio file missing" in capsys.readouterr().out


def test_metadata_audit_reports_unverified_audio(absent_audio_manifest):
    result = validate_manifest_integrity(*absent_audio_manifest, metadata_only=True)
    assert result["validation_scope"] == "metadata_only"
    assert result["audio_files_verified"] == 0
    assert result["audio_files_unverified"] == 1


def test_report_audit_marks_undefined_intervals_unavailable(tmp_path, capsys):
    report = json.loads(Path("results/evaluation_report.json").read_text())
    report["models"]["vaanishield_fusion"]["test"]["overall"]["eer_ci"] = [None, None]
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(report))
    run_verification(eval_json=str(path), metadata_only=True)
    assert "[CHECK 6 UNAVAILABLE]" in capsys.readouterr().out


def test_report_audit_marks_undefined_eer_unavailable(tmp_path, capsys):
    report = json.loads(Path("results/evaluation_report.json").read_text())
    overall = report["models"]["vaanishield_fusion"]["test"]["overall"]
    overall.update(eer=None, eer_ci=[None, None])
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(report))
    run_verification(eval_json=str(path), metadata_only=True)
    assert "[CHECK 7 UNAVAILABLE]" in capsys.readouterr().out


def test_audit_does_not_compare_different_fusion_models(tmp_path, capsys):
    report = json.loads(Path("results/ablation_report.json").read_text())
    report["fused"]["architecture"] = "early_feature_fusion"
    report["fused"]["test_eval"]["overall"]["eer"] = 0.42
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps(report))
    run_verification(ablation_json=str(path), metadata_only=True)
    assert "[CHECK 7 NOT COMPARABLE]" in capsys.readouterr().out
