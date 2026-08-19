"""Тесты безопасных ограничений путей, доступа и MC-бюджета."""

from __future__ import annotations

from pathlib import Path

import pytest

from admission_sim.model import ApplicantProfile, ApplicationRow, Dataset
from admission_sim.path_safety import ensure_not_overwriting, resolve_safe_path
from admission_sim.pipeline import estimate_probabilities_for_applicants
from admission_sim.report import write_markdown_report
from app import external_access_blocked


def _profile(code: int, consent: bool, rank: int) -> ApplicantProfile:
    app = ApplicationRow(
        applicant_code=code,
        program="A",
        priority=1,
        score=300.0 - rank,
        rank=rank,
        consent=consent,
        status="участвует",
    )
    profile = ApplicantProfile(code=code, applications=[app], consent=consent)
    profile.rebuild_index()
    return profile


def test_resolve_safe_path_blocks_escape(tmp_path: Path) -> None:
    inside = resolve_safe_path(tmp_path / "report.md", allowed_roots=(tmp_path,))
    assert inside == (tmp_path / "report.md").resolve()
    with pytest.raises(ValueError):
        resolve_safe_path(tmp_path / ".." / "outside.md", allowed_roots=(tmp_path,))


def test_write_markdown_report_requires_force_for_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    write_markdown_report(target, "v1", allowed_roots=(tmp_path,))
    assert target.read_text(encoding="utf-8") == "v1"
    with pytest.raises(ValueError):
        write_markdown_report(target, "v2", allowed_roots=(tmp_path,))
    write_markdown_report(target, "v2", allowed_roots=(tmp_path,), force=True)
    assert target.read_text(encoding="utf-8") == "v2"


def test_ensure_not_overwriting_blocks_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "exists.txt"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        ensure_not_overwriting(target, force=False)
    ensure_not_overwriting(target, force=True)


def test_external_access_guard_defaults_to_safe() -> None:
    assert not external_access_blocked("localhost", None)
    assert not external_access_blocked("127.0.0.1", None)
    assert external_access_blocked("0.0.0.0", None)
    assert not external_access_blocked("0.0.0.0", "1")


def test_personal_mc_budget_caps_competitors_and_runs(monkeypatch) -> None:
    applicants = {code: _profile(code, False, rank=code) for code in range(1, 7)}
    dataset = Dataset(
        applicants=applicants,
        seats={"A": 2},
        programs=["A"],
        source_files=[],
        incomplete_priority_codes=[],
    )

    calls: list[tuple[int, int]] = []

    class DummyEstimate:
        def __init__(self, n_simulations: int):
            self.n_simulations = n_simulations
            self.by_program = {"A": 0.0}
            self.any_loaded = 0.0
            self.external = 0.0
            self.none = 1.0
            self.mean_consent_probability = 0.0
            self.consent_model_description = "dummy"
            self.scenario = "auto"
            self.focus_program = None

    def _fake_estimate(*_args, **kwargs):
        code = int(_args[2])
        n_sim = int(kwargs["n_simulations"])
        calls.append((code, n_sim))
        return DummyEstimate(n_sim)

    monkeypatch.setattr("admission_sim.pipeline.estimate_probability", _fake_estimate)

    out = estimate_probabilities_for_applicants(
        dataset,
        applicant_codes=[1, 2, 3, 4, 5, 6],
        n_simulations=1000,
        max_applicants=3,
        max_total_runs=1500,
    )
    assert sorted(out.keys()) == [1, 2, 3]
    assert calls and all(n_sim == 500 for _code, n_sim in calls)
