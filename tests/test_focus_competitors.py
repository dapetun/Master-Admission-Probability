"""Тесты секции конкурентов на фокусной программе."""

from __future__ import annotations

from app import competitors_score_slider_config
from admission_sim.model import ApplicantProfile, ApplicationRow, Dataset
from admission_sim.report import focus_competitor_rows, focus_competitor_table_rows
from admission_sim.simulate import simulate_enrollment


def _profile(code: int, consent: bool, rows: list[tuple[str, int, float, int]]) -> ApplicantProfile:
    apps = [
        ApplicationRow(
            applicant_code=code,
            program=program,
            priority=priority,
            score=score,
            rank=rank,
            consent=consent,
            status="участвует",
        )
        for program, priority, score, rank in rows
    ]
    profile = ApplicantProfile(code=code, applications=apps, consent=consent)
    profile.rebuild_index()
    return profile


def test_focus_competitor_rows_marks_above_and_sorts() -> None:
    applicants = {
        100: _profile(100, False, [("A", 1, 300.0, 3)]),
        200: _profile(200, True, [("A", 1, 320.0, 1)]),
        300: _profile(300, False, [("A", 1, 310.0, 2)]),
        400: _profile(400, True, [("A", 2, 290.0, 5)]),
    }
    dataset = Dataset(
        applicants=applicants,
        seats={"A": 2},
        programs=["A"],
        source_files=[],
        incomplete_priority_codes=[],
    )
    vpp = simulate_enrollment(applicants, dataset.seats, consent_only=True)
    ovp = simulate_enrollment(applicants, dataset.seats, consent_only=False)

    rows = focus_competitor_rows(dataset, 100, "A", vpp=vpp, ovp=ovp)

    assert [row.code for row in rows] == [200, 300, 400]
    assert [row.above_me for row in rows] == [True, True, False]
    assert [row.relative_to_me for row in rows] == [2, 1, -2]


def test_focus_competitor_table_rows_contains_readable_columns() -> None:
    applicants = {
        100: _profile(100, False, [("A", 1, 300.0, 3)]),
        200: _profile(200, True, [("A", 1, 320.0, 1)]),
    }
    dataset = Dataset(
        applicants=applicants,
        seats={"A": 1},
        programs=["A"],
        source_files=[],
        incomplete_priority_codes=[],
    )
    vpp = simulate_enrollment(applicants, dataset.seats, consent_only=True)
    ovp = simulate_enrollment(applicants, dataset.seats, consent_only=False)
    rows = focus_competitor_rows(dataset, 100, "A", vpp=vpp, ovp=ovp)

    table = focus_competitor_table_rows(rows)
    assert len(table) == 1
    assert table[0]["Код"] == 200
    assert table[0]["Относительно меня"] == "выше на 2"
    assert table[0]["Согласие"] == "да"
    assert table[0]["_above_me"] is True


def test_competitors_score_slider_config_limits_min_by_me_score() -> None:
    config = competitors_score_slider_config([51.0, 58.0, 63.0], me_score=58.0)
    assert config["min_value"] == 58.0
    assert config["max_value"] == 63.0
    assert config["value"] == (58.0, 63.0)
    assert config["disabled"] is False
    assert "Ниже вашего балла" in str(config["caption"])
    assert float(config["inactive_share"]) > 0


def test_competitors_score_slider_config_handles_empty_scores() -> None:
    config = competitors_score_slider_config([], me_score=58.0)
    assert config["disabled"] is True
    assert config["value"] == (0.0, 0.0)
    assert config["caption"] == "Нет данных по баллам: фильтр отключён."


def test_competitors_score_slider_config_handles_none_me_score() -> None:
    config = competitors_score_slider_config([55.0, 61.0], me_score=None)
    assert config["min_value"] == 55.0
    assert config["max_value"] == 61.0
    assert config["value"] == (55.0, 61.0)
    assert config["caption"] is None
    assert config["inactive_share"] == 0.0


def test_competitors_score_slider_config_handles_equal_scores() -> None:
    config = competitors_score_slider_config([60.0, 60.0], me_score=60.0)
    assert config["min_value"] == 60.0
    assert config["max_value"] == 60.0
    assert config["value"] == (60.0, 60.0)
    assert config["inactive_share"] == 0.0


def test_competitors_score_slider_config_handles_me_above_max() -> None:
    config = competitors_score_slider_config([55.0, 61.0], me_score=80.0)
    assert config["min_value"] == 61.0
    assert config["max_value"] == 61.0
    assert config["value"] == (61.0, 61.0)
    assert "выше максимума конкурентов" in str(config["caption"])
