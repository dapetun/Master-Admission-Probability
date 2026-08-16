"""Базовые тесты загрузки и симуляции."""

from __future__ import annotations

from pathlib import Path

import yaml

from admission_sim.load import (
    build_profiles,
    filter_profiles_by_status,
    load_dataset,
    program_and_campus_from_budget_xlsx,
    program_key,
    program_name_from_csv,
    _read_budget_xlsx,
)
from admission_sim.model import EXTERNAL, ApplicationRow
from admission_sim.scenarios import estimate_probability, what_if_consent
from admission_sim.simulate import preference_list, simulate_enrollment


def test_program_name_from_csv() -> None:
    path = Path("Математика_машинного обучения.2026-08-16_20-17-34.csv")
    assert program_name_from_csv(path) == "Математика_машинного обучения"


def _row(
    code: int,
    program: str,
    priority: int,
    score: float,
    rank: int,
    *,
    consent: bool = False,
    status: str = "Участвуете в конкурсе",
) -> ApplicationRow:
    return ApplicationRow(
        applicant_code=code,
        program=program,
        priority=priority,
        score=score,
        rank=rank,
        consent=consent,
        status=status,
    )


def test_missing_higher_priority_flag() -> None:
    rows = [
        _row(1, "A", priority=3, score=90, rank=1, consent=True),
        _row(2, "A", priority=1, score=80, rank=2, consent=True),
    ]
    profiles = build_profiles(rows)
    assert profiles[1].missing_higher_priority is True
    assert profiles[2].missing_higher_priority is False


def test_preference_list_inserts_external() -> None:
    rows = [_row(1, "A", priority=3, score=90, rank=1)]
    profile = build_profiles(rows)[1]
    assert preference_list(profile) == [EXTERNAL, EXTERNAL, "A"]


def test_cascade_simple_two_programs(tmp_path: Path) -> None:
    """
    A: 1 место, B: 1 место.

    code1: A pri1 score/rank лучше, согласие
    code2: A pri1 хуже, B pri2, согласие
    code3: B pri1, согласие

    Ожидание: 1→A, 3→B, 2 не проходит на A и идёт на B но место занято → None
    """
    rows = [
        _row(1, "A", 1, 100, 1, consent=True),
        _row(2, "A", 1, 90, 2, consent=True),
        _row(2, "B", 2, 95, 2, consent=True),
        _row(3, "B", 1, 80, 1, consent=True),
    ]
    profiles = build_profiles(rows)
    seats = {"A": 1, "B": 1}
    result = simulate_enrollment(profiles, seats, consent_only=True)
    assert result.destination(1) == "A"
    assert result.destination(3) == "B"
    assert result.destination(2) is None


def test_higher_score_bumps_after_fallback() -> None:
    """
    A: 1 место. B: 0 мест (внешняя дыра не нужна).

    X: pri1=B (нет мест / отвергнут), pri2=A, rank 1 на A
    Y: pri1=A, rank 2 на A

    X должен вытеснить Y на A.
    """
    rows = [
        _row(10, "B", 1, 50, 1, consent=True),
        _row(10, "A", 2, 100, 1, consent=True),
        _row(20, "A", 1, 80, 2, consent=True),
    ]
    profiles = build_profiles(rows)
    result = simulate_enrollment(profiles, {"A": 1, "B": 0}, consent_only=True)
    assert result.destination(10) == "A"
    assert result.destination(20) is None


def test_external_absorbs_consent_with_gap() -> None:
    rows = [
        _row(1, "A", 3, 100, 1, consent=True),
        _row(2, "A", 1, 50, 2, consent=True),
    ]
    profiles = build_profiles(rows)
    result = simulate_enrollment(profiles, {"A": 1}, consent_only=True)
    assert result.destination(1) == EXTERNAL
    assert result.destination(2) == "A"


def test_ovp_vs_vpp() -> None:
    rows = [
        _row(1, "A", 1, 100, 1, consent=False),
        _row(2, "A", 1, 90, 2, consent=True),
    ]
    profiles = build_profiles(rows)
    seats = {"A": 1}
    vpp = simulate_enrollment(profiles, seats, consent_only=True)
    ovp = simulate_enrollment(profiles, seats, consent_only=False)
    assert vpp.destination(2) == "A"
    assert vpp.destination(1) is None  # нет в assignment активных без согласия
    assert 1 not in vpp.assignment or vpp.destination(1) is None
    assert ovp.destination(1) == "A"
    assert ovp.destination(2) is None


def test_what_if_consent() -> None:
    rows = [
        _row(1, "A", 1, 100, 1, consent=False),
        _row(2, "A", 1, 90, 2, consent=True),
    ]
    profiles = build_profiles(rows)
    seats = {"A": 1}
    assert simulate_enrollment(profiles, seats).destination(2) == "A"
    forced = what_if_consent(profiles, seats, 1)
    assert forced.destination(1) == "A"
    assert forced.destination(2) is None


def test_monte_carlo_seed_stable() -> None:
    rows = [
        _row(1, "A", 1, 70, 3, consent=False),
        _row(2, "A", 1, 100, 1, consent=True),
        _row(3, "A", 1, 90, 2, consent=False),
        _row(4, "A", 1, 10, 4, consent=False),
    ]
    profiles = build_profiles(rows)
    seats = {"A": 2}
    a = estimate_probability(profiles, seats, 1, n_simulations=50, seed=7)
    b = estimate_probability(profiles, seats, 1, n_simulations=50, seed=7)
    assert a.by_program == b.by_program
    assert a.any_loaded == b.any_loaded
    assert "Авто" in a.consent_model_description or "авто" in a.consent_model_description.lower()
    assert a.scenario == "auto"


def test_monte_carlo_by_program_only_my_apps() -> None:
    """by_program не раздувается всеми ключами seats.yaml."""
    rows = [
        _row(1, "Mine", 1, 80, 2, consent=False),
        _row(2, "Mine", 1, 100, 1, consent=True),
        _row(3, "Other", 1, 90, 1, consent=True),
    ]
    profiles = build_profiles(rows)
    seats = {"Mine": 1, "Other": 5, "Unrelated": 10}
    est = estimate_probability(profiles, seats, 1, n_simulations=40, seed=1)
    assert list(est.by_program.keys()) == ["Mine"]
    assert "Other" not in est.by_program
    assert "Unrelated" not in est.by_program


def test_optimistic_lower_mean_than_pessimistic() -> None:
    from admission_sim.scenarios import estimate_consent_model

    rows = [
        _row(1, "A", 1, 100, 1, consent=True),
        _row(2, "A", 1, 90, 2, consent=False),
        _row(3, "A", 1, 80, 3, consent=False),
        _row(4, "A", 1, 10, 4, consent=False),
    ]
    profiles = build_profiles(rows)
    seats = {"A": 2}
    opt = estimate_consent_model(profiles, seats, scenario="optimistic")
    pes = estimate_consent_model(profiles, seats, scenario="pessimistic")
    assert opt.mean_undecided < pes.mean_undecided


def test_consent_model_higher_for_competitive() -> None:
    from admission_sim.scenarios import estimate_consent_model

    rows = [
        _row(1, "A", 1, 100, 1, consent=True),
        _row(2, "A", 1, 90, 2, consent=True),
        _row(3, "A", 1, 80, 3, consent=False),
        _row(4, "A", 1, 10, 4, consent=False),
    ]
    profiles = build_profiles(rows)
    model = estimate_consent_model(profiles, {"A": 2})
    assert model.competitive_rate == 1.0
    assert model.noncompetitive_rate == 0.0
    assert model.by_code[3] == model.noncompetitive_rate
    assert model.by_code[4] == model.noncompetitive_rate


def test_load_seats_accepts_extended_entries(tmp_path: Path) -> None:
    from admission_sim.load import load_seats

    path = tmp_path / "seats.yaml"
    path.write_text(
        yaml.dump(
            {
                "programs": {
                    "A": 3,
                    "B": {"seats": 5, "title": "Test", "campus": "Москва"},
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    assert load_seats(path) == {"A": 3, "B": 5}


def test_load_seats_allows_null(tmp_path: Path) -> None:
    from admission_sim.load import load_seats

    path = tmp_path / "seats.yaml"
    path.write_text(
        yaml.dump({"programs": {"A": {"seats": None}}}, allow_unicode=True),
        encoding="utf-8",
    )
    assert load_seats(path) == {"A": None}


def test_unknown_seats_absorb_as_external() -> None:
    rows = [
        _row(1, "SPB", 1, 100, 1, consent=True),
        _row(1, "MSK", 2, 90, 2, consent=True),
        _row(2, "MSK", 1, 80, 1, consent=True),
    ]
    profiles = build_profiles(rows)
    # SPB seats unknown → 1 goes EXTERNAL, 2 gets MSK
    result = simulate_enrollment(profiles, {"SPB": None, "MSK": 1}, consent_only=True)
    assert result.destination(1) == EXTERNAL
    assert result.destination(2) == "MSK"


def test_markdown_report_focuses_on_my_programs() -> None:
    from admission_sim.model import Dataset
    from admission_sim.report import build_markdown_report
    from admission_sim.scenarios import counterfactuals_for_threats

    rows = [
        _row(1, "Mine", 3, 100, 1, consent=False),
        _row(2, "Mine", 1, 90, 2, consent=True),
        _row(3, "Noise", 1, 80, 1, consent=True),
    ]
    profiles = build_profiles(rows)
    seats = {"Mine": 2, "Noise": 5, "Extra": 10}
    dataset = Dataset(
        applicants=profiles,
        seats=seats,
        programs=["Mine", "Noise", "Extra"],
        source_files=["a.xlsx", "b.xlsx", "c.xlsx"],
        incomplete_priority_codes=[1],
    )
    vpp = simulate_enrollment(profiles, seats, consent_only=True)
    vpp_if = what_if_consent(profiles, seats, 1)
    ovp = simulate_enrollment(profiles, seats, consent_only=False)
    prob = estimate_probability(profiles, seats, 1, n_simulations=20, seed=3)
    threats = counterfactuals_for_threats(profiles, seats, 1, limit=5)
    md = build_markdown_report(
        dataset,
        1,
        vpp=vpp,
        vpp_if_consent=vpp_if,
        ovp=ovp,
        counterfactuals=threats,
        probability=prob,
        include_pending=False,
    )
    assert "| Mine |" in md
    assert "нет приоритетов: 1, 2" in md
    assert "### Ваши программы" in md
    assert "#### Mine" in md
    assert "### Остальные программы (сводка)" in md
    assert "Других конкурсов в модели: **2**" in md
    # Не дампим чужие конкурсы как отдельные секции
    assert "#### Noise" not in md
    assert "#### Extra" not in md
    assert md.count("| Noise |") == 0
    assert "Источники: 3 файлов" in md


def test_compound_status_filter() -> None:
    """Составной статус MAGREPORTS не должен выпадать из модели."""
    rows = [
        _row(
            1,
            "A",
            1,
            45,
            1,
            status="Участвует в конкурсе / Ожидание результатов ВИ",
        ),
        _row(2, "A", 1, 40, 2, status="На рассмотрении"),
        _row(3, "A", 1, 0, 3, status="Ожидание результатов ВИ"),
    ]
    profiles = build_profiles(rows)
    active = filter_profiles_by_status(profiles, include_pending=False)
    assert 1 in active
    assert 2 not in active
    assert 3 not in active
    pending = filter_profiles_by_status(profiles, include_pending=True)
    assert 1 in pending
    assert 3 in pending
    assert 2 not in pending


def test_pending_apps_kept_by_default() -> None:
    """По умолчанию pending-заявки остаются в профиле (не «теряются»)."""
    rows = [
        _row(
            1,
            "Прикладные модели ИИ",
            1,
            0,
            10,
            status="Ожидание результатов ВИ",
        ),
        _row(1, "Финтех", 3, 100, 5, status="Участвует в конкурсе"),
    ]
    profiles = build_profiles(rows)
    kept = filter_profiles_by_status(profiles)
    assert 1 in kept
    programs = {a.program for a in kept[1].applications}
    assert programs == {"Прикладные модели ИИ", "Финтех"}
    dropped = filter_profiles_by_status(profiles, include_pending=False)
    assert {a.program for a in dropped[1].applications} == {"Финтех"}


def test_markdown_report_shows_pending_program() -> None:
    """Pending-программа видна в паспорте, MC и блоке мест при include_pending."""
    from admission_sim.model import Dataset
    from admission_sim.report import build_markdown_report
    from admission_sim.scenarios import counterfactuals_for_threats

    applied = "Прикладные модели искусственного интеллекта (Москва)"
    rows = [
        _row(1, applied, 1, 0, 115, status="Ожидание результатов ВИ"),
        _row(1, "ММО (Москва)", 2, 0, 50, status="Ожидание результатов ВИ"),
        _row(1, "Финтех (НН)", 3, 100, 5, status="Участвует в конкурсе"),
        _row(2, applied, 1, 90, 1, consent=True),
        _row(3, "Финтех (НН)", 1, 80, 2, consent=True),
    ]
    profiles = filter_profiles_by_status(build_profiles(rows), include_pending=True)
    seats = {applied: 5, "ММО (Москва)": 3, "Финтех (НН)": 15}
    dataset = Dataset(
        applicants=profiles,
        seats=seats,
        programs=[applied, "ММО (Москва)", "Финтех (НН)"],
        source_files=["a.xlsx", "b.xlsx", "c.xlsx"],
        incomplete_priority_codes=[],
    )
    vpp = simulate_enrollment(profiles, seats, consent_only=True)
    vpp_if = what_if_consent(profiles, seats, 1)
    ovp = simulate_enrollment(profiles, seats, consent_only=False)
    prob = estimate_probability(profiles, seats, 1, n_simulations=20, seed=3)
    md = build_markdown_report(
        dataset,
        1,
        vpp=vpp,
        vpp_if_consent=vpp_if,
        ovp=ovp,
        counterfactuals=counterfactuals_for_threats(profiles, seats, 1, limit=5),
        probability=prob,
        include_pending=True,
    )
    assert applied in md
    assert "| Прикладные модели искусственного интеллекта (Москва) |" in md
    assert "#### Прикладные модели искусственного интеллекта (Москва)" in md
    assert "Учитывать pending-статусы: да" in md
    assert "нет приоритетов:" not in md


def test_markdown_warns_when_pending_disabled() -> None:
    from admission_sim.model import Dataset
    from admission_sim.report import build_markdown_report

    rows = [_row(1, "Mine", 1, 100, 1)]
    profiles = build_profiles(rows)
    seats = {"Mine": 1}
    dataset = Dataset(
        applicants=profiles,
        seats=seats,
        programs=["Mine"],
        source_files=["a.xlsx"],
        incomplete_priority_codes=[],
    )
    vpp = simulate_enrollment(profiles, seats, consent_only=True)
    md = build_markdown_report(
        dataset,
        1,
        vpp=vpp,
        vpp_if_consent=vpp,
        ovp=vpp,
        counterfactuals=[],
        probability=None,
        include_pending=False,
    )
    assert "Pending-статусы **не учтены**" in md


def test_read_budget_xlsx_fixture(tmp_path: Path) -> None:
    """Мини-Budget.xlsx: шапка, ~id~, приоритет/сумма/согласие «Б»."""
    from openpyxl import Workbook

    path = tmp_path / "fixture_Budget.xlsx"
    wb = Workbook()
    ws = wb.active
    ws["A1"] = (
        'Список лиц, подавших документы\n'
        'Образовательная программа "Тестовая программа"\n'
        "Направление подготовки 01.04.02\n"
        "Москва"
    )
    headers = [
        "№ п/п",
        "Рег. номер",
        "Уникальный код поступающего",
        "Поступление на места по целевой квоте",
        "Приоритет бюджетного места",
        "Приоритет целевого места",
        "Конкурс портфолио",
        "Сумма конкурсных баллов",
        "Сумма конкурсных баллов в рамках квоты на целевые места",
        "Все оценки положительные",
        "Статус участия в конкурсе",
        "Согласие на зачисление",
    ]
    for col, name in enumerate(headers, start=1):
        ws.cell(7, col, name)
    ws.cell(8, 7, "~123~")
    data_rows = [
        (1, 100, 1111111, "-", 2, None, 100, 100, 100, "+", "Участвует в конкурсе", "Б"),
        (2, 101, 2222222, "-", 1, None, None, 0, 0, "-", "Ожидание результатов ВИ", None),
    ]
    for r_idx, values in enumerate(data_rows, start=9):
        for c_idx, value in enumerate(values, start=1):
            ws.cell(r_idx, c_idx, value)
    wb.save(path)

    bare, campus = program_and_campus_from_budget_xlsx(path)
    assert bare == "Тестовая программа"
    assert campus == "Москва"
    rows = _read_budget_xlsx(path, program_key(bare, campus))
    assert len(rows) == 2
    assert rows[0].applicant_code == 1111111
    assert rows[0].priority == 2
    assert rows[0].score == 100.0
    assert rows[0].rank == 1
    assert rows[0].consent is True
    assert rows[0].status == "Участвует в конкурсе"
    assert rows[1].applicant_code == 2222222
    assert rows[1].priority == 1
    assert rows[1].score == 0.0
    assert rows[1].consent is False
    assert rows[1].status == "Ожидание результатов ВИ"


def test_load_dataset_from_fixtures(tmp_path: Path) -> None:
    csv_a = tmp_path / "ProgA.2026-08-16_12-00-00.csv"
    csv_a.write_text(
        "\n".join(
            [
                '"Порядковый номер";"Приоритет конкурса";"Подано согласие";'
                '"Сумма баллов";"Баллы за ВИ";"Баллы за ИД";"Статус";'
                '"Код поступающего";"Дата выбора конкурсной группы по Москве"',
                '1;1;"Электронное";100;"100";0;"Участвуете в конкурсе";111;"01.01.2026 в 12:00"',
                '2;1;"—";90;"90";0;"Участвуете в конкурсе";222;"01.01.2026 в 12:00"',
            ]
        ),
        encoding="utf-8",
    )
    seats_path = tmp_path / "seats.yaml"
    seats_path.write_text(
        yaml.dump({"programs": {"ProgA": 1}}, allow_unicode=True),
        encoding="utf-8",
    )
    dataset = load_dataset(tmp_path, seats_path)
    assert 111 in dataset.applicants
    assert dataset.applicants[111].consent is True
    assert dataset.seats["ProgA"] == 1
    result = simulate_enrollment(dataset.applicants, dataset.seats)
    assert result.destination(111) == "ProgA"
