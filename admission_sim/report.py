"""Формирование отчёта для терминала и Markdown."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from admission_sim.model import EXTERNAL, Dataset
from admission_sim.scenarios import SCENARIO_LABELS, Counterfactual, ProbabilityEstimate
from admission_sim.simulate import EnrollmentResult


@dataclass(frozen=True, slots=True)
class ProgramFillStats:
    """Сводка заполнения одной программы по результату ВПП."""

    program: str
    seats: int | None
    enrolled: int
    cutoff: float | None
    priority1: int
    fallen: int


def _fmt_dest(dest: str | None) -> str:
    if dest is None:
        return "не зачислен"
    if dest == EXTERNAL:
        return "вне загруженных программ"
    return dest


def my_program_names(dataset: Dataset, my_code: int) -> list[str]:
    """Программы, на которые подал заявки указанный код (по приоритету)."""
    return [app.program for app in dataset.applicants[my_code].sorted_applications()]


def _priority_gap_note(dataset: Dataset, my_code: int) -> str | None:
    """Пояснение, если в данных нет части приоритетов абитуриента."""
    me = dataset.applicants[my_code]
    if not me.missing_higher_priority:
        return None
    priorities = sorted({app.priority for app in me.applications if app.priority < 10**6})
    if not priorities:
        return None
    present = ", ".join(str(p) for p in priorities)
    missing: list[int] = []
    for expected in range(1, max(priorities) + 1):
        if expected not in priorities:
            missing.append(expected)
    if missing:
        miss = ", ".join(str(p) for p in missing)
        return (
            f"В загруженных списках есть приоритеты: {present}; "
            f"нет приоритетов: {miss}. При согласии модель считает уход "
            "на внешнюю (незагруженную) программу."
        )
    return (
        "Неполный граф приоритетов: в данных есть пропуски между заявками. "
        "При согласии возможен уход на внешнюю программу."
    )


def program_fill_stats(
    dataset: Dataset,
    vpp: EnrollmentResult,
    program: str,
) -> ProgramFillStats:
    """Считает заполнение мест и проходной балл для программы."""
    k = dataset.seats.get(program)
    enrolled_codes = vpp.enrolled_for(program)
    scores: list[float] = []
    pri1 = 0
    for code in enrolled_codes:
        profile = dataset.applicants.get(code)
        if profile is None:
            continue
        app = profile.application_for(program)
        if app is None:
            continue
        scores.append(app.score)
        if app.priority == 1:
            pri1 += 1
    return ProgramFillStats(
        program=program,
        seats=k,
        enrolled=len(enrolled_codes),
        cutoff=min(scores) if scores else None,
        priority1=pri1,
        fallen=len(enrolled_codes) - pri1,
    )


def _summarize_other_programs(
    dataset: Dataset,
    vpp: EnrollmentResult,
    skip: set[str],
) -> tuple[int, int, int, int, int]:
    """
    Сводка по программам вне skip.

    Returns:
        (n_other, known_k_programs, total_seats, total_enrolled, unknown_k)
    """
    n_other = 0
    known = 0
    total_seats = 0
    total_enrolled = 0
    unknown = 0
    for program in dataset.programs:
        if program in skip:
            continue
        n_other += 1
        stats = program_fill_stats(dataset, vpp, program)
        total_enrolled += stats.enrolled
        if stats.seats is None:
            unknown += 1
        else:
            known += 1
            total_seats += int(stats.seats)
    return n_other, known, total_seats, total_enrolled, unknown


def _append_fill_block(lines: list[str], stats: ProgramFillStats) -> None:
    k_label = "?" if stats.seats is None else str(stats.seats)
    cutoff = stats.cutoff if stats.cutoff is not None else "—"
    lines.extend(
        [
            f"#### {stats.program}",
            "",
            f"- Мест: **{k_label}**, зачислено по ВПП: **{stats.enrolled}**",
            f"- Проходной балл среди зачисленных: {cutoff}",
            f"- Из них с приоритетом 1: {stats.priority1}, "
            f"«свалились» с других: {stats.fallen}",
            "",
        ]
    )


def build_markdown_report(
    dataset: Dataset,
    my_code: int,
    *,
    vpp: EnrollmentResult,
    vpp_if_consent: EnrollmentResult,
    ovp: EnrollmentResult,
    counterfactuals: list[Counterfactual],
    probability: ProbabilityEstimate | None,
    include_pending: bool,
) -> str:
    """Собирает полный Markdown-отчёт."""
    me = dataset.applicants[my_code]
    my_programs = my_program_names(dataset, my_code)
    my_set = set(my_programs)

    lines: list[str] = [
        "# Отчёт симулятора поступления",
        "",
        "> **Дисклеймер.** Неофициальная модель для личного анализа. "
        "Не является расчётом приёмной комиссии, вуза или Госуслуг "
        "и не гарантирует результат зачисления. Лицензия: MIT.",
        "",
        "## 1. Ваш паспорт",
        "",
        f"- Код поступающего: `{my_code}`",
        f"- Согласие в данных: **{'да' if me.consent else 'нет'}**",
        f"- Неполный граф приоритетов: **{'да' if me.missing_higher_priority else 'нет'}**",
        f"- Учитывать pending-статусы: {'да' if include_pending else 'нет'}",
        "",
        "| Программа | Приоритет | Баллы | Место | Статус |",
        "|---|---:|---:|---:|---|",
    ]
    for app in me.sorted_applications():
        lines.append(
            f"| {app.program} | {app.priority} | {app.score:g} | {app.rank} | {app.status} |"
        )

    gap_note = _priority_gap_note(dataset, my_code)
    if gap_note:
        lines.extend(["", f"> {gap_note}", ""])
    else:
        lines.append("")

    lines.extend(
        [
            "### Сценарии зачисления",
            "",
            f"- **ВПП сейчас:** {_fmt_dest(vpp.destination(my_code))}",
            f"- **ВПП если вы подадите согласие:** {_fmt_dest(vpp_if_consent.destination(my_code))}",
            f"- **ОВП (если согласятся все):** {_fmt_dest(ovp.destination(my_code))}",
            "",
        ]
    )

    if probability is not None:
        lines.extend(
            [
                "### Оценка вероятности (Monte Carlo)",
                "",
                f"- Прогонов: {probability.n_simulations}",
                f"- Сценарий согласий: **{SCENARIO_LABELS.get(probability.scenario, probability.scenario)}**",
                f"- {probability.consent_model_description}",
                f"- Средний p у без согласия: {probability.mean_consent_probability:.0%}",
                f"- P(зачисление на загруженную программу) = **{probability.any_loaded:.1%}**",
                f"- P(уход вовне) = {probability.external:.1%}",
                f"- P(не зачислен) = {probability.none:.1%}",
                "",
                "В таблице — только **ваши** программы: доля прогонов, где вы зачислены "
                "именно на эту программу (в одном прогоне — только одна). "
                "Сумма по строкам ~ P(зачисление на загруженную программу).",
                "",
                "| Программа | P |",
                "|---|---:|",
            ]
        )
        for program in my_programs:
            value = probability.by_program.get(program, 0.0)
            lines.append(f"| {program} | {value:.1%} |")
        lines.append("")

    my_unknown = [p for p in my_programs if dataset.seats.get(p) is None]
    my_zero = [
        p
        for p in my_programs
        if dataset.seats.get(p) is not None and int(dataset.seats[p] or 0) <= 0
    ]
    incomplete_n = len(dataset.incomplete_priority_codes)
    other_unknown = sum(
        1
        for p in dataset.programs
        if p not in my_set and dataset.seats.get(p) is None
    )

    lines.extend(
        [
            "### Предупреждения",
            "",
            f"- Файлов конкурсов: {len(dataset.source_files)}",
            f"- Абитуриентов в модели: {len(dataset.applicants)}",
            f"- С дырами в приоритетах (вне CSV): {incomplete_n}",
            "- При дырах в приоритетах модель считает, что согласившиеся уходят "
            "на внешние программы и не занимают места здесь.",
        ]
    )
    if not include_pending:
        lines.append(
            "- Pending-статусы **не учтены**: заявки только с "
            "«Ожидание результатов ВИ» скрыты из паспорта и модели. "
            "Если не видите свои программы — включите pending "
            "(CLI: уберите `--no-include-pending`; Streamlit: галочка слева)."
        )
    if my_unknown:
        lines.append(
            "- У ваших программ нет КЦП в seats.yaml: " + ", ".join(my_unknown)
        )
    if my_zero:
        lines.append("- У ваших программ нулевые места: " + ", ".join(my_zero))
    if other_unknown:
        lines.append(
            f"- У остальных загруженных конкурсов без КЦП: {other_unknown} "
            "(в сводке ниже не расписываются)."
        )
    lines.extend(["", "## 2. Заполнение мест (ВПП сейчас)", ""])

    lines.append("### Ваши программы")
    lines.append("")
    if not my_programs:
        lines.append("В загруженных списках нет ваших заявок.")
        lines.append("")
    else:
        for program in my_programs:
            _append_fill_block(lines, program_fill_stats(dataset, vpp, program))

    n_other, known, total_seats, total_enrolled, unknown = _summarize_other_programs(
        dataset, vpp, my_set
    )
    if n_other:
        lines.extend(
            [
                "### Остальные программы (сводка)",
                "",
                f"- Других конкурсов в модели: **{n_other}**",
                f"- С известным КЦП: {known} (мест: {total_seats}, "
                f"зачислено по ВПП: {total_enrolled})",
                f"- Без КЦП: {unknown}",
                "",
            ]
        )

    lines.extend(["## 3. Контрфакты конкурентов (угрозы)", ""])
    if not counterfactuals:
        lines.append("Нет кандидатов без согласия выше вас в ваших программах.")
    else:
        lines.extend(
            [
                "| Код | Куда попадёт | Вытесняет вас | Вы до | Вы после |",
                "|---:|---|---|---|---|",
            ]
        )
        for item in counterfactuals:
            lines.append(
                f"| {item.code} | {_fmt_dest(item.destination)} | "
                f"{'да' if item.displaces_me else 'нет'} | "
                f"{_fmt_dest(item.my_destination_before)} | "
                f"{_fmt_dest(item.my_destination_after)} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            f"Источники: {len(dataset.source_files)} файлов "
            f"({', '.join(dataset.source_files[:3])}"
            f"{', …' if len(dataset.source_files) > 3 else ''})",
            "",
        ]
    )
    return "\n".join(lines)


def print_cli_report(
    dataset: Dataset,
    my_code: int,
    *,
    vpp: EnrollmentResult,
    vpp_if_consent: EnrollmentResult,
    ovp: EnrollmentResult,
    counterfactuals: list[Counterfactual],
    probability: ProbabilityEstimate | None,
    include_pending: bool,
    console: Console | None = None,
) -> None:
    """Печатает три экрана отчёта в терминал."""
    console = console or Console()
    me = dataset.applicants[my_code]
    my_programs = my_program_names(dataset, my_code)
    my_set = set(my_programs)

    console.print(
        "[dim]Дисклеймер: неофициально, не ПК/вуз/Госуслуги, "
        "не гарантия зачисления. Лицензия: MIT.[/dim]"
    )
    console.rule("[bold]1. Ваш паспорт[/bold]")
    console.print(
        f"Код [cyan]{my_code}[/cyan] · согласие: "
        f"{'да' if me.consent else 'нет'} · "
        f"неполный граф: {'да' if me.missing_higher_priority else 'нет'} · "
        f"pending: {'да' if include_pending else 'нет'}"
    )
    gap_note = _priority_gap_note(dataset, my_code)
    if gap_note:
        console.print(f"[yellow]{gap_note}[/yellow]")

    apps = Table(show_header=True, header_style="bold")
    apps.add_column("Программа")
    apps.add_column("Приоритет", justify="right")
    apps.add_column("Баллы", justify="right")
    apps.add_column("Место", justify="right")
    apps.add_column("Статус")
    for app in me.sorted_applications():
        apps.add_row(
            app.program,
            str(app.priority),
            f"{app.score:g}",
            str(app.rank),
            app.status,
        )
    console.print(apps)

    scenarios = Table(show_header=True, header_style="bold", title="Сценарии")
    scenarios.add_column("Сценарий")
    scenarios.add_column("Результат")
    scenarios.add_row("ВПП сейчас", _fmt_dest(vpp.destination(my_code)))
    scenarios.add_row("ВПП + ваше согласие", _fmt_dest(vpp_if_consent.destination(my_code)))
    scenarios.add_row("ОВП (все согласны)", _fmt_dest(ovp.destination(my_code)))
    console.print(scenarios)

    if probability is not None:
        prob = Table(show_header=True, header_style="bold", title="Monte Carlo")
        prob.add_column("Исход")
        prob.add_column("P", justify="right")
        prob.add_row("Любая загруженная", f"{probability.any_loaded:.1%}")
        prob.add_row("Вне загруженных", f"{probability.external:.1%}")
        prob.add_row("Не зачислен", f"{probability.none:.1%}")
        for program in my_programs:
            value = probability.by_program.get(program, 0.0)
            prob.add_row(program, f"{value:.1%}")
        console.print(prob)
        console.print(
            f"[dim]N={probability.n_simulations}, "
            f"{SCENARIO_LABELS.get(probability.scenario, probability.scenario)} · "
            f"{probability.consent_model_description}[/dim]"
        )
        console.print(
            "[dim]Таблица P — только ваши программы "
            "(сумма ~ P любой загруженной).[/dim]"
        )

    if dataset.incomplete_priority_codes:
        console.print(
            f"[yellow]Предупреждение:[/yellow] у "
            f"{len(dataset.incomplete_priority_codes)} абитуриентов "
            "приоритеты указывают на программы вне CSV — догрузите списки."
        )
    if not include_pending:
        console.print(
            "[yellow]Pending выключен:[/yellow] заявки только с "
            "«Ожидание результатов ВИ» скрыты. Если не видите свои программы — "
            "оставьте pending включённым (по умолчанию)."
        )

    console.rule("[bold]2. Заполнение мест (ВПП)[/bold]")
    seats_table = Table(show_header=True, header_style="bold", title="Ваши программы")
    seats_table.add_column("Программа")
    seats_table.add_column("K", justify="right")
    seats_table.add_column("Зачислено", justify="right")
    seats_table.add_column("Проходной", justify="right")
    seats_table.add_column("Приоритет 1", justify="right")
    for program in my_programs:
        stats = program_fill_stats(dataset, vpp, program)
        k_label = "?" if stats.seats is None else str(stats.seats)
        cutoff = f"{stats.cutoff:g}" if stats.cutoff is not None else "—"
        seats_table.add_row(
            program,
            k_label,
            str(stats.enrolled),
            cutoff,
            str(stats.priority1),
        )
    console.print(seats_table)

    n_other, known, total_seats, total_enrolled, unknown = _summarize_other_programs(
        dataset, vpp, my_set
    )
    if n_other:
        console.print(
            f"[dim]Остальные конкурсы: {n_other} "
            f"(с КЦП: {known}, мест {total_seats}, зачислено {total_enrolled}; "
            f"без КЦП: {unknown})[/dim]"
        )

    console.rule("[bold]3. Контрфакты (угрозы)[/bold]")
    if not counterfactuals:
        console.print("Нет угроз среди абитуриентов без согласия выше вас.")
        return

    threats = Table(show_header=True, header_style="bold")
    threats.add_column("Код", justify="right")
    threats.add_column("Куда")
    threats.add_column("Вытесняет")
    threats.add_column("Вы до")
    threats.add_column("Вы после")
    for item in counterfactuals:
        threats.add_row(
            str(item.code),
            _fmt_dest(item.destination),
            "да" if item.displaces_me else "нет",
            _fmt_dest(item.my_destination_before),
            _fmt_dest(item.my_destination_after),
        )
    console.print(threats)


def write_markdown_report(path: Path | str, content: str) -> None:
    """Пишет Markdown-отчёт на диск."""
    target = Path(path)
    target.write_text(content, encoding="utf-8")
