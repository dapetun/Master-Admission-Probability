"""Формирование отчёта для терминала и Markdown."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from admission_sim.load import is_pending_status
from admission_sim.model import EXTERNAL, ApplicantProfile, Dataset
from admission_sim.scenarios import SCENARIO_LABELS, Counterfactual, ProbabilityEstimate
from admission_sim.simulate import EnrollmentResult


@dataclass(frozen=True, slots=True)
class PendingAheadRow:
    """Сколько на программе ждут экзамен выше вас (по месту в файле)."""

    program: str
    pending_ahead: int
    pending_on_program: int
    my_rank: int


@dataclass(frozen=True, slots=True)
class ProgramFillStats:
    """Сводка заполнения одной программы по результату ВПП."""

    program: str
    seats: int | None
    enrolled: int
    vacant: int | None
    cutoff: float | None
    priority1_ahead: int
    enrolled_ahead: int
    enrolled_below: int


@dataclass(frozen=True, slots=True)
class ThreatsView:
    """Как показать угрозы: набор строк зависит от того, зачислены ли вы сейчас."""

    title: str
    caption: str | None
    empty_message: str | None
    user_enrolled: bool
    shown: list[Counterfactual]
    displacing_n: int
    total_n: int
    count_caption: str | None = None


SEAT_FILL_CAPTION = (
    "Зачисление только среди уже подавших согласие: кто ещё не согласился, "
    "в зачисление не входит. «Зачислено сейчас» и «свободно» — фактическая "
    "занятость среди уже подавших согласие, в порядке мест. "
    "«Зачислено сейчас» — не «занято выше вас»: в это число входят и люди "
    "ниже вас на оставшихся местах. «Свободно» — места, которые пока никто "
    "с согласием не занял; часть уже занятых ниже вас не мешает вам, "
    "если вы согласитесь. Для вашего шанса смотрите: места − «выше вас с согласием» "
    "(кто выше остался на этой программе с согласием). "
    "Люди ниже вас в «зачислено сейчас» вас не обгоняют. "
    "Счётчики «выше вас» — только те, кто стоит выше в списке; "
    "ожидание вступительных с 0 ниже вас в конкуренты впереди не входит."
)
PENDING_SCORE_CAPTION = (
    "Кто ещё ждёт результаты вступительных, стоит в списке с баллами из файла "
    "(часто 0). Модель не подставляет будущие баллы и не считает, что они "
    "«займут» места пропорционально."
)
MC_SCENARIO_UNCERTAINTY_NOTE = (
    "Погрешность от числа прогонов — это шум самой случайности. "
    "Разброс между сценариями согласий (авто / пессимистичный) обычно больше: "
    "сравнивайте сценарии, а не гонитесь за тысячными долями."
)
UNKNOWN_SEATS_LABEL = "число мест неизвестно"
NO_RANK_THREATS_MESSAGE = (
    "Нет абитуриентов выше вас без согласия на ваших программах"
)
FILTERED_EMPTY_THREATS_MESSAGE = (
    "На выбранных программах в этом списке никого нет"
)
TAKES_OVERLAP_LABEL = "займёт программу пересечения"
LEAVES_OVERLAP_LABEL = "уйдёт на другую / вовне"
THREATS_FILTER_NOTE = (
    "В таблице не весь список выше вас. Только люди без согласия, "
    "которые стоят выше вас хотя бы на одной вашей программе. "
    "Уже согласившиеся и так участвуют в зачислении «сейчас». "
    "Кто выше только на чужих программах, без пересечения с вами, не показан. "
    "Человек выше сразу на нескольких ваших программах попадает в список "
    "по каждой из них."
)
THREATS_LEAVERS_HIDDEN_NOTE = (
    "Скрыты те, кто при согласии ушёл бы на другую программу или вовне."
)
THREATS_WHATIF_NOTE = (
    "Колонка «куда, если согласится» — полный пересчёт зачисления, "
    "если согласие подаст только этот человек (плюс уже согласившиеся). "
    "Это не «гарантия, когда согласие подадут все». "
    "Если при текущих согласиях он проходит на другую программу "
    "с более высоким приоритетом — он уйдёт туда и вашу программу пересечения "
    "не займёт. Если другую программу он занимает только когда согласие "
    "подадут ещё люди — здесь это не считается гарантированным уходом."
)
THREATS_CAPTION_NOT_ENROLLED = (
    "Абитуриенты выше вас без согласия на ваших программах. "
    "Если подадут согласие — могут занять место раньше вас.\n\n"
    f"{THREATS_FILTER_NOTE}\n\n"
    f"{THREATS_LEAVERS_HIDDEN_NOTE}\n\n"
    f"{THREATS_WHATIF_NOTE}"
)
THREATS_ALL_LEAVE_MESSAGE = (
    "никто при согласии не остаётся на программе пересечения "
    "(ушли бы на другую / вовне)."
)
THREATS_SELECT_PROGRAM_CAPTION = "выберите программу"


def filter_counterfactuals_by_overlap(
    items: list[Counterfactual],
    programs: set[str],
) -> list[Counterfactual]:
    """Оставляет контрфакты, у которых программа пересечения в ``programs``."""
    return [item for item in items if item.overlap_program in programs]


def stays_on_overlap_program(item: Counterfactual) -> bool:
    """True, если при согласии только этого человека он занимает программу пересечения."""
    return item.destination == item.overlap_program


def filter_counterfactuals_staying_on_overlap(
    items: list[Counterfactual],
) -> list[Counterfactual]:
    """Оставляет тех, кто при согласии остаётся на программе пересечения."""
    return [item for item in items if stays_on_overlap_program(item)]


def threats_stay_count_caption(shown_n: int, total_n: int) -> str:
    """Подпись: видимые строки vs все выше без согласия на выбранных программах."""
    return (
        f"В таблице {shown_n} из {total_n} человек выше вас без согласия, "
        "которые при согласии остаются на этой программе. "
        f"Случайные прогоны учитывают всех {total_n} "
        "(и каскад других конкурсов), не только видимые строки."
    )


def _fmt_dest(dest: str | None) -> str:
    if dest is None:
        return "не зачислен"
    if dest == EXTERNAL:
        return "вне загруженных программ"
    return dest


def mc_standard_error(proportion: float, n_simulations: int) -> float:
    """Стандартная ошибка оценки доли по N прогонам Monte Carlo."""
    if n_simulations <= 0:
        return 0.0
    p = min(max(proportion, 0.0), 1.0)
    return (p * (1.0 - p) / n_simulations) ** 0.5


def mc_ci95_note(probability: ProbabilityEstimate) -> str:
    """Краткая пометка о погрешности доли зачисления на загруженную программу."""
    se = mc_standard_error(probability.any_loaded, probability.n_simulations)
    half = 2.0 * se
    return (
        f"Погрешность доли зачисления (примерно 95%): ±{half:.1%} "
        f"при {probability.n_simulations} прогонах"
    )


def my_program_names(dataset: Dataset, my_code: int) -> list[str]:
    """Программы, на которые подал заявки указанный код (по приоритету)."""
    return [app.program for app in dataset.applicants[my_code].sorted_applications()]


def pending_ahead_rows(
    applicants: dict[int, ApplicantProfile],
    my_code: int,
) -> list[PendingAheadRow]:
    """Для каждой программы пользователя: сколько ждут экзамен выше по месту."""
    me = applicants[my_code]
    rows: list[PendingAheadRow] = []
    for app in me.sorted_applications():
        ahead = 0
        total = 0
        for code, profile in applicants.items():
            if code == my_code:
                continue
            other = profile.application_for(app.program)
            if other is None or not is_pending_status(other.status):
                continue
            total += 1
            if other.rank < app.rank:
                ahead += 1
        rows.append(
            PendingAheadRow(
                program=app.program,
                pending_ahead=ahead,
                pending_on_program=total,
                my_rank=app.rank,
            )
        )
    return rows


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
    my_code: int,
) -> ProgramFillStats:
    """Считает заполнение мест и персональные счётчики относительно пользователя."""
    k = dataset.seats.get(program)
    enrolled_codes = vpp.enrolled_for(program)
    me = dataset.applicants.get(my_code)
    my_app = me.application_for(program) if me is not None else None
    my_rank = my_app.rank if my_app is not None else None

    scores: list[float] = []
    enrolled_ahead = 0
    enrolled_below = 0
    for code in enrolled_codes:
        profile = dataset.applicants.get(code)
        if profile is None:
            continue
        app = profile.application_for(program)
        if app is None:
            continue
        scores.append(app.score)
        if my_rank is None or code == my_code:
            continue
        if app.rank < my_rank:
            enrolled_ahead += 1
        elif app.rank > my_rank:
            enrolled_below += 1

    priority1_ahead = 0
    if my_rank is not None:
        for code, profile in dataset.applicants.items():
            if code == my_code or not profile.consent:
                continue
            app = profile.application_for(program)
            if app is None or app.priority != 1:
                continue
            if app.rank < my_rank:
                priority1_ahead += 1

    n_enrolled = len(enrolled_codes)
    vacant = None if k is None else max(0, int(k) - n_enrolled)
    return ProgramFillStats(
        program=program,
        seats=k,
        enrolled=n_enrolled,
        vacant=vacant,
        cutoff=min(scores) if scores else None,
        priority1_ahead=priority1_ahead,
        enrolled_ahead=enrolled_ahead,
        enrolled_below=enrolled_below,
    )


def format_seats(seats: int | None) -> str:
    """Число бюджетных мест или пометка, что оно неизвестно."""
    if seats is None:
        return UNKNOWN_SEATS_LABEL
    return str(seats)


def format_vacant(vacant: int | None) -> str:
    """Свободные места; «—», если число мест неизвестно."""
    if vacant is None:
        return "—"
    return str(vacant)


def format_cutoff(cutoff: float | None) -> str:
    """Проходной балл или тире, если никто не зачислен."""
    if cutoff is None:
        return "—"
    return f"{cutoff:g}"


def other_programs_note(n_other: int) -> str | None:
    """Короткая подпись про конкурсы вне ваших программ."""
    if n_other <= 0:
        return None
    return f"Ещё {n_other} конкурсов вне ваших программ — в таблице не показаны."


def seat_fill_rows(stats_list: list[ProgramFillStats]) -> list[dict[str, object]]:
    """Строки таблицы заполнения мест (понятные русские колонки)."""
    rows: list[dict[str, object]] = []
    for stats in stats_list:
        rows.append(
            {
                "Программа": stats.program,
                "Места": format_seats(stats.seats),
                "Зачислено сейчас": stats.enrolled,
                "Свободно": format_vacant(stats.vacant),
                "Проходной балл": format_cutoff(stats.cutoff),
                "Выше вас с согласием": stats.enrolled_ahead,
                "Выше вас, 1-й приоритет": stats.priority1_ahead,
                "Ниже вас зачислены": stats.enrolled_below,
            }
        )
    return rows


def vpp_user_enrolled(dest: str | None) -> bool:
    """True, если сейчас зачислены на загруженную программу (не EXTERNAL)."""
    return dest is not None and dest != EXTERNAL


def prepare_threats_view(
    counterfactuals: list[Counterfactual],
    my_vpp_dest: str | None,
    *,
    filtered: bool = False,
) -> ThreatsView:
    """
    Фильтрует контрфакты для показа.

    Если вы не зачислены, «вытесняет» всегда нет — показываем тех,
    кто при согласии остаётся на программе пересечения.
    Если зачислены — только тех, кто реально вытесняет, плюс счётчик N из M.
    ``filtered`` — пустой набор после фильтра программ, а не «угроз нет вообще».
    """
    enrolled = vpp_user_enrolled(my_vpp_dest)
    total = len(counterfactuals)
    displacing_n = sum(1 for item in counterfactuals if item.displaces_me)

    if total == 0:
        empty = (
            FILTERED_EMPTY_THREATS_MESSAGE if filtered else NO_RANK_THREATS_MESSAGE
        )
        return ThreatsView(
            title="Угрозы",
            caption=None,
            empty_message=empty,
            user_enrolled=enrolled,
            shown=[],
            displacing_n=0,
            total_n=0,
        )

    if enrolled:
        shown = [item for item in counterfactuals if item.displaces_me]
        if shown:
            return ThreatsView(
                title="Контрфакты (угрозы)",
                caption=(
                    f"{displacing_n} из {total} вытесняют вас, "
                    f"если подадут согласие.\n\n{THREATS_FILTER_NOTE}\n\n"
                    f"{THREATS_WHATIF_NOTE}"
                ),
                empty_message=None,
                user_enrolled=True,
                shown=shown,
                displacing_n=displacing_n,
                total_n=total,
            )
        return ThreatsView(
            title="Контрфакты (угрозы)",
            caption=None,
            empty_message=(
                f"Из {total} абитуриентов выше вас без согласия "
                "никто не вытесняет вас с текущего места."
            ),
            user_enrolled=True,
            shown=[],
            displacing_n=0,
            total_n=total,
        )

    shown = filter_counterfactuals_staying_on_overlap(counterfactuals)
    count_caption = threats_stay_count_caption(len(shown), total)
    if shown:
        return ThreatsView(
            title="Абитуриенты выше вас без согласия",
            caption=THREATS_CAPTION_NOT_ENROLLED,
            empty_message=None,
            user_enrolled=False,
            shown=shown,
            displacing_n=displacing_n,
            total_n=total,
            count_caption=count_caption,
        )
    return ThreatsView(
        title="Абитуриенты выше вас без согласия",
        caption=THREATS_CAPTION_NOT_ENROLLED,
        empty_message=f"Из {total} абитуриентов выше вас без согласия {THREATS_ALL_LEAVE_MESSAGE}",
        user_enrolled=False,
        shown=[],
        displacing_n=displacing_n,
        total_n=total,
        count_caption=count_caption,
    )


def threat_if_consents_effect(item: Counterfactual) -> str:
    """Куда попадёт человек, если согласие подаст только он."""
    dest = item.destination
    if dest is None:
        return _fmt_dest(None)
    if dest == item.overlap_program:
        return TAKES_OVERLAP_LABEL
    return LEAVES_OVERLAP_LABEL


def threat_table_rows(view: ThreatsView) -> list[dict[str, object]]:
    """Строки таблицы угроз: колонки зависят от того, зачислены ли вы."""
    if view.user_enrolled:
        return [
            {
                "Код": item.code,
                "Программа пересечения": item.overlap_program,
                "Их место": item.their_rank,
                "Ваше место": item.my_rank,
                "Вытесняет": "да" if item.displaces_me else "нет",
                "Если согласится": threat_if_consents_effect(item),
                "Куда (если согласится)": _fmt_dest(item.destination),
                "Вы до": _fmt_dest(item.my_destination_before),
                "Вы после": _fmt_dest(item.my_destination_after),
            }
            for item in view.shown
        ]
    return [
        {
            "Код": item.code,
            "Программа пересечения": item.overlap_program,
            "Их место": item.their_rank,
            "Ваше место": item.my_rank,
            "Разрыв": item.gap,
            "Если согласится": threat_if_consents_effect(item),
            "Куда (если согласится)": _fmt_dest(item.destination),
        }
        for item in view.shown
    ]


def _append_fill_block(lines: list[str], stats: ProgramFillStats) -> None:
    cutoff = format_cutoff(stats.cutoff)
    lines.extend(
        [
            f"#### {stats.program}",
            "",
            f"- Мест: **{format_seats(stats.seats)}**, "
            f"зачислено сейчас: **{stats.enrolled}**, "
            f"свободно: **{format_vacant(stats.vacant)}**",
            f"- Проходной балл среди зачисленных: {cutoff}",
            f"- Выше вас с этой программой 1-м приоритетом и согласием: "
            f"{stats.priority1_ahead}",
            f"- Выше вас остались на этой программе с согласием: "
            f"{stats.enrolled_ahead}",
            f"- Из зачисленных сейчас ниже вас: {stats.enrolled_below}",
            "",
        ]
    )


_NUMERIC_TABLE_KEYS = frozenset(
    {"Код", "Их место", "Ваше место", "Разрыв", "Зачислено сейчас"}
)


def _markdown_from_rows(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return []
    keys = list(rows[0].keys())
    header = "| " + " | ".join(keys) + " |"
    sep = "|" + "|".join(
        "---:" if key in _NUMERIC_TABLE_KEYS else "---" for key in keys
    ) + "|"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in keys) + " |")
    return lines


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
        f"- В загруженных списках нет части приоритетов: "
        f"**{'да' if me.missing_higher_priority else 'нет'}**",
        f"- Учитывать ожидание результатов вступительных испытаний: "
        f"{'да' if include_pending else 'нет'}",
        "",
        "| Программа | Приоритет | Баллы | Место | Статус | Выше вас ждут экзамен |",
        "|---|---:|---:|---:|---|---:|",
    ]
    pending_rows = {
        row.program: row
        for row in pending_ahead_rows(dataset.applicants, my_code)
    }
    for app in me.sorted_applications():
        ahead = pending_rows.get(app.program)
        ahead_cell = (
            f"{ahead.pending_ahead} из {ahead.pending_on_program}"
            if ahead is not None
            else "—"
        )
        lines.append(
            f"| {app.program} | {app.priority} | {app.score:g} | {app.rank} | "
            f"{app.status} | {ahead_cell} |"
        )

    if include_pending:
        lines.extend(["", f"> {PENDING_SCORE_CAPTION}", ""])
    else:
        lines.append("")

    gap_note = _priority_gap_note(dataset, my_code)
    if gap_note:
        lines.extend(["", f"> {gap_note}", ""])
    else:
        lines.append("")

    lines.extend(
        [
            "### Сценарии зачисления",
            "",
            "Зачисление только среди уже подавших согласие — снимок «как сейчас»: "
            "кто ещё не согласился, в зачисление не входит.",
            "",
            f"- **Сейчас, с текущими согласиями:** {_fmt_dest(vpp.destination(my_code))}",
            f"- **Если вы подадите согласие:** {_fmt_dest(vpp_if_consent.destination(my_code))}",
            f"- **Если согласие подадут все:** {_fmt_dest(ovp.destination(my_code))} "
            "— крайний случай: согласие как будто у всех в загруженных списках; "
            "если здесь вы зачислены, место есть даже когда соглашается каждый",
            "",
        ]
    )

    if probability is not None:
        lines.extend(
            [
                "### Случайные прогоны",
                "",
                "Много раз случайно решаем, кто из неопределившихся подаст согласие. "
                "Это не крайний случай «согласие подадут все»: здесь не предполагается, "
                "что согласятся все.",
                "",
                f"- Прогонов: {probability.n_simulations}",
                f"- Сценарий согласий: **{SCENARIO_LABELS.get(probability.scenario, probability.scenario)}**",
                f"- {probability.consent_model_description}",
                f"- Средняя вероятность согласия у тех, кто ещё не согласился: "
                f"{probability.mean_consent_probability:.0%} "
                "(не 100%: «согласие подадут все» — только крайний случай выше)",
                f"- Шанс зачисления на загруженную программу = **{probability.any_loaded:.1%}**",
                f"- {mc_ci95_note(probability)}",
                f"- {MC_SCENARIO_UNCERTAINTY_NOTE}",
                f"- Шанс ухода вне загруженных программ = {probability.external:.1%}",
                f"- Шанс без зачисления = {probability.none:.1%}",
                "",
            ]
        )
        if probability.focus_program:
            lines.extend(
                [
                    f"Случайные прогоны считались для программы "
                    f"**{probability.focus_program}**. "
                    "В расчёт входят и конкурсы с более высоким приоритетом, "
                    "куда могут уйти конкуренты.",
                    "",
                ]
            )
        lines.extend(
            [
                "В таблице — только **ваши** программы, которые попали в этот расчёт: "
                "доля прогонов, где вы зачислены именно на эту программу "
                "(в одном прогоне — только одна). "
                "Сумма по строкам ≈ шанс зачисления на загруженную программу.",
                "",
                "| Программа | Доля прогонов |",
                "|---|---:|",
            ]
        )
        for program, value in probability.by_program.items():
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
            "- Заявки со статусом «ожидание результатов вступительных испытаний» "
            "**не учтены**: они скрыты из паспорта и модели. "
            "Если не видите свои программы — включите эту опцию "
            "(в командной строке уберите `--no-include-pending`; "
            "в приложении — галочка слева)."
        )
    if my_unknown:
        lines.append(
            "- У ваших программ нет числа бюджетных мест в seats.yaml: "
            + ", ".join(my_unknown)
        )
    if my_zero:
        lines.append("- У ваших программ нулевые места: " + ", ".join(my_zero))
    if other_unknown:
        lines.append(
            f"- У остальных загруженных конкурсов без числа бюджетных мест: "
            f"{other_unknown} (в таблице не показаны)."
        )
    lines.extend(["", "## 2. Заполнение мест (сейчас, с текущими согласиями)", ""])
    lines.append(SEAT_FILL_CAPTION)
    lines.append("")
    lines.append("### Ваши программы")
    lines.append("")
    if not my_programs:
        lines.append("В загруженных списках нет ваших заявок.")
        lines.append("")
    else:
        for program in my_programs:
            _append_fill_block(
                lines, program_fill_stats(dataset, vpp, program, my_code)
            )

    n_other = sum(1 for p in dataset.programs if p not in my_set)
    if n_other:
        lines.extend(
            [
                "",
                f"Ещё **{n_other}** конкурсов вне ваших программ — в таблице не показаны.",
                "",
            ]
        )

    threats_view = prepare_threats_view(counterfactuals, vpp.destination(my_code))
    lines.extend(["## 3. " + threats_view.title, ""])
    if threats_view.caption:
        lines.append(threats_view.caption)
        lines.append("")
    if threats_view.empty_message:
        lines.append(threats_view.empty_message)
    else:
        lines.extend(_markdown_from_rows(threat_table_rows(threats_view)))
    if threats_view.count_caption:
        lines.extend(["", threats_view.count_caption])

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
        "[dim]Дисклеймер: неофициально, не приёмная комиссия / вуз / Госуслуги, "
        "не гарантия зачисления. Лицензия: MIT.[/dim]"
    )
    console.rule("[bold]1. Ваш паспорт[/bold]")
    console.print(
        f"Код [cyan]{my_code}[/cyan] · согласие: "
        f"{'да' if me.consent else 'нет'} · "
        f"нет части приоритетов: {'да' if me.missing_higher_priority else 'нет'} · "
        f"ожидание вступительных испытаний: {'да' if include_pending else 'нет'}"
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
    apps.add_column("Выше вас ждут экзамен", justify="right")
    pending_rows = {
        row.program: row
        for row in pending_ahead_rows(dataset.applicants, my_code)
    }
    for app in me.sorted_applications():
        ahead = pending_rows.get(app.program)
        ahead_cell = (
            f"{ahead.pending_ahead} из {ahead.pending_on_program}"
            if ahead is not None
            else "—"
        )
        apps.add_row(
            app.program,
            str(app.priority),
            f"{app.score:g}",
            str(app.rank),
            app.status,
            ahead_cell,
        )
    console.print(apps)
    if include_pending:
        console.print(f"[dim]{PENDING_SCORE_CAPTION}[/dim]")

    scenarios = Table(show_header=True, header_style="bold", title="Сценарии")
    scenarios.add_column("Сценарий")
    scenarios.add_column("Результат")
    scenarios.add_row(
        "Сейчас, с текущими согласиями",
        _fmt_dest(vpp.destination(my_code)),
    )
    scenarios.add_row(
        "Если вы подадите согласие",
        _fmt_dest(vpp_if_consent.destination(my_code)),
    )
    scenarios.add_row(
        "Если согласие подадут все",
        _fmt_dest(ovp.destination(my_code))
        + " · крайний случай, не случайные прогоны",
    )
    console.print(scenarios)
    console.print(
        "[dim]Сейчас — зачисление только среди уже подавших согласие. "
        "«Если согласие подадут все» — крайний случай полной конкуренции.[/dim]"
    )

    if probability is not None:
        prob = Table(
            show_header=True,
            header_style="bold",
            title="Случайные прогоны",
        )
        prob.add_column("Исход")
        prob.add_column("Доля", justify="right")
        prob.add_row("Любая загруженная", f"{probability.any_loaded:.1%}")
        prob.add_row("Вне загруженных", f"{probability.external:.1%}")
        prob.add_row("Не зачислен", f"{probability.none:.1%}")
        for program, value in probability.by_program.items():
            prob.add_row(program, f"{value:.1%}")
        console.print(prob)
        console.print(
            f"[dim]{probability.n_simulations} прогонов, "
            f"{SCENARIO_LABELS.get(probability.scenario, probability.scenario)} · "
            f"{probability.consent_model_description} · "
            f"{mc_ci95_note(probability)} · "
            f"{MC_SCENARIO_UNCERTAINTY_NOTE}[/dim]"
        )
        focus_note = ""
        if probability.focus_program:
            focus_note = (
                f" Расчёт для «{probability.focus_program}» "
                "(плюс конкурсы выше по приоритету у конкурентов)."
            )
        console.print(
            "[dim]Доли — только ваши программы в этом расчёте "
            "(сумма ≈ шанс любой загруженной). "
            "Много раз случайно решаем, кто из неопределившихся подаст согласие."
            f"{focus_note}[/dim]"
        )

    if dataset.incomplete_priority_codes:
        console.print(
            f"[yellow]Предупреждение:[/yellow] у "
            f"{len(dataset.incomplete_priority_codes)} абитуриентов "
            "приоритеты указывают на программы вне CSV — догрузите списки."
        )
    if not include_pending:
        console.print(
            "[yellow]Ожидание вступительных испытаний выключено:[/yellow] "
            "заявки только с этим статусом скрыты. Если не видите свои программы — "
            "оставьте опцию включённой (по умолчанию)."
        )

    console.rule("[bold]2. Заполнение мест (сейчас, с текущими согласиями)[/bold]")
    console.print(f"[dim]{SEAT_FILL_CAPTION}[/dim]")
    seats_table = Table(show_header=True, header_style="bold", title="Ваши программы")
    fill_stats = [
        program_fill_stats(dataset, vpp, program, my_code)
        for program in my_programs
    ]
    fill_rows = seat_fill_rows(fill_stats)
    if fill_rows:
        keys = list(fill_rows[0].keys())
        for key in keys:
            justify = "left" if key == "Программа" else "right"
            seats_table.add_column(str(key), justify=justify)
        for row in fill_rows:
            seats_table.add_row(*[str(row[key]) for key in keys])
    console.print(seats_table)

    n_other = sum(1 for p in dataset.programs if p not in my_set)
    other_note = other_programs_note(n_other)
    if other_note:
        console.print(f"[dim]{other_note}[/dim]")

    threats_view = prepare_threats_view(counterfactuals, vpp.destination(my_code))
    console.rule(f"[bold]3. {threats_view.title}[/bold]")
    if threats_view.caption:
        console.print(f"[dim]{threats_view.caption}[/dim]")
    if threats_view.empty_message:
        console.print(threats_view.empty_message)
        if threats_view.count_caption:
            console.print(f"[dim]{threats_view.count_caption}[/dim]")
        return

    threat_rows = threat_table_rows(threats_view)
    threats = Table(show_header=True, header_style="bold")
    if threat_rows:
        for key in threat_rows[0]:
            justify = "right" if key in _NUMERIC_TABLE_KEYS else "left"
            threats.add_column(str(key), justify=justify)
        for row in threat_rows:
            threats.add_row(*[str(value) for value in row.values()])
        console.print(threats)
    if threats_view.count_caption:
        console.print(f"[dim]{threats_view.count_caption}[/dim]")


def write_markdown_report(path: Path | str, content: str) -> None:
    """Пишет Markdown-отчёт на диск."""
    target = Path(path)
    target.write_text(content, encoding="utf-8")
