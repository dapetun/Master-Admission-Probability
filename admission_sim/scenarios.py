"""Сценарии what-if и оценка вероятности (Monte Carlo)."""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal

from admission_sim.load import profile_has_known_score
from admission_sim.model import EXTERNAL, ApplicantProfile
from admission_sim.simulate import (
    EnrollmentResult,
    resolve_focus_program,
    simulate_enrollment,
    subgraph_for_program,
    subgraph_for_programs,
)

ConsentScenario = Literal["auto", "balanced", "optimistic", "pessimistic"]

SCENARIO_LABELS: dict[ConsentScenario, str] = {
    "auto": "Авто (из данных)",
    "balanced": "Сбалансированный",
    "optimistic": "Оптимистичный (для вас)",
    "pessimistic": "Пессимистичный (для вас)",
}

# «Все согласятся» — только ОВП. Для MC без согласия не подставляем 1.0.
# Приоры: если в группе нет вариации (все уже с согласием), эмпирика молчит.
# None — без потолка строк на программу (иначе таблица молча обрезает угроз).
DEFAULT_THREATS_PER_PROGRAM: int | None = None
UNDECIDED_PRIOR_OVERALL = 0.30
UNDECIDED_PRIOR_COMPETITIVE = 0.50
UNDECIDED_PRIOR_OTHER = 0.20
PESSIMISTIC_COMPETITIVE_FLOOR = 0.85
PESSIMISTIC_UNDECIDED_CAP = 0.95
# Трети конкурентов ОВП: меньше — доля шумная, берём общую p_comp.
MIN_CONSENT_BAND_N = 15
CONSENT_BAND_LABELS = ("верх списка", "середина", "низ списка")


@dataclass(frozen=True, slots=True)
class Counterfactual:
    """Результат «если только этот код подаст согласие»."""

    code: int
    destination: str | None
    displaces_me: bool
    my_destination_before: str | None
    my_destination_after: str | None
    overlap_program: str
    their_rank: int
    my_rank: int
    gap: int


@dataclass(frozen=True, slots=True)
class ConsentBand:
    """Доля уже подавших согласие в трети конкурентов ОВП."""

    label: str
    n: int
    consented: int
    rate: float
    n_undecided: int


@dataclass(frozen=True, slots=True)
class ConsentModel:
    """Индивидуальные P(согласие) и сводка для отчёта."""

    by_code: dict[int, float]
    overall_rate: float
    competitive_rate: float
    noncompetitive_rate: float
    mean_undecided: float
    description: str
    scored_n: int = 0
    pending_unknown_n: int = 0
    bands: tuple[ConsentBand, ...] = ()


@dataclass(frozen=True, slots=True)
class ProbabilityEstimate:
    """Доли прогонов Monte Carlo по программам и суммарно."""

    n_simulations: int
    by_program: dict[str, float]
    any_loaded: float
    external: float
    none: float
    mean_consent_probability: float
    consent_model_description: str
    scenario: ConsentScenario
    focus_program: str | None = None


def what_if_consent(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    applicant_code: int,
) -> EnrollmentResult:
    """Пересчитывает ВПП в предположении, что указанный код подал согласие."""
    if applicant_code not in applicants:
        raise KeyError(f"Код {applicant_code} отсутствует в загруженных списках")
    return simulate_enrollment(
        applicants,
        seats,
        consent_only=True,
        extra_consent=frozenset({applicant_code}),
    )


def _displaces_me(before: str | None, after: str | None) -> bool:
    """Вытеснение: раньше место на загруженной программе, после — нет."""
    if before is not None and before != EXTERNAL and (after is None or after == EXTERNAL):
        return True
    return before != after and before is not None and before != EXTERNAL


def counterfactuals_for_threats(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    my_code: int,
    *,
    limit: int | None = DEFAULT_THREATS_PER_PROGRAM,
) -> list[Counterfactual]:
    """
    Контрфакты для абитуриентов без согласия, которые выше вас
    хотя бы в одной вашей программе.

    Не весь список выше вас: уже согласившиеся и так в зачислении
    «сейчас»; люди без пересечения программ не входят. На каждую
    вашу программу — все такие люди (отдельная строка, даже если
    человек выше сразу на нескольких). ``limit`` — необязательный
    потолок с наибольшим разрывом по месту; ``None`` — без потолка.
    What-if считается один раз на код, на подграфе ваших программ
    (те же конкурсы, что влияют на ваше зачисление).
    Это не крайний случай «согласятся все».
    """
    if my_code not in applicants:
        raise KeyError(f"Код {my_code} отсутствует в загруженных списках")

    me = applicants[my_code]
    my_rank_by_program = {app.program: app.rank for app in me.applications}
    my_program_order = [app.program for app in me.sorted_applications()]
    sim_applicants, sim_seats = applicants, seats
    if my_program_order:
        try:
            sim_applicants, sim_seats = subgraph_for_programs(
                applicants, seats, my_program_order
            )
        except KeyError:
            sim_applicants, sim_seats = applicants, seats
        if my_code not in sim_applicants:
            sim_applicants, sim_seats = applicants, seats
    baseline = simulate_enrollment(sim_applicants, sim_seats, consent_only=True)
    my_before = baseline.destination(my_code)

    # программа → (разрыв, их место, ваше место, код)
    by_program: dict[str, list[tuple[int, int, int, int]]] = {
        program: [] for program in my_program_order
    }
    for code, profile in applicants.items():
        if code == my_code or profile.consent:
            continue
        for app in profile.applications:
            my_rank = my_rank_by_program.get(app.program)
            if my_rank is None or app.rank >= my_rank:
                continue
            by_program[app.program].append(
                (my_rank - app.rank, app.rank, my_rank, code)
            )

    selected: list[tuple[str, int, int, int, int]] = []
    for program in my_program_order:
        rows = by_program[program]
        rows.sort(key=lambda row: (-row[0], row[3]))
        picked = rows if limit is None else rows[: max(0, limit)]
        for gap, their_rank, my_rank, code in picked:
            selected.append((program, gap, their_rank, my_rank, code))

    whatifs: dict[int, EnrollmentResult] = {}
    for code in dict.fromkeys(item[4] for item in selected):
        if code in sim_applicants:
            whatifs[code] = what_if_consent(sim_applicants, sim_seats, code)
        else:
            whatifs[code] = what_if_consent(applicants, seats, code)

    results: list[Counterfactual] = []
    for program, gap, their_rank, my_rank, code in selected:
        result = whatifs[code]
        my_after = result.destination(my_code)
        dest = result.destination(code)
        results.append(
            Counterfactual(
                code=code,
                destination=dest,
                displaces_me=_displaces_me(my_before, my_after),
                my_destination_before=my_before,
                my_destination_after=my_after,
                overlap_program=program,
                their_rank=their_rank,
                my_rank=my_rank,
                gap=gap,
            )
        )
    return results


def _rate(codes: list[int], applicants: dict[int, ApplicantProfile]) -> float | None:
    if not codes:
        return None
    return sum(1 for c in codes if applicants[c].consent) / len(codes)


def _undecided_rate(observed: float | None, prior: float) -> float:
    """P для неопределившихся: эмпирическая доля 1.0 не переносится."""
    if observed is None or observed >= 1.0:
        return prior
    return float(observed)


def _pessimistic_p(base: float, *, competitor: bool, overall: float) -> float:
    """Выше auto для конкурентов, но строго < 1.0 у неопределившихся."""
    cap = PESSIMISTIC_UNDECIDED_CAP
    if competitor:
        bumped = max(base * 1.5, PESSIMISTIC_COMPETITIVE_FLOOR)
    else:
        floor = overall if overall < 1.0 else UNDECIDED_PRIOR_OVERALL
        bumped = max(base * 1.2, floor)
    bumped = min(cap, bumped)
    if competitor and bumped <= base:
        bumped = min(cap, (base + cap) / 2)
    return float(bumped)


def _rank_percentiles_on_programs(
    scored: dict[int, ApplicantProfile],
) -> dict[tuple[int, str], float]:
    """0 — лучшее место среди людей с известными баллами на программе."""
    by_program: dict[str, list[tuple[int, int]]] = {}
    for code, profile in scored.items():
        for app in profile.applications:
            by_program.setdefault(app.program, []).append((app.rank, code))
    out: dict[tuple[int, str], float] = {}
    for program, rows in by_program.items():
        rows.sort()
        n = len(rows)
        denom = n - 1
        for index, (_rank, code) in enumerate(rows):
            out[(code, program)] = (index / denom) if denom else 0.0
    return out


def _is_pri1_on_dest(profile: ApplicantProfile, dest: str) -> bool:
    app = profile.application_for(dest)
    return app is not None and app.priority == 1


def _competitive_band_probs(
    competitive: list[int],
    scored: dict[int, ApplicantProfile],
    ovp_dest: dict[int, str | None],
    p_comp: float,
) -> tuple[tuple[ConsentBand, ...], dict[int, float]]:
    """Трети конкурентов по месту; при малом n — общая p_comp."""
    if not competitive:
        return (), {}

    percentiles = _rank_percentiles_on_programs(scored)
    keyed: list[tuple[float, int]] = []
    missing: list[int] = []
    for code in competitive:
        dest = ovp_dest.get(code)
        if dest is None or dest == EXTERNAL:
            missing.append(code)
            continue
        pct = percentiles.get((code, dest))
        if pct is None:
            missing.append(code)
            continue
        keyed.append((pct, code))

    keyed.sort()
    n_keyed = len(keyed)
    band_codes: list[list[int]] = [[], [], []]
    for index, (_pct, code) in enumerate(keyed):
        band = min(2, (index * 3) // n_keyed) if n_keyed else 0
        band_codes[band].append(code)

    bands: list[ConsentBand] = []
    band_p: list[float] = []
    for codes in band_codes:
        raw = _rate(codes, scored)
        consented = sum(1 for c in codes if scored[c].consent)
        p = (
            p_comp
            if len(codes) < MIN_CONSENT_BAND_N
            else _undecided_rate(raw, UNDECIDED_PRIOR_COMPETITIVE)
        )
        bands.append(
            ConsentBand(
                label=CONSENT_BAND_LABELS[len(bands)],
                n=len(codes),
                consented=consented,
                rate=float(raw if raw is not None else 0.0),
                n_undecided=len(codes) - consented,
            )
        )
        band_p.append(p)

    assigned: dict[int, float] = {code: p_comp for code in missing}
    for band_i, codes in enumerate(band_codes):
        base = band_p[band_i]
        pri1 = [
            c
            for c in codes
            if (dest := ovp_dest.get(c))
            and dest != EXTERNAL
            and _is_pri1_on_dest(scored[c], dest)
        ]
        rest = [c for c in codes if c not in set(pri1)]
        can_split = (
            len(codes) >= MIN_CONSENT_BAND_N
            and len(pri1) >= MIN_CONSENT_BAND_N
            and len(rest) >= MIN_CONSENT_BAND_N
        )
        if can_split:
            p1 = _undecided_rate(_rate(pri1, scored), UNDECIDED_PRIOR_COMPETITIVE)
            p_rest = _undecided_rate(
                _rate(rest, scored), UNDECIDED_PRIOR_COMPETITIVE
            )
            for c in pri1:
                assigned[c] = p1
            for c in rest:
                assigned[c] = p_rest
        else:
            for c in codes:
                assigned[c] = base
    return tuple(bands), assigned


def estimate_consent_model(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    *,
    scenario: ConsentScenario = "auto",
) -> ConsentModel:
    """
    Оценивает P(согласие) по текущим спискам.

    Уже согласившиеся всегда p=1. У неопределившихся p<1: доля 100%
    в снимке не копируется (это только ОВП).

    Доли считаются только по людям с известными баллами: ожидание
    вступительных с 0 в файле не размывает эмпирику и не делает
    человека «конкурентом» из-за свободных мест.

    Среди конкурентов ОВП «Авто» берёт долю согласия в трети списка
    (по месту на программе назначения ОВП). Маленькая треть (<15)
    откатывается к общей доле конкурентов. Приоритет 1 внутри трети —
    только если обе ячейки достаточно большие.

    scenario:
      auto — индивидуально из долей (конкурентные по трети / внешние / прочие);
      balanced — одна общая доля overall (не 100%, если это артефакт снимка);
      optimistic — конкуренты реже соглашаются (лучше для вас);
      pessimistic — конкуренты чаще, но не 100%.
    """
    if not applicants:
        return ConsentModel(
            by_code={},
            overall_rate=0.0,
            competitive_rate=0.0,
            noncompetitive_rate=0.0,
            mean_undecided=0.0,
            description="нет данных",
        )

    scored = {
        code: profile
        for code, profile in applicants.items()
        if profile_has_known_score(profile)
    }
    pending_unknown_n = len(applicants) - len(scored)

    if not scored:
        overall = UNDECIDED_PRIOR_OVERALL
        by_code = {
            code: (1.0 if profile.consent else float(overall))
            for code, profile in applicants.items()
        }
        undecided = [c for c, p in applicants.items() if not p.consent]
        mean_und = (
            sum(by_code[c] for c in undecided) / len(undecided) if undecided else 0.0
        )
        label = SCENARIO_LABELS.get(scenario, scenario)
        description = (
            f"{label}: в списках нет людей с известными баллами; "
            f"для ожидающих экзамен с 0 взята доля {overall:.0%} "
            "(не из заглушек в файле)"
        )
        return ConsentModel(
            by_code=by_code,
            overall_rate=0.0,
            competitive_rate=0.0,
            noncompetitive_rate=0.0,
            mean_undecided=float(mean_und),
            description=description,
            scored_n=0,
            pending_unknown_n=pending_unknown_n,
        )

    overall_raw = _rate(list(scored.keys()), scored) or 0.0
    # Классификация «конкурент / нет» — среди тех, у кого баллы уже есть.
    ovp = simulate_enrollment(scored, seats, consent_only=False)

    competitive: list[int] = []
    external: list[int] = []
    noncompetitive: list[int] = []
    ovp_dest: dict[int, str | None] = {}
    for code in scored:
        dest = ovp.destination(code)
        ovp_dest[code] = dest
        if dest is None:
            noncompetitive.append(code)
        elif dest == EXTERNAL:
            external.append(code)
        else:
            competitive.append(code)

    p_comp_raw = _rate(competitive, scored)
    p_ext_raw = _rate(external, scored)
    p_non_raw = _rate(noncompetitive, scored)
    # Для отчёта: прочие = не занявшие загруженное место в ОВП.
    p_other_raw = _rate(external + noncompetitive, scored)

    p_comp = _undecided_rate(p_comp_raw, UNDECIDED_PRIOR_COMPETITIVE)
    p_ext = _undecided_rate(
        p_ext_raw,
        p_other_raw if p_other_raw is not None and p_other_raw < 1.0 else UNDECIDED_PRIOR_OTHER,
    )
    p_non = _undecided_rate(
        p_non_raw,
        p_other_raw if p_other_raw is not None and p_other_raw < 1.0 else UNDECIDED_PRIOR_OTHER,
    )
    overall = _undecided_rate(overall_raw, UNDECIDED_PRIOR_OVERALL)

    bands, competitive_p = _competitive_band_probs(
        competitive, scored, ovp_dest, p_comp
    )

    # Базовые (auto) вероятности — только для неопределившихся < 1.0
    auto_probs: dict[int, float] = {}
    competitor_flags: dict[int, bool] = {}
    for code, profile in applicants.items():
        if profile.consent:
            auto_probs[code] = 1.0
            competitor_flags[code] = False
            continue
        if code not in scored:
            # Ждут экзамен, балл-заглушка: не классифицируем как конкурентов ОВП.
            auto_probs[code] = float(overall)
            competitor_flags[code] = False
            continue
        dest = ovp_dest.get(code)
        if dest is not None and dest != EXTERNAL:
            auto_probs[code] = float(competitive_p.get(code, p_comp))
            competitor_flags[code] = True
        elif dest == EXTERNAL:
            auto_probs[code] = float(p_ext)
            competitor_flags[code] = False
        else:
            auto_probs[code] = float(p_non)
            competitor_flags[code] = False

    by_code: dict[int, float] = {}
    for code, profile in applicants.items():
        if profile.consent:
            by_code[code] = 1.0
            continue
        base = auto_probs[code]
        is_loaded_competitor = competitor_flags[code]

        if scenario == "auto":
            by_code[code] = base
        elif scenario == "balanced":
            by_code[code] = float(overall)
        elif scenario == "optimistic":
            by_code[code] = float(base * 0.5) if is_loaded_competitor else float(base * 0.75)
        elif scenario == "pessimistic":
            by_code[code] = _pessimistic_p(
                base, competitor=is_loaded_competitor, overall=overall
            )
        else:
            by_code[code] = base

    undecided = [c for c, p in applicants.items() if not p.consent]
    mean_und = (
        sum(by_code[c] for c in undecided) / len(undecided) if undecided else 0.0
    )
    label = SCENARIO_LABELS.get(scenario, scenario)
    report_comp = p_comp_raw if p_comp_raw is not None else p_comp
    report_other = p_other_raw if p_other_raw is not None else p_non
    pending_note = (
        f"доли по {len(scored)} с известными баллами"
        + (
            f", без {pending_unknown_n} ждущих экзамен с 0 в файле"
            if pending_unknown_n
            else ""
        )
    )
    band_note = ""
    if any(band.n for band in bands):
        parts = [
            f"{band.label}={band.rate:.0%} (n={band.n})" for band in bands
        ]
        band_note = "; по месту среди конкурентов: " + ", ".join(parts)
    description = (
        f"{label}: {pending_note} (не 100%; "
        f"«согласие подадут все» — отдельный крайний случай); "
        f"все вместе={overall_raw:.0%}, "
        f"конкуренты={report_comp:.0%}, прочие={report_other:.0%}"
        f"{band_note}; "
        f"средняя доля без согласия={mean_und:.0%}"
    )
    return ConsentModel(
        by_code=by_code,
        overall_rate=float(overall_raw),
        competitive_rate=float(p_comp_raw if p_comp_raw is not None else 0.0),
        noncompetitive_rate=float(p_other_raw if p_other_raw is not None else 0.0),
        mean_undecided=float(mean_und),
        description=description,
        scored_n=len(scored),
        pending_unknown_n=pending_unknown_n,
        bands=bands,
    )


_MC_CTX: dict = {}


def _iter_rng(seed: int, iteration: int) -> random.Random:
    """Детерминированный RNG прогона (одинаков в serial и parallel)."""
    return random.Random((seed * 1_000_003 + iteration) & 0xFFFFFFFF)


def _mc_init(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    applicant_code: int,
    undecided: list[int],
    probs: dict[int, float],
    mean_p: float,
    my_programs: list[str],
) -> None:
    """Инициализация воркера ProcessPool (Windows spawn)."""
    global _MC_CTX
    _MC_CTX = {
        "applicants": applicants,
        "seats": seats,
        "applicant_code": applicant_code,
        "undecided": undecided,
        "probs": probs,
        "mean_p": mean_p,
        "my_programs": my_programs,
    }


def _mc_chunk(start: int, count: int, seed: int) -> dict[str, int | dict[str, int]]:
    """Серия MC-прогонов для одного процесса."""
    return _mc_chunk_with_ctx(_MC_CTX, start, count, seed)


def _mc_chunk_with_ctx(
    ctx: dict,
    start: int,
    count: int,
    seed: int,
) -> dict[str, int | dict[str, int]]:
    """Серия MC-прогонов с переданным контекстом (без global race в serial)."""
    counts = {program: 0 for program in ctx["my_programs"]}
    external_count = 0
    none_count = 0
    any_count = 0

    for i in range(count):
        iteration = start + i
        rng = _iter_rng(seed, iteration)
        consented: set[int] = {ctx["applicant_code"]}
        for code in ctx["undecided"]:
            if rng.random() < ctx["probs"].get(code, ctx["mean_p"]):
                consented.add(code)

        result = simulate_enrollment(
            ctx["applicants"],
            ctx["seats"],
            consent_only=True,
            extra_consent=frozenset(consented),
        )
        dest = result.destination(ctx["applicant_code"])
        if dest is None:
            none_count += 1
        elif dest == EXTERNAL:
            external_count += 1
        else:
            if dest in counts:
                counts[dest] += 1
            any_count += 1

    return {
        "counts": counts,
        "external": external_count,
        "none": none_count,
        "any": any_count,
    }


def _merge_mc_partials(
    partials: list[dict[str, int | dict[str, int]]],
    my_programs: list[str],
) -> tuple[dict[str, int], int, int, int]:
    """Суммирует счётчики частичных прогонов."""
    counts = {program: 0 for program in my_programs}
    external_count = 0
    none_count = 0
    any_count = 0
    for part in partials:
        part_counts = part["counts"]
        assert isinstance(part_counts, dict)
        for program in my_programs:
            counts[program] += int(part_counts.get(program, 0))
        external_count += int(part["external"])
        none_count += int(part["none"])
        any_count += int(part["any"])
    return counts, external_count, none_count, any_count


def estimate_probability(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    applicant_code: int,
    *,
    n_simulations: int = 1000,
    scenario: ConsentScenario = "auto",
    consent_probability: float | None = None,
    seed: int = 42,
    n_workers: int = 1,
    focus_program: str | None = None,
    consent_model: ConsentModel | None = None,
) -> ProbabilityEstimate:
    """
    Monte Carlo: у каждого без согласия — своя P(согласие).

    scenario — авто / сбалансированный / оптим. / пессим.
    consent_probability — ручной override одной константой (игнорирует scenario).
    n_workers — параллельные процессы для MC (>1 ускоряет на многоядерном CPU).
    focus_program — считать шансы для одной программы (подграф DA, быстрее).
    consent_model — готовая модель согласий; иначе считается здесь.
    Ваш код в каждом прогоне считается согласившимся.
    """
    if applicant_code not in applicants:
        raise KeyError(f"Код {applicant_code} отсутствует в загруженных списках")
    if n_simulations <= 0:
        raise ValueError("n_simulations должно быть > 0")

    my_all = [
        app.program for app in applicants[applicant_code].sorted_applications()
    ]
    resolved_focus: str | None = None
    sim_applicants = applicants
    sim_seats = seats
    if focus_program:
        known = list(dict.fromkeys([*my_all, *seats.keys()]))
        resolved_focus = resolve_focus_program(
            focus_program, known, preferred=my_all
        )
        if resolved_focus not in my_all:
            raise ValueError(
                f"Код {applicant_code} не подавал на «{resolved_focus}»"
            )
        sim_applicants, sim_seats = subgraph_for_program(
            applicants, seats, resolved_focus
        )
        if applicant_code not in sim_applicants:
            raise KeyError(
                f"Код {applicant_code} не попал в подграф «{resolved_focus}»"
            )

    effective_scenario: ConsentScenario = scenario
    if consent_probability is not None:
        probs = {
            code: (1.0 if profile.consent else float(consent_probability))
            for code, profile in applicants.items()
        }
        model_desc = (
            f"вручную задана вероятность согласия {consent_probability:.0%} "
            "для всех без согласия"
        )
        mean_p = float(consent_probability)
    else:
        model = consent_model or estimate_consent_model(
            applicants, seats, scenario=scenario
        )
        probs = model.by_code
        model_desc = model.description
        mean_p = model.mean_undecided

    remaining = {
        app.program for app in sim_applicants[applicant_code].applications
    }
    my_programs = [name for name in my_all if name in remaining]

    undecided = [
        code
        for code, profile in sim_applicants.items()
        if not profile.consent and code != applicant_code
    ]

    workers = max(1, n_workers)
    # Windows spawn дороже короткого DA: мелкий N оставляем serial.
    if workers > 1 and n_simulations >= max(32, workers * 8):
        chunk_size = (n_simulations + workers - 1) // workers
        tasks = []
        for w in range(workers):
            start = w * chunk_size
            if start >= n_simulations:
                break
            count = min(chunk_size, n_simulations - start)
            tasks.append((start, count))

        partials: list[dict[str, int | dict[str, int]]] = []
        with ProcessPoolExecutor(
            max_workers=len(tasks),
            initializer=_mc_init,
            initargs=(
                sim_applicants,
                sim_seats,
                applicant_code,
                undecided,
                probs,
                mean_p,
                my_programs,
            ),
        ) as pool:
            futures = [
                pool.submit(_mc_chunk, start, count, seed) for start, count in tasks
            ]
            for fut in as_completed(futures):
                partials.append(fut.result())
        counts, external_count, none_count, any_count = _merge_mc_partials(
            partials, my_programs
        )
    else:
        local_ctx = {
            "applicants": sim_applicants,
            "seats": sim_seats,
            "applicant_code": applicant_code,
            "undecided": undecided,
            "probs": probs,
            "mean_p": mean_p,
            "my_programs": my_programs,
        }
        counts, external_count, none_count, any_count = _merge_mc_partials(
            [_mc_chunk_with_ctx(local_ctx, 0, n_simulations, seed)],
            my_programs,
        )

    n = float(n_simulations)
    return ProbabilityEstimate(
        n_simulations=n_simulations,
        by_program={p: counts[p] / n for p in my_programs},
        any_loaded=any_count / n,
        external=external_count / n,
        none=none_count / n,
        mean_consent_probability=mean_p,
        consent_model_description=model_desc,
        scenario=effective_scenario,
        focus_program=resolved_focus,
    )
