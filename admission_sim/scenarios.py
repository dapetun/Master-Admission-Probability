"""Сценарии what-if и оценка вероятности (Monte Carlo)."""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Literal

from admission_sim.model import EXTERNAL, ApplicantProfile
from admission_sim.simulate import EnrollmentResult, simulate_enrollment, with_forced_consent

ConsentScenario = Literal["auto", "balanced", "optimistic", "pessimistic"]

SCENARIO_LABELS: dict[ConsentScenario, str] = {
    "auto": "Авто (из данных)",
    "balanced": "Сбалансированный",
    "optimistic": "Оптимистичный (для вас)",
    "pessimistic": "Пессимистичный (для вас)",
}


@dataclass(frozen=True, slots=True)
class Counterfactual:
    """Результат «если только этот код подаст согласие»."""

    code: int
    destination: str | None
    displaces_me: bool
    my_destination_before: str | None
    my_destination_after: str | None


@dataclass(frozen=True, slots=True)
class ConsentModel:
    """Индивидуальные P(согласие) и сводка для отчёта."""

    by_code: dict[int, float]
    overall_rate: float
    competitive_rate: float
    noncompetitive_rate: float
    mean_undecided: float
    description: str


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


def what_if_consent(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    applicant_code: int,
) -> EnrollmentResult:
    """Пересчитывает ВПП в предположении, что указанный код подал согласие."""
    forced = with_forced_consent(applicants, applicant_code)
    return simulate_enrollment(forced, seats, consent_only=True)


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
    limit: int = 30,
) -> list[Counterfactual]:
    """
    Контрфакты для абитуриентов без согласия, которые выше вас
    хотя бы в одной вашей программе.
    """
    if my_code not in applicants:
        raise KeyError(f"Код {my_code} отсутствует в загруженных списках")

    me = applicants[my_code]
    my_programs = {app.program: app.rank for app in me.applications}
    baseline = simulate_enrollment(applicants, seats, consent_only=True)
    my_before = baseline.destination(my_code)

    candidates: list[tuple[int, int]] = []
    for code, profile in applicants.items():
        if code == my_code or profile.consent:
            continue
        best_threat = None
        for app in profile.applications:
            my_rank = my_programs.get(app.program)
            if my_rank is None:
                continue
            if app.rank < my_rank:
                gap = my_rank - app.rank
                if best_threat is None or gap > best_threat:
                    best_threat = gap
        if best_threat is not None:
            candidates.append((best_threat, code))

    candidates.sort(reverse=True)
    results: list[Counterfactual] = []
    for _, code in candidates[:limit]:
        result = what_if_consent(applicants, seats, code)
        my_after = result.destination(my_code)
        dest = result.destination(code)
        results.append(
            Counterfactual(
                code=code,
                destination=dest,
                displaces_me=_displaces_me(my_before, my_after),
                my_destination_before=my_before,
                my_destination_after=my_after,
            )
        )
    return results


def _rate(codes: list[int], applicants: dict[int, ApplicantProfile]) -> float | None:
    if not codes:
        return None
    return sum(1 for c in codes if applicants[c].consent) / len(codes)


def estimate_consent_model(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    *,
    scenario: ConsentScenario = "auto",
) -> ConsentModel:
    """
    Оценивает P(согласие) по текущим спискам.

    scenario:
      auto — индивидуально из долей согласий (конкурентные / прочие);
      balanced — одна общая доля overall для всех без согласия;
      optimistic — конкуренты реже соглашаются (лучше для вас);
      pessimistic — конкуренты чаще соглашаются (хуже для вас).
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

    overall = _rate(list(applicants.keys()), applicants) or 0.0
    ovp = simulate_enrollment(applicants, seats, consent_only=False)

    competitive: list[int] = []
    noncompetitive: list[int] = []
    ovp_dest: dict[int, str | None] = {}
    for code in applicants:
        dest = ovp.destination(code)
        ovp_dest[code] = dest
        if dest is not None and dest != EXTERNAL:
            competitive.append(code)
        else:
            noncompetitive.append(code)

    p_comp = _rate(competitive, applicants)
    p_non = _rate(noncompetitive, applicants)
    if p_comp is None:
        p_comp = overall
    if p_non is None:
        p_non = min(overall, 0.15)

    pri1_comp = [
        c
        for c in competitive
        if any(
            a.priority == 1 and a.program == ovp_dest[c]
            for a in applicants[c].applications
        )
    ]
    p_pri1 = _rate(pri1_comp, applicants)
    if p_pri1 is None:
        p_pri1 = min(1.0, p_comp + 0.1)

    # Базовые (auto) вероятности
    auto_probs: dict[int, float] = {}
    for code, profile in applicants.items():
        if profile.consent:
            auto_probs[code] = 1.0
            continue
        dest = ovp_dest.get(code)
        if dest is not None and dest != EXTERNAL:
            app = profile.application_for(dest)
            if app is not None and app.priority == 1:
                auto_probs[code] = float(p_pri1)
            else:
                auto_probs[code] = float(p_comp)
        elif dest == EXTERNAL:
            auto_probs[code] = float(p_comp)
        else:
            auto_probs[code] = float(p_non)

    by_code: dict[int, float] = {}
    for code, profile in applicants.items():
        if profile.consent:
            by_code[code] = 1.0
            continue
        base = auto_probs[code]
        is_comp = ovp_dest.get(code) is not None  # loaded seat or EXTERNAL

        if scenario == "auto":
            by_code[code] = base
        elif scenario == "balanced":
            by_code[code] = float(overall)
        elif scenario == "optimistic":
            # Реже соглашаются те, кто мог бы занять место
            by_code[code] = float(base * 0.5) if is_comp else float(base * 0.75)
        elif scenario == "pessimistic":
            if ovp_dest.get(code) is not None and ovp_dest[code] != EXTERNAL:
                by_code[code] = float(min(1.0, max(base * 1.5, 0.85)))
            elif ovp_dest.get(code) == EXTERNAL:
                by_code[code] = float(min(1.0, max(base, 0.7)))
            else:
                by_code[code] = float(min(1.0, max(overall, base)))
        else:
            by_code[code] = base

    undecided = [c for c, p in applicants.items() if not p.consent]
    mean_und = (
        sum(by_code[c] for c in undecided) / len(undecided) if undecided else 0.0
    )
    label = SCENARIO_LABELS.get(scenario, scenario)
    description = (
        f"{label}: overall={overall:.0%}, "
        f"конкурентные(ОВП)={p_comp:.0%}, прочие={p_non:.0%}, "
        f"средний p без согласия={mean_und:.0%}"
    )
    return ConsentModel(
        by_code=by_code,
        overall_rate=float(overall),
        competitive_rate=float(p_comp),
        noncompetitive_rate=float(p_non),
        mean_undecided=float(mean_und),
        description=description,
    )


def estimate_probability(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    applicant_code: int,
    *,
    n_simulations: int = 1000,
    scenario: ConsentScenario = "auto",
    consent_probability: float | None = None,
    seed: int = 42,
) -> ProbabilityEstimate:
    """
    Monte Carlo: у каждого без согласия — своя P(согласие).

    scenario — авто / сбалансированный / оптим. / пессим.
    consent_probability — ручной override одной константой (игнорирует scenario).
    Ваш код в каждом прогоне считается согласившимся.
    """
    if applicant_code not in applicants:
        raise KeyError(f"Код {applicant_code} отсутствует в загруженных списках")
    if n_simulations <= 0:
        raise ValueError("n_simulations должно быть > 0")

    effective_scenario: ConsentScenario = scenario
    if consent_probability is not None:
        probs = {
            code: (1.0 if profile.consent else float(consent_probability))
            for code, profile in applicants.items()
        }
        model_desc = f"ручной override p={consent_probability:.2f} для всех без согласия"
        mean_p = float(consent_probability)
    else:
        model = estimate_consent_model(applicants, seats, scenario=scenario)
        probs = model.by_code
        model_desc = model.description
        mean_p = model.mean_undecided

    rng = random.Random(seed)
    # Только программы, куда подал заявку сам абитуриент — не все ключи seats.
    my_programs = [
        app.program for app in applicants[applicant_code].sorted_applications()
    ]
    counts = {program: 0 for program in my_programs}
    external_count = 0
    none_count = 0
    any_count = 0

    undecided = [
        code
        for code, profile in applicants.items()
        if not profile.consent and code != applicant_code
    ]

    for _ in range(n_simulations):
        cloned: dict[int, ApplicantProfile] = {
            c: deepcopy(p) for c, p in applicants.items()
        }
        me = cloned[applicant_code]
        me.consent = True
        me.applications = [replace(a, consent=True) for a in me.applications]

        for code in undecided:
            if rng.random() < probs.get(code, mean_p):
                profile = cloned[code]
                profile.consent = True
                profile.applications = [
                    replace(a, consent=True) for a in profile.applications
                ]

        result = simulate_enrollment(cloned, seats, consent_only=True)
        dest = result.destination(applicant_code)
        if dest is None:
            none_count += 1
        elif dest == EXTERNAL:
            external_count += 1
        else:
            if dest in counts:
                counts[dest] += 1
            any_count += 1

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
    )
