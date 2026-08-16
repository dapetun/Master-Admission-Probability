"""Общий пайплайн анализа для CLI и UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from admission_sim.load import load_dataset
from admission_sim.model import Dataset
from admission_sim.scenarios import (
    ConsentScenario,
    Counterfactual,
    ProbabilityEstimate,
    counterfactuals_for_threats,
    estimate_probability,
    what_if_consent,
)
from admission_sim.simulate import EnrollmentResult, simulate_enrollment


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Полный результат расчёта для одного кода поступающего."""

    dataset: Dataset
    my_code: int
    vpp: EnrollmentResult
    vpp_if_consent: EnrollmentResult
    ovp: EnrollmentResult
    counterfactuals: list[Counterfactual]
    probability: ProbabilityEstimate | None
    include_pending: bool
    zero_seat_programs: list[str]
    unknown_seat_programs: list[str]


def run_analysis(
    data_dir: Path,
    seats_path: Path,
    my_code: int,
    *,
    include_pending: bool = True,
    campus: str | None = None,
    monte_carlo: int = 0,
    scenario: ConsentScenario = "auto",
    consent_p: float | None = None,
    seed: int = 42,
    threats: int = 30,
) -> AnalysisResult:
    """Загружает данные и считает ВПП / ОВП / what-if / Monte Carlo."""
    dataset = load_dataset(
        data_dir,
        seats_path,
        include_pending=include_pending,
        campus=campus,
    )
    if my_code not in dataset.applicants:
        raise KeyError(
            f"Код {my_code} не найден среди {len(dataset.applicants)} "
            f"абитуриентов в {data_dir}"
        )

    zero_seat_programs = [
        name
        for name in dataset.programs
        if dataset.seats.get(name) is not None and int(dataset.seats[name] or 0) <= 0
    ]
    unknown_seat_programs = [
        name for name in dataset.programs if dataset.seats.get(name) is None
    ]
    vpp = simulate_enrollment(dataset.applicants, dataset.seats, consent_only=True)
    vpp_if_consent = what_if_consent(dataset.applicants, dataset.seats, my_code)
    ovp = simulate_enrollment(dataset.applicants, dataset.seats, consent_only=False)
    counterfactuals = counterfactuals_for_threats(
        dataset.applicants,
        dataset.seats,
        my_code,
        limit=threats,
    )
    probability = None
    if monte_carlo > 0:
        probability = estimate_probability(
            dataset.applicants,
            dataset.seats,
            my_code,
            n_simulations=monte_carlo,
            scenario=scenario,
            consent_probability=consent_p,
            seed=seed,
        )

    return AnalysisResult(
        dataset=dataset,
        my_code=my_code,
        vpp=vpp,
        vpp_if_consent=vpp_if_consent,
        ovp=ovp,
        counterfactuals=counterfactuals,
        probability=probability,
        include_pending=include_pending,
        zero_seat_programs=zero_seat_programs,
        unknown_seat_programs=unknown_seat_programs,
    )
