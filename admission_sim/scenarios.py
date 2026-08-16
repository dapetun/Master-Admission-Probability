"""Сценарии what-if и оценка вероятности (Monte Carlo)."""

from __future__ import annotations

from admission_sim.model import ApplicantProfile


def what_if_consent(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int],
    applicant_code: int,
) -> dict[int, str | None]:
    """Пересчитывает ВПП в предположении, что указанный код подал согласие."""
    raise NotImplementedError("Сценарии what-if будут реализованы позже.")


def estimate_probability(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int],
    applicant_code: int,
    *,
    n_simulations: int = 1000,
) -> dict[str, float]:
    """Оценивает долю прогонов, где абитуриент зачислен (по программам)."""
    raise NotImplementedError("Monte Carlo будет реализован позже.")
