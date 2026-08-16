"""Каскадная симуляция зачисления по высшему проходному приоритету."""

from __future__ import annotations

from admission_sim.model import ApplicantProfile


def simulate_enrollment(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int],
    *,
    consent_only: bool = True,
) -> dict[int, str | None]:
    """
    Возвращает отображение код → программа | None.

    При consent_only=True учитываются только подавшие согласие (ВПП).
    При False — все участники конкурса (ОВП, «если бы все согласились»).
    """
    raise NotImplementedError("Симулятор зачисления будет реализован позже.")
