"""Доменные модели абитуриентов и программ."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ApplicationRow:
    """Строка конкурсного списка по одной программе."""

    applicant_code: int
    program: str
    priority: int
    score: float
    rank: int
    consent: bool
    status: str


@dataclass(slots=True)
class ApplicantProfile:
    """Профиль поступающего, собранный по всем загруженным программам."""

    code: int
    applications: list[ApplicationRow] = field(default_factory=list)
    consent: bool = False
    missing_higher_priority: bool = False
