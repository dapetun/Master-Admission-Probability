"""Доменные модели абитуриентов и программ."""

from __future__ import annotations

from dataclasses import dataclass, field

# Маркер зачисления на программу вне загруженных CSV.
EXTERNAL = "__external__"


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

    def application_for(self, program: str) -> ApplicationRow | None:
        """Возвращает заявку на программу или None."""
        for app in self.applications:
            if app.program == program:
                return app
        return None

    def sorted_applications(self) -> list[ApplicationRow]:
        """Заявки по возрастанию приоритета (1 — высший)."""
        return sorted(self.applications, key=lambda a: a.priority)


@dataclass(frozen=True, slots=True)
class Dataset:
    """Загруженные профили, места и имена программ."""

    applicants: dict[int, ApplicantProfile]
    seats: dict[str, int | None]
    programs: list[str]
    source_files: list[str]
    incomplete_priority_codes: list[int]
