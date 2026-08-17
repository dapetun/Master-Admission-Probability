"""Доменные модели абитуриентов и программ."""

from __future__ import annotations

from dataclasses import dataclass, field

# Маркер зачисления на программу вне загруженных CSV.
EXTERNAL = "__external__"


def _preference_list_from_sorted(apps: list[ApplicationRow]) -> list[str]:
    """Один EXTERNAL на разрыв: DA сразу уходит и дальше не идёт."""
    if not apps:
        return []
    prefs: list[str] = []
    expected = 1
    for app in apps:
        if expected < app.priority:
            prefs.append(EXTERNAL)
            break
        prefs.append(app.program)
        expected = app.priority + 1
    return prefs


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
    _by_program: dict[str, ApplicationRow] = field(default_factory=dict, repr=False)
    _sorted: list[ApplicationRow] = field(default_factory=list, repr=False)
    _prefs: list[str] = field(default_factory=list, repr=False)

    def rebuild_index(self) -> None:
        """Пересчитывает словари заявок и кэш предпочтений после сборки списка."""
        by_program: dict[str, ApplicationRow] = {}
        for app in self.applications:
            by_program[app.program] = app
        self._by_program = by_program
        self._sorted = sorted(self.applications, key=lambda a: a.priority)
        self._prefs = _preference_list_from_sorted(self._sorted)

    def _ensure_index(self) -> None:
        if self._by_program or not self.applications:
            return
        self.rebuild_index()

    def application_for(self, program: str) -> ApplicationRow | None:
        """Возвращает заявку на программу или None."""
        self._ensure_index()
        return self._by_program.get(program)

    def sorted_applications(self) -> list[ApplicationRow]:
        """Заявки по возрастанию приоритета (1 — высший)."""
        self._ensure_index()
        return self._sorted

    def cached_preference_list(self) -> list[str]:
        """Предпочтения с EXTERNAL на разрывах; кэш после rebuild_index."""
        self._ensure_index()
        return self._prefs


@dataclass(frozen=True, slots=True)
class Dataset:
    """Загруженные профили, места и имена программ."""

    applicants: dict[int, ApplicantProfile]
    seats: dict[str, int | None]
    programs: list[str]
    source_files: list[str]
    incomplete_priority_codes: list[int]
