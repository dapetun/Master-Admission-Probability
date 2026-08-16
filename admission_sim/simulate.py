"""Каскадная симуляция зачисления по высшему проходному приоритету."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

from admission_sim.model import EXTERNAL, ApplicantProfile, ApplicationRow


@dataclass(frozen=True, slots=True)
class EnrollmentResult:
    """Результат симуляции: код → программа, EXTERNAL или None."""

    assignment: dict[int, str | None]
    consent_only: bool

    def enrolled_for(self, program: str) -> list[int]:
        """Коды, зачисленные на программу."""
        return [code for code, dest in self.assignment.items() if dest == program]

    def destination(self, code: int) -> str | None:
        """Куда зачислен абитуриент (None — не в assignment)."""
        return self.assignment.get(code)


def preference_list(profile: ApplicantProfile) -> list[str]:
    """
    Предпочтения с виртуальными EXTERNAL на разрывах приоритетов.

    EXTERNAL моделирует программы вне загруженных CSV: при согласии
    абитуриент уходит туда и не занимает места в загруженных конкурсах.
    """
    apps = profile.sorted_applications()
    if not apps:
        return []

    prefs: list[str] = []
    expected = 1
    for app in apps:
        while expected < app.priority:
            prefs.append(EXTERNAL)
            expected += 1
        prefs.append(app.program)
        expected = app.priority + 1
    return prefs


def _rank_on_program(profile: ApplicantProfile, program: str) -> int:
    app = profile.application_for(program)
    if app is None:
        return 10**9
    return app.rank


def simulate_enrollment(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    *,
    consent_only: bool = True,
) -> EnrollmentResult:
    """
    Отложенный приём (deferred acceptance) по приоритетам внутри одного вуза.

    seats[program] = None — КЦП неизвестны: программа поглощает абитуриента
    как EXTERNAL (не сбрасывает его на следующие приоритеты других кампусов).
    """
    active = {
        code: profile
        for code, profile in applicants.items()
        if profile.applications and (profile.consent or not consent_only)
    }
    if not active:
        return EnrollmentResult(assignment={}, consent_only=consent_only)

    prefs = {code: preference_list(profile) for code, profile in active.items()}
    next_idx = {code: 0 for code in active}
    held: dict[str, list[tuple[int, int]]] = {
        program: [] for program, capacity in seats.items() if capacity is not None
    }

    free: list[int] = list(active.keys())
    final_external: set[int] = set()
    exhausted: set[int] = set()

    while free:
        code = free.pop()
        if code in final_external or code in exhausted:
            continue

        pref = prefs[code]
        idx = next_idx[code]
        if idx >= len(pref):
            exhausted.add(code)
            continue

        choice = pref[idx]
        next_idx[code] = idx + 1

        if choice == EXTERNAL:
            final_external.add(code)
            continue

        capacity = seats.get(choice)
        if capacity is None:
            # Нет КЦП (часто другой кампус без сводки) — не роняем на следующие.
            final_external.add(code)
            continue

        rank = _rank_on_program(active[code], choice)
        bucket = held.setdefault(choice, [])
        bucket.append((rank, code))
        bucket.sort(key=lambda item: item[0])

        if len(bucket) > int(capacity):
            _, bumped = bucket.pop()
            free.append(bumped)

    assignment: dict[int, str | None] = {}
    for program, bucket in held.items():
        capacity = seats.get(program)
        if capacity is None:
            continue
        for _, code in bucket[: max(int(capacity), 0)]:
            assignment[code] = program

    for code in final_external:
        assignment[code] = EXTERNAL

    for code in active:
        if code not in assignment:
            assignment[code] = None

    return EnrollmentResult(assignment=assignment, consent_only=consent_only)


def with_forced_consent(
    applicants: dict[int, ApplicantProfile],
    code: int,
) -> dict[int, ApplicantProfile]:
    """Копия профилей, где у указанного кода принудительно есть согласие."""
    if code not in applicants:
        raise KeyError(f"Код {code} отсутствует в загруженных списках")

    cloned: dict[int, ApplicantProfile] = {
        c: deepcopy(profile) for c, profile in applicants.items()
    }
    profile = cloned[code]
    profile.applications = [
        replace(app, consent=True) if isinstance(app, ApplicationRow) else app
        for app in profile.applications
    ]
    profile.consent = True
    return cloned
