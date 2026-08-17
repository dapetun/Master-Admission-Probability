"""Каскадная симуляция зачисления по высшему проходному приоритету."""

from __future__ import annotations

import bisect
from collections import defaultdict, deque
from dataclasses import dataclass

from admission_sim.load import _finalize_profile
from admission_sim.model import EXTERNAL, ApplicantProfile


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
    Один EXTERNAL на первый разрыв: DA сразу поглощает и дальше не идёт.
    """
    return profile.cached_preference_list()


def resolve_focus_program(
    query: str,
    programs: list[str],
    *,
    preferred: list[str] | None = None,
) -> str:
    """
    Находит программу по точному ключу или уникальной подстроке.

    Сначала смотрит preferred (обычно программы пользователя), затем все.
    """
    text = query.strip()
    if not text:
        raise ValueError("Пустое название программы")

    preferred_list = list(preferred or [])
    if text in preferred_list:
        return text
    if text in programs:
        return text

    needle = text.lower()

    def unique_hit(pool: list[str]) -> str | None:
        hits = [name for name in pool if needle in name.lower()]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            shown = ", ".join(hits[:8])
            extra = "" if len(hits) <= 8 else f" и ещё {len(hits) - 8}"
            raise ValueError(f"Несколько программ подходят под «{text}»: {shown}{extra}")
        return None

    found = unique_hit(preferred_list)
    if found is not None:
        return found
    found = unique_hit(programs)
    if found is not None:
        return found
    raise KeyError(f"Программа «{text}» не найдена")


def subgraph_for_program(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    target_program: str,
) -> tuple[dict[int, ApplicantProfile], dict[str, int | None]]:
    """
    Сужает данные до выбранной программы и тех, что могут увести с неё.

    В граф входят сама программа; конкурсы с более высоким приоритетом
    у тех, кто на неё же подался; рекурсивно — конкурсы, способные
    перетянуть этих людей (и всех, кто с ними конкурирует за места).
    Заявки ниже по приоритету отбрасываются: на зачисление на target
    они не влияют. Пробелы приоритетов по-прежнему дают EXTERNAL.
    Конкурсы без числа мест (кроме выбранной) не включаем: заявка
    пропадает, в приоритетах появляется EXTERNAL — то же, что поглотитель.
    Конкурсы с нулём мест оставляем (отказ и переход дальше), но чужих
    абитуриентов с них не берём.
    """
    try:
        return subgraph_for_programs(applicants, seats, [target_program])
    except KeyError:
        raise KeyError(
            f"Программа «{target_program}» отсутствует в заявках"
        ) from None


def subgraph_for_programs(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    target_programs: list[str],
) -> tuple[dict[int, ApplicantProfile], dict[str, int | None]]:
    """Объединение подграфов нескольких целевых программ."""
    by_program: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for code, profile in applicants.items():
        for app in profile.applications:
            by_program[app.program].append((code, app.priority))

    targets = [name for name in dict.fromkeys(target_programs) if name in by_program]
    if not targets:
        raise KeyError("Ни одна из программ не найдена в заявках")
    return _subgraph_from_targets(applicants, seats, by_program, targets)


def _subgraph_from_targets(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    by_program: dict[str, list[tuple[int, int]]],
    target_programs: list[str],
) -> tuple[dict[int, ApplicantProfile], dict[str, int | None]]:
    target_set = set(target_programs)
    needed_programs: set[str] = set(target_set)
    needed_applicants: set[int] = set()
    queue: deque[str] = deque(target_programs)

    def _competes(program: str) -> bool:
        """Чужие заявки нужны только где есть положительное число мест."""
        if program in target_set:
            return True
        capacity = seats.get(program)
        return capacity is not None and int(capacity) > 0

    while queue:
        program = queue.popleft()
        if not _competes(program):
            # Нет мест / неизвестны: поглотитель или мгновенный отказ.
            # Чужих абитуриентов с этого конкурса не подмешиваем.
            continue
        for code, prio_at_program in by_program.get(program, ()):
            needed_applicants.add(code)
            profile = applicants[code]
            for app in profile.applications:
                if app.priority >= prio_at_program:
                    continue
                if app.program in needed_programs:
                    continue
                capacity = seats.get(app.program)
                if capacity is None and app.program not in target_set:
                    # Как EXTERNAL: согласившийся уходит и не занимает места.
                    continue
                needed_programs.add(app.program)
                queue.append(app.program)

    reduced: dict[int, ApplicantProfile] = {}
    for code in needed_applicants:
        profile = applicants[code]
        apps = [a for a in profile.applications if a.program in needed_programs]
        if not apps:
            continue
        new_profile = ApplicantProfile(
            code=code,
            applications=list(apps),
            consent=profile.consent,
        )
        _finalize_profile(new_profile)
        reduced[code] = new_profile

    reduced_seats = {name: seats.get(name) for name in needed_programs}
    return reduced, reduced_seats


def _rank_on_program(profile: ApplicantProfile, program: str) -> int:
    app = profile.application_for(program)
    if app is None:
        return 10**9
    return app.rank


_CONSENT_ACTIVE: tuple[int, dict[int, ApplicantProfile]] | None = None


def _consent_only_active(
    applicants: dict[int, ApplicantProfile],
) -> dict[int, ApplicantProfile]:
    """Кэш уже согласившихся: what-if и MC много раз гоняют один и тот же dict."""
    global _CONSENT_ACTIVE
    cached = _CONSENT_ACTIVE
    key = id(applicants)
    if cached is not None and cached[0] == key:
        return cached[1]
    active = {
        code: profile
        for code, profile in applicants.items()
        if profile.consent and profile.applications
    }
    _CONSENT_ACTIVE = (key, active)
    return active


def simulate_enrollment(
    applicants: dict[int, ApplicantProfile],
    seats: dict[str, int | None],
    *,
    consent_only: bool = True,
    extra_consent: frozenset[int] | None = None,
) -> EnrollmentResult:
    """
    Отложенный приём (deferred acceptance) по приоритетам внутри одного вуза.

    seats[program] = None — КЦП неизвестны: программа поглощает абитуриента
    как EXTERNAL (не сбрасывает его на следующие приоритеты других кампусов).
    """
    extra = extra_consent
    if consent_only:
        active = _consent_only_active(applicants)
        if extra:
            active = dict(active)
            for code in extra:
                if code in active:
                    continue
                profile = applicants.get(code)
                if profile is not None and profile.applications:
                    active[code] = profile
    else:
        active = {
            code: profile
            for code, profile in applicants.items()
            if profile.applications
        }
    if not active:
        return EnrollmentResult(assignment={}, consent_only=consent_only)

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

        pref = active[code]._prefs
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
        bisect.insort(bucket, (rank, code))

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
