"""Загрузка CSV/Budget.xlsx конкурсных списков и конфигурации мест."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from admission_sim.model import ApplicantProfile, ApplicationRow, Dataset

CONSENT_YES = frozenset({"электронное", "бумажное", "б", "да", "yes", "+"})
DEFAULT_ACTIVE_STATUSES = frozenset(
    {
        "Участвуете в конкурсе",
        "Участвует в конкурсе",
    }
)
PENDING_STATUSES = frozenset(
    {
        "Ожидаются результаты испытаний",
        "Ожидание результатов ВИ",
    }
)

_DATE_SUFFIX = re.compile(r"\.20\d{2}-\d{2}-\d{2}_")
_PROGRAM_TITLE = re.compile(r'Образовательная программа "([^"]+)"')


def program_key(program: str, campus: str) -> str:
    """Уникальный ключ программы с кампусом (имена повторяются между городами)."""
    return f"{program} ({campus})"


def normalize_campus(raw: str) -> str:
    """Приводит строку кампуса из шапки отчёта к короткому виду."""
    text = " ".join(raw.replace("\n", " ").split())
    low = text.lower()
    if "санкт" in low or "петербург" in low:
        return "Санкт-Петербург"
    if "нижн" in low:
        return "Нижний Новгород"
    if "перм" in low:
        return "Пермь"
    if "москв" in low and "высшая школа" not in low:
        return "Москва"
    if "высшая школа" in low:
        return "Высшая школа бизнеса"
    if text.startswith("НИУ ВШЭ - "):
        return text.replace("НИУ ВШЭ - ", "").strip()
    return text or "неизвестно"


def program_and_campus_from_budget_xlsx(path: Path) -> tuple[str, str]:
    """Достаёт название программы и кампус из шапки Budget.xlsx."""
    header = pd.read_excel(path, header=None, nrows=1)
    title = str(header.iloc[0, 0])
    prog = _PROGRAM_TITLE.search(title)
    if not prog:
        raise ValueError(f"{path.name}: не найдено название программы в шапке")
    lines = [line.strip() for line in title.splitlines() if line.strip()]
    campus_raw = lines[-1] if lines else "неизвестно"
    return prog.group(1).strip(), normalize_campus(campus_raw)

REQUIRED_COLUMNS_CSV = (
    "Порядковый номер",
    "Приоритет конкурса",
    "Подано согласие",
    "Сумма баллов",
    "Статус",
    "Код поступающего",
)


def program_name_from_csv(path: Path) -> str:
    """Извлекает название программы из имени файла выгрузки CSV."""
    match = _DATE_SUFFIX.search(path.name)
    if match:
        return path.name[: match.start()]
    if ".20" in path.name:
        return path.name.split(".20", 1)[0]
    return path.stem


def _seats_from_entry(name: str, value: object) -> int | None:
    """
    Достаёт K из краткой (int) или полной ({seats: N, ...}) записи.

    None / seats: null — КЦП неизвестны (программа станет «поглотителем»
    EXTERNAL в симуляции, чтобы не сбрасывать людей на другие кампусы зря).
    """
    if value is None:
        return None
    if isinstance(value, dict):
        if value.get("seats") is None:
            return None
        return int(value["seats"])
    return int(value)


def load_seats(path: Path) -> dict[str, int | None]:
    """Читает бюджетные места из YAML (int или null)."""
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл мест: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    programs = raw.get("programs")
    if not isinstance(programs, dict) or not programs:
        raise ValueError(f"В {path} ожидается непустой словарь programs")
    return {
        str(name): _seats_from_entry(str(name), value)
        for name, value in programs.items()
    }


def load_csv_directory(directory: Path) -> list[Path]:
    """Возвращает список CSV в каталоге."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Каталог данных не найден: {directory}")
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"В {directory} нет CSV-файлов")
    return files


def list_data_files(
    directory: Path,
    *,
    campus: str | None = None,
) -> list[tuple[Path, str, str]]:
    """
    Находит CSV и *_Budget.xlsx по всему вузу (все кампусы по умолчанию).

    Имя программы для Budget.xlsx — с суффиксом кампуса: «Название (Москва)».
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Каталог данных не найден: {directory}")

    items: list[tuple[Path, str, str]] = []
    for path in sorted(directory.glob("*.csv")):
        items.append((path, program_name_from_csv(path), "csv"))

    for path in sorted(directory.glob("*_Budget.xlsx")):
        bare, file_campus = program_and_campus_from_budget_xlsx(path)
        if campus and campus.lower() not in file_campus.lower():
            continue
        items.append((path, program_key(bare, file_campus), "budget_xlsx"))

    if not items:
        raise FileNotFoundError(
            f"В {directory} нет CSV / *_Budget.xlsx"
            + (f" для кампуса «{campus}»" if campus else "")
        )
    return items


def _parse_consent(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().lower().replace("—", "-").replace("–", "-")
    if text in CONSENT_YES:
        return True
    if text in {"-", "", "nan", "none"}:
        return False
    return "электрон" in text or "бумаж" in text or text == "б"


def _parse_score(value: object) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    text = str(value).strip().replace(",", ".")
    if text in {"-", "—", "–", "", "nan"}:
        return 0.0
    return float(text)


def _parse_priority(value: object) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 10**6
    text = str(value).strip()
    if text in {"-", "—", "–", "", "nan"}:
        return 10**6
    return int(float(text))


def _read_program_csv(path: Path, program: str) -> list[ApplicationRow]:
    df = pd.read_csv(path, sep=";")
    missing = [c for c in REQUIRED_COLUMNS_CSV if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: нет колонок {missing}")

    rows: list[ApplicationRow] = []
    for _, series in df.iterrows():
        rows.append(
            ApplicationRow(
                applicant_code=int(series["Код поступающего"]),
                program=program,
                priority=int(series["Приоритет конкурса"]),
                score=float(series["Сумма баллов"]),
                rank=int(series["Порядковый номер"]),
                consent=_parse_consent(series["Подано согласие"]),
                status=str(series["Статус"]).strip(),
            )
        )
    return rows


def _normalize_col(name: object) -> str:
    return " ".join(str(name).replace("\n", " ").split())


def _read_budget_xlsx(path: Path, program: str) -> list[ApplicationRow]:
    """Читает конкурсный список из MAGREPORTS *_Budget.xlsx."""
    raw = pd.read_excel(path, header=None)
    header_idx = None
    for idx in range(min(15, len(raw))):
        joined = " ".join(str(x) for x in raw.iloc[idx].tolist())
        if "Уникальный код" in joined and "Приоритет" in joined:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"{path.name}: не найдена строка заголовков")

    header = [_normalize_col(c) for c in raw.iloc[header_idx].tolist()]
    data = raw.iloc[header_idx + 1 :].copy()
    data.columns = [
        header[i] if i < len(header) else f"col_{i}" for i in range(data.shape[1])
    ]
    # Пропускаем служебную строку с ~id~
    first_col = data.columns[0]
    data = data[data[first_col].apply(lambda x: str(x).strip().isdigit())]

    def find_col(*needles: str) -> str:
        for col in data.columns:
            low = col.lower()
            if all(n.lower() in low for n in needles):
                return col
        raise ValueError(f"{path.name}: нет колонки {needles}")

    code_col = find_col("уникальный код")
    rank_col = data.columns[0]
    try:
        pri_col = find_col("приоритет бюджет")
    except ValueError:
        pri_col = find_col("приоритет")
    score_candidates = [
        c
        for c in data.columns
        if "сумма конкурсных баллов" in c.lower() and "квот" not in c.lower()
    ]
    if not score_candidates:
        raise ValueError(f"{path.name}: нет колонки суммы баллов")
    score_col = score_candidates[0]
    status_col = find_col("статус")
    consent_col = find_col("согласие")

    rows: list[ApplicationRow] = []
    for _, series in data.iterrows():
        code_raw = series[code_col]
        if pd.isna(code_raw):
            continue
        rows.append(
            ApplicationRow(
                applicant_code=int(float(code_raw)),
                program=program,
                priority=_parse_priority(series[pri_col]),
                score=_parse_score(series[score_col]),
                rank=int(float(series[rank_col])),
                consent=_parse_consent(series[consent_col]),
                status=str(series[status_col]).strip(),
            )
        )
    return rows


def _finalize_profile(profile: ApplicantProfile) -> None:
    profile.consent = any(app.consent for app in profile.applications)
    priorities = [app.priority for app in profile.applications if app.priority < 10**6]
    if not priorities:
        profile.missing_higher_priority = False
        return
    profile.missing_higher_priority = min(priorities) > 1
    sorted_pri = sorted(set(priorities))
    for left, right in zip(sorted_pri, sorted_pri[1:]):
        if right - left > 1:
            profile.missing_higher_priority = True
            return


def build_profiles(rows: list[ApplicationRow]) -> dict[int, ApplicantProfile]:
    """Склеивает строки списков в профили по коду поступающего."""
    profiles: dict[int, ApplicantProfile] = {}
    for row in rows:
        profile = profiles.get(row.applicant_code)
        if profile is None:
            profile = ApplicantProfile(code=row.applicant_code)
            profiles[row.applicant_code] = profile
        profile.applications.append(row)

    for profile in profiles.values():
        _finalize_profile(profile)
    return profiles


def _status_allowed(status: str, allowed: frozenset[str] | set[str]) -> bool:
    """
    Проверяет статус, в т.ч. составной («A / B» из MAGREPORTS).

    Достаточно совпадения любой части после split по «/».
    """
    text = status.strip()
    if text in allowed:
        return True
    return any(part.strip() in allowed for part in text.split("/"))


def filter_profiles_by_status(
    profiles: dict[int, ApplicantProfile],
    *,
    include_pending: bool = True,
) -> dict[int, ApplicantProfile]:
    """Оставляет заявки с допустимым статусом (pending включён по умолчанию)."""
    allowed: set[str] = set(DEFAULT_ACTIVE_STATUSES)
    if include_pending:
        allowed |= PENDING_STATUSES

    filtered: dict[int, ApplicantProfile] = {}
    for code, profile in profiles.items():
        apps = [a for a in profile.applications if _status_allowed(a.status, allowed)]
        if not apps:
            continue
        new_profile = ApplicantProfile(
            code=code,
            applications=list(apps),
            consent=profile.consent,
        )
        _finalize_profile(new_profile)
        if profile.consent:
            new_profile.consent = True
        filtered[code] = new_profile
    return filtered


def load_dataset(
    data_dir: Path,
    seats_path: Path,
    *,
    include_pending: bool = True,
    campus: str | None = None,
) -> Dataset:
    """Загружает CSV / Budget.xlsx и seats.yaml (все кампусы по умолчанию)."""
    seats = load_seats(seats_path)
    files = list_data_files(data_dir, campus=campus)

    all_rows: list[ApplicationRow] = []
    programs: list[str] = []
    source_files: list[str] = []
    for path, program, kind in files:
        programs.append(program)
        source_files.append(path.name)
        if kind == "csv":
            all_rows.extend(_read_program_csv(path, program))
        else:
            all_rows.extend(_read_budget_xlsx(path, program))

    profiles = build_profiles(all_rows)
    profiles = filter_profiles_by_status(profiles, include_pending=include_pending)

    normalized_seats: dict[str, int | None] = {
        name: seats[name] if name in seats else None for name in programs
    }
    for name, value in seats.items():
        if name not in normalized_seats:
            normalized_seats[name] = value

    incomplete = sorted(
        code for code, profile in profiles.items() if profile.missing_higher_priority
    )
    return Dataset(
        applicants=profiles,
        seats=normalized_seats,
        programs=programs,
        source_files=source_files,
        incomplete_priority_codes=incomplete,
    )


def moscow_seats_from_summary(summary_xls: Path) -> dict[str, int]:
    """Словарь «название программы → K» из московской сводки (без суффикса кампуса)."""
    summary = pd.read_excel(summary_xls, header=3)
    scol, bcol = summary.columns[0], summary.columns[1]
    seats_map: dict[str, int] = {}
    for _, row in summary.iterrows():
        name = str(row[scol]).strip()
        if name.startswith(("Направление", "Всего", "Итого")) or name in {
            "Москва",
            "nan",
        }:
            continue
        value = row[bcol]
        if value is None or (isinstance(value, float) and pd.isna(value)):
            k = 0
        else:
            text = str(value).strip()
            k = 0 if text in {"-", "—", "–", ""} else int(float(text))
        seats_map[name] = k
    return seats_map


def build_seats_yaml_from_summary(
    summary_xls: Path,
    output_yaml: Path,
    *,
    programs_filter: set[str] | None = None,
) -> dict[str, int]:
    """Обратная совместимость: пишет только московские ключи без суффикса."""
    seats_map = moscow_seats_from_summary(summary_xls)
    programs: dict[str, dict[str, object]] = {}
    for name, k in seats_map.items():
        if programs_filter is not None and name not in programs_filter:
            continue
        if k <= 0 and programs_filter is None:
            continue
        programs[name] = {
            "seats": k,
            "title": name,
            "campus": "Москва",
            "notes": "из сводки поданных заявлений",
        }
    output_yaml.write_text(
        yaml.safe_dump(
            {"programs": programs},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {n: int(m["seats"]) for n, m in programs.items()}  # type: ignore[arg-type]
