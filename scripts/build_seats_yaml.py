"""Собрать seats.yaml по Budget.xlsx / CSV в data/raw."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from admission_sim.load import (
    list_data_files,
    load_seats,
    moscow_seats_from_summary,
    program_and_campus_from_budget_xlsx,
    program_key,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "seats.yaml"

# Известные КЦП вне сводки (уточняйте по официальным карточкам программ).
KNOWN_OVERRIDES: dict[str, int] = {}


def main(argv: list[str] | None = None) -> int:
    """Собирает локальный seats.yaml; пример seats.example.yaml не перезаписывает."""
    parser = argparse.ArgumentParser(
        description="Сборка seats.yaml из выгрузок в data/raw"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Опциональный .xls со сводкой мест (локальный файл, не в git)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="Куда писать seats.yaml (по умолчанию ./seats.yaml)",
    )
    args = parser.parse_args(argv)

    files = list_data_files(RAW, campus=None)
    moscow_bare: dict[str, int] = {}
    if args.summary is not None and args.summary.is_file():
        moscow_bare = moscow_seats_from_summary(args.summary)

    programs: dict[str, dict[str, object]] = {}
    for path, key, kind in files:
        if kind == "csv":
            bare, campus = key, "CSV"
        else:
            bare, campus = program_and_campus_from_budget_xlsx(path)
            key = program_key(bare, campus)

        seats: int | None
        notes: str
        if key in KNOWN_OVERRIDES:
            seats = KNOWN_OVERRIDES[key]
            notes = "ручной override (подтвердите по официальной карточке)"
        elif campus == "Москва" and bare in moscow_bare:
            seats = moscow_bare[bare]
            notes = f"из сводки {args.summary.name}" if args.summary else "из сводки"
        else:
            seats = None
            notes = "КЦП неизвестны — в модели программа поглощает как EXTERNAL"

        programs[key] = {
            "seats": seats,
            "title": bare,
            "campus": campus,
            "notes": notes,
        }

    payload = {"programs": programs}
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    args.out.write_text(text, encoding="utf-8")

    loaded = load_seats(args.out)
    known = sum(1 for v in loaded.values() if v is not None)
    positive = sum(1 for v in loaded.values() if isinstance(v, int) and v > 0)
    unknown = sum(1 for v in loaded.values() if v is None)
    print(
        f"программ: {len(loaded)}; с известным K: {known}; "
        f"K>0: {positive}; неизвестно: {unknown}"
    )
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
