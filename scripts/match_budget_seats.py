"""Разовая проверка: имена из Budget.xlsx ↔ сводка мест."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TITLE_RE = re.compile(r'Образовательная программа "([^"]+)"')


def program_from_xlsx(path: Path) -> str | None:
    """Достаёт название программы из первой ячейки Budget.xlsx."""
    df = pd.read_excel(path, header=None, nrows=1)
    title = str(df.iloc[0, 0])
    match = TITLE_RE.search(title)
    return match.group(1) if match else None


def seats_val(value: object) -> int:
    """Нормализует значение мест из сводки."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    text = str(value).strip()
    if text in {"-", "—", "–", "", "nan"}:
        return 0
    return int(float(text))


def main(argv: list[str] | None = None) -> int:
    """Сравнивает имена Budget.xlsx со сводкой; пишет локальный _match_report.txt."""
    parser = argparse.ArgumentParser(description="Сверка Budget.xlsx со сводкой мест")
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Локальный .xls со сводкой мест",
    )
    args = parser.parse_args(argv)

    names = [(f.name, program_from_xlsx(f)) for f in sorted(RAW.glob("*_Budget.xlsx"))]
    summary = pd.read_excel(args.summary, header=3)
    scol, bcol = summary.columns[0], summary.columns[1]
    seats: dict[str, int] = {}
    for _, row in summary.iterrows():
        name = str(row[scol]).strip()
        if name.startswith(("Направление", "Всего", "Итого")) or name in {
            "Москва",
            "nan",
        }:
            continue
        seats[name] = seats_val(row[bcol])

    matched = [(fname, pname, seats.get(pname) if pname else None) for fname, pname in names]
    ok = [x for x in matched if x[1] and x[2] and x[2] > 0]
    zero = [x for x in matched if x[1] and x[2] == 0]
    miss = [x for x in matched if x[1] and x[2] is None]
    none_name = [x for x in matched if not x[1]]

    out = RAW / "_match_report.txt"
    with out.open("w", encoding="utf-8") as fh:
        fh.write(
            f"budget={len(names)} ok={len(ok)} zero={len(zero)} "
            f"miss={len(miss)} noname={len(none_name)}\n\n"
        )
        fh.write("MISS\n")
        for item in miss:
            fh.write(f"{item}\n")
        fh.write("\nZERO\n")
        for item in zero:
            fh.write(f"{item}\n")
        fh.write("\nOK sample\n")
        for item in ok[:20]:
            fh.write(f"{item}\n")
    print(f"ok={len(ok)} zero={len(zero)} miss={len(miss)} noname={len(none_name)}")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
