"""Точка входа CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from admission_sim import __version__
from admission_sim.report import render_cli_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="admission-sim",
        description=(
            "Локальный симулятор приоритетов поступления в магистратуру "
            "по CSV конкурсных списков."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw"),
        help="Каталог с CSV выгрузками (по умолчанию: data/raw)",
    )
    parser.add_argument(
        "--seats",
        type=Path,
        default=Path("seats.yaml"),
        help="YAML с числом бюджетных мест",
    )
    parser.add_argument(
        "--me",
        type=int,
        required=False,
        help="Ваш уникальный код поступающего",
    )
    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=0,
        help="Число прогонов Monte Carlo (0 — отключить)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.me is None:
        print("Укажите --me <код_поступающего>. Каркас CLI готов, логика — позже.")
        return 0
    print(render_cli_summary(applicant_code=args.me))
    print(f"Каталог данных: {args.data.resolve()}")
    print(f"Конфиг мест: {args.seats.resolve()}")
    if args.monte_carlo:
        print(f"Monte Carlo: {args.monte_carlo} прогонов (ещё не реализовано)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
