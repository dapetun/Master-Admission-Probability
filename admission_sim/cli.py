"""Точка входа CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt

from admission_sim import __version__
from admission_sim.pipeline import run_analysis
from admission_sim.report import build_markdown_report, print_cli_report, write_markdown_report


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
        default=None,
        help="Каталог с CSV (по умолчанию: data/raw, в интерактиве можно изменить)",
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
        default=None,
        help="Ваш уникальный код поступающего (если не указан — спросим)",
    )
    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=None,
        help="Число прогонов Monte Carlo (по умолчанию в интерактиве: 500)",
    )
    parser.add_argument(
        "--campus",
        type=str,
        default="",
        help="Фильтр кампуса для Budget.xlsx (пусто = весь вуз)",
    )
    parser.add_argument(
        "--scenario",
        choices=["auto", "balanced", "optimistic", "pessimistic"],
        default="auto",
        help="Сценарий согласий прочих (по умолчанию auto)",
    )
    parser.add_argument(
        "--consent-p",
        type=float,
        default=None,
        help="Ручной override P(согласие); игнорирует --scenario",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для Monte Carlo",
    )
    parser.add_argument(
        "--include-pending",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Учитывать pending («Ожидание результатов ВИ»). "
            "По умолчанию включено; отключить: --no-include-pending"
        ),
    )
    parser.add_argument(
        "--threats",
        type=int,
        default=30,
        help="Сколько контрфактов-угроз показать",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("report.md"),
        help="Куда сохранить Markdown-отчёт",
    )
    parser.add_argument(
        "--no-report-file",
        action="store_true",
        help="Не писать report.md",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Без интерактивных вопросов (нужен --me)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _resolve_data_dir(raw: Path | None) -> Path:
    if raw is not None:
        return raw
    default = Path("data/raw")
    if default.is_dir() and any(default.glob("*.csv")):
        return default
    cwd_csv = list(Path(".").glob("*.csv"))
    if cwd_csv:
        return Path(".")
    return default


def _interactive_params(args: argparse.Namespace, console: Console) -> argparse.Namespace:
    console.print("[bold]Симулятор поступления[/bold] — ввод параметров\n")

    data = _resolve_data_dir(args.data)
    data_str = Prompt.ask("Каталог с CSV", default=str(data))
    args.data = Path(data_str)

    if not args.seats.exists() and Path("seats.example.yaml").exists():
        console.print(
            f"[yellow]Нет {args.seats}[/yellow] — скопируйте seats.example.yaml → seats.yaml"
        )

    while True:
        code = IntPrompt.ask("Ваш код поступающего")
        if code > 0:
            args.me = code
            break
        console.print("[red]Код должен быть положительным числом[/red]")

    if args.monte_carlo is None:
        if Confirm.ask("Запустить Monte Carlo?", default=True):
            args.monte_carlo = IntPrompt.ask("Число прогонов", default=500)
        else:
            args.monte_carlo = 0

    args.include_pending = Confirm.ask(
        "Учитывать «Ожидание результатов ВИ» / pending?",
        default=args.include_pending,
    )
    return args


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console(stderr=True)
    console.print(
        "[dim]Дисклеймер: неофициальный локальный симулятор. "
        "Не расчёт ПК/вуза/Госуслуг, не гарантия зачисления. "
        "Лицензия: MIT.[/dim]\n"
    )

    if args.me is None:
        if args.yes:
            console.print("[red]Нужен --me или интерактивный режим без --yes[/red]")
            return 2
        args = _interactive_params(args, console)
    else:
        args.data = _resolve_data_dir(args.data)
        if args.monte_carlo is None:
            args.monte_carlo = 0

    try:
        result = run_analysis(
            args.data,
            args.seats,
            args.me,
            include_pending=args.include_pending,
            campus=args.campus or None,
            monte_carlo=args.monte_carlo,
            scenario=args.scenario,  # type: ignore[arg-type]
            consent_p=args.consent_p,
            seed=args.seed,
            threats=args.threats,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]Ошибка загрузки:[/red] {exc}")
        return 1
    except ValueError as exc:
        console.print(f"[red]Ошибка данных:[/red] {exc}")
        return 1
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        return 1

    me = result.dataset.applicants[result.my_code]
    my_set = {app.program for app in me.applications}
    my_zero = [p for p in result.zero_seat_programs if p in my_set]
    my_unknown = [p for p in result.unknown_seat_programs if p in my_set]
    other_unknown = len(result.unknown_seat_programs) - len(my_unknown)
    if my_zero:
        console.print(
            "[yellow]Внимание:[/yellow] нулевые места у ваших программ: "
            + ", ".join(my_zero)
        )
    if my_unknown:
        console.print(
            "[yellow]Без КЦП у ваших программ:[/yellow] "
            + ", ".join(my_unknown)
            + " — поглощаются как EXTERNAL."
        )
    elif other_unknown:
        console.print(
            f"[dim]Без КЦП у прочих конкурсов: {other_unknown} "
            "(на ваш паспорт не влияет напрямую).[/dim]"
        )

    print_cli_report(
        result.dataset,
        result.my_code,
        vpp=result.vpp,
        vpp_if_consent=result.vpp_if_consent,
        ovp=result.ovp,
        counterfactuals=result.counterfactuals,
        probability=result.probability,
        include_pending=result.include_pending,
        console=Console(),
    )

    if not args.no_report_file:
        md = build_markdown_report(
            result.dataset,
            result.my_code,
            vpp=result.vpp,
            vpp_if_consent=result.vpp_if_consent,
            ovp=result.ovp,
            counterfactuals=result.counterfactuals,
            probability=result.probability,
            include_pending=result.include_pending,
        )
        write_markdown_report(args.report, md)
        console.print(f"[green]Отчёт сохранён:[/green] {args.report.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
