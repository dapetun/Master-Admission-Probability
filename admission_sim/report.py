"""Формирование отчёта для терминала и Markdown."""

from __future__ import annotations


def render_cli_summary(*, applicant_code: int) -> str:
    """Краткая текстовая сводка (заглушка)."""
    return (
        f"Отчёт для кода {applicant_code} пока не сформирован: "
        "логика симуляции ещё не подключена."
    )


def write_markdown_report(path: str, content: str) -> None:
    """Пишет Markdown-отчёт на диск."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
