"""Загрузка CSV конкурсных списков и конфигурации мест."""

from __future__ import annotations

from pathlib import Path


def program_name_from_csv(path: Path) -> str:
    """Извлекает название программы из имени файла выгрузки."""
    stem = path.name
    # Имя до суффикса .YYYY-MM-DD_...
    if ".20" in stem:
        return stem.split(".20", 1)[0]
    return path.stem


def load_seats(path: Path) -> dict[str, int]:
    """Читает число бюджетных мест из YAML."""
    raise NotImplementedError("Загрузка seats.yaml будет реализована позже.")


def load_csv_directory(directory: Path) -> list[Path]:
    """Возвращает список CSV в каталоге (без чтения содержимого)."""
    return sorted(directory.glob("*.csv"))
