"""Базовые тесты (логика симуляции подключится позже)."""

from pathlib import Path

from admission_sim.load import program_name_from_csv


def test_program_name_from_csv() -> None:
    path = Path("Математика_машинного обучения.2026-08-16_20-17-34.csv")
    assert program_name_from_csv(path) == "Математика_машинного обучения"
