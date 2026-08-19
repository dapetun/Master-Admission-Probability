"""Безопасная работа с путями записи внутри доверенных корней."""

from __future__ import annotations

from pathlib import Path


def resolve_safe_path(
    raw_path: Path | str,
    *,
    allowed_roots: list[Path] | tuple[Path, ...],
) -> Path:
    """Возвращает абсолютный путь внутри allowlist-корней или бросает ValueError."""
    target = Path(raw_path).expanduser()
    resolved_target = target.resolve()
    resolved_roots = [root.expanduser().resolve() for root in allowed_roots]
    if not resolved_roots:
        raise ValueError("Не задан allowlist корней для записи")
    for root in resolved_roots:
        if resolved_target == root or resolved_target.is_relative_to(root):
            return resolved_target
    roots = ", ".join(str(root) for root in resolved_roots)
    raise ValueError(
        f"Запись разрешена только внутри: {roots}; получен путь: {resolved_target}"
    )


def ensure_not_overwriting(target: Path, *, force: bool = False) -> None:
    """Запрещает перезапись существующего файла без явного force."""
    if target.exists() and not force:
        raise ValueError(
            f"Файл уже существует: {target}. "
            "Для перезаписи используйте явный флаг force."
        )
