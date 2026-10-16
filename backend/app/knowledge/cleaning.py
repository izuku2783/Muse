import re
import unicodedata
from collections.abc import Iterable


_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]+")


def clean_text(value: str | None) -> str | None:
    """Return stable, single-line Unicode text without control characters."""
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _CONTROL_CHARACTERS.sub(" ", normalized)
    cleaned = " ".join(normalized.split())
    return cleaned or None


def sorted_unique(values: Iterable[str | None]) -> list[str]:
    """Clean, deduplicate, and sort human-readable values deterministically."""
    unique: dict[str, str] = {}
    for value in values:
        cleaned = clean_text(value)
        if cleaned:
            unique.setdefault(cleaned.casefold(), cleaned)
    return sorted(unique.values(), key=lambda item: (item.casefold(), item))


def natural_join(values: Iterable[str]) -> str:
    items = list(values)
    if len(items) < 2:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"
