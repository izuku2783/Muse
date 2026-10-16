import re
import unicodedata


def normalize_text(value: str) -> str:
    """Normalize text for conservative, case-insensitive identity comparisons."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", value).split())
