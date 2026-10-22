"""The two languages the assistant speaks, and telling them apart by alphabet."""

from __future__ import annotations

import re

SUPPORTED_LANGUAGES = ("el", "en")
LANGUAGE_NAMES = {"el": "Greek", "en": "English"}

_GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
_LATIN = re.compile(r"[A-Za-z]")


def text_language(text: str, fallback: str) -> str:
    """'el' if the text is mostly Greek letters, 'en' if mostly Latin, else the fallback."""
    greek, latin = len(_GREEK.findall(text)), len(_LATIN.findall(text))
    if greek == latin == 0:
        return fallback
    return "el" if greek >= latin else "en"
