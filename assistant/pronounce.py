"""Rewrites text so the voices pronounce it well. Only used for speaking: captions keep the
original text.

- Words the voices get wrong get a fixed spelling ("Ridebly" -> "Ράιντμπλι" / "Ridebli").
- In Greek sentences, any other word in Latin letters ("booking", "online", "Google") is
  respelled in Greek letters from its English pronunciation. The Greek voice would otherwise
  read it with English sounds it never learned, which comes out garbled.
- In English sentences, Greek words are written in Latin letters ("Φώτης" -> "Fotis"); the
  English voice would otherwise spell them out letter by letter.
- Prices and times are written the way people say them ("€19,99" -> "19 ευρώ και 99 λεπτά",
  "09:00-23:00" -> "9 έως 23" / "9 am to 11 pm").
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from functools import lru_cache

# Fixed spellings, checked before anything else. Keys are matched as whole words, ignoring case.
SPELLINGS = {
    "el": {
        "ridebly": "Ράιντμπλι",
        "flarmio": "Φλάρμιο",
        "visionarrays": "Βίζιον Αρέις",
        "vision arrays": "Βίζιον Αρέις",
        "quickord": "Κουίκορντ",
        "roomismo": "Ρουμίσμο",
        "bid2board": "Μπιντ του Μπορντ",
        "rent a car": "ρέντακαρ",  # a lone "α" is read as the letter name
        "rent-a-car": "ρέντακαρ",
        "booking engine": "μπούκινγκ έντζιν",
        "booking form": "μπούκινγκ φορμ",
        "booking": "μπούκινγκ",
        "transfer": "τράνσφερ",
        "transfers": "τράνσφερ",
        "shuttle": "σάτλ",
        "online": "όνλαϊν",
        "site": "σάιτ",
        "website": "γουέμπσαϊτ",
        "link": "λινκ",
        "links": "λινκ",
        "email": "ίμεϊλ",
        "app": "απ",
        "qr code": "κιου αρ κόουντ",
        "qr": "κιου αρ",
        "google maps": "Γκούγκλ Μαπς",
        "google": "Γκούγκλ",
        "desmos": "Ντέσμος",
        "viva": "Βίβα",
        "nexi": "Νέξι",
        "anna": "Άννα",
        "mike": "Μάικ",
        "demo": "ντέμο",
        "wifi": "γουάιφάι",
        "ok": "οκέι",
        "okay": "οκέι",
        "pms": "πι εμ ες",
        "login": "λόγκιν",
        "log in": "λογκ ιν",
        "logout": "λογκάουτ",
        "manager": "μάνατζερ",
        "admin": "άντμιν",
        "dashboard": "ντάσμπορντ",
        "fleet": "φλιτ",
        "calendar": "κάλεντερ",
        "check-in": "τσεκ ιν",
        "check-out": "τσεκ άουτ",
        "channel manager": "τσάνελ μάνατζερ",
    },
    "en": {
        "ridebly": "Ridebli",
        "exarchos": "Eksarhos",
        "exarhos": "Eksarhos",
        "fotis": "Foh-tiss",
        "visionarrays": "Vision Arrays",
        "bid2board": "bid to board",
    },
}

# ---------- English pronunciation (espeak phonemes) -> Greek letters ----------

# Longest symbols first. Vowels map to (unstressed, stressed) spellings.
_VOWELS = {
    "aɪ": ("αϊ", "άι"), "aʊ": ("αου", "άου"), "eɪ": ("εϊ", "έι"), "ɔɪ": ("οϊ", "όι"),
    "oʊ": ("ο", "ό"), "əʊ": ("ο", "ό"), "ɪə": ("ια", "ία"), "eə": ("ερ", "έρ"), "ɛə": ("ερ", "έρ"),
    "ʊə": ("ουα", "ούα"), "ɜː": ("ερ", "έρ"), "ɜ": ("ερ", "έρ"), "ɚ": ("ερ", "έρ"),
    "ɑː": ("α", "ά"), "ɑ": ("α", "ά"), "a": ("α", "ά"), "æ": ("α", "ά"), "ʌ": ("α", "ά"), "ɐ": ("α", "ά"),
    "ə": ("ε", "έ"), "ɛ": ("ε", "έ"), "e": ("ε", "έ"),
    "iː": ("ι", "ί"), "i": ("ι", "ί"), "ɪ": ("ι", "ί"), "ᵻ": ("ι", "ί"), "ɨ": ("ι", "ί"),
    "ɔː": ("ο", "ό"), "ɔ": ("ο", "ό"), "ɒ": ("ο", "ό"), "o": ("ο", "ό"),
    "uː": ("ου", "ού"), "u": ("ου", "ού"), "ʊ": ("ου", "ού"),
}
_CONSONANTS = {
    "tʃ": "τσ", "dʒ": "τζ", "ŋk": "νκ", "ŋɡ": "νγκ", "ŋ": "νγκ",
    "p": "π", "b": "μπ", "t": "τ", "d": "ντ", "ɾ": "τ", "k": "κ", "ɡ": "γκ", "g": "γκ",
    "f": "φ", "v": "β", "θ": "θ", "ð": "δ", "s": "σ", "z": "ζ", "ʃ": "σ", "ʒ": "ζ",
    "h": "χ", "x": "χ", "m": "μ", "n": "ν", "l": "λ", "ɬ": "λ", "ɹ": "ρ", "r": "ρ",
    "w": "ου", "j": "ι",
}
_SYMBOLS = sorted([*_VOWELS, *_CONSONANTS], key=len, reverse=True)
_IGNORED = set("ːˌʲʔ-‿ ")


def phonemes_to_greek(phonemes: str) -> str:
    out: list[str] = []
    stress_next = False
    stressed = False  # Greek words carry one accent
    i = 0
    while i < len(phonemes):
        if phonemes[i] == "ˈ":
            stress_next = not stressed
            i += 1
            continue
        if phonemes[i] in _IGNORED:
            i += 1
            continue
        symbol = next((s for s in _SYMBOLS if phonemes.startswith(s, i)), None)
        if symbol is None:
            i += 1
            continue
        if symbol in _VOWELS:
            plain, accented = _VOWELS[symbol]
            out.append(accented if stress_next else plain)
            stressed |= stress_next
            stress_next = False
        else:
            out.append(_CONSONANTS[symbol])
        i += len(symbol)
    word = "".join(out)
    for doubled, single in (("νντ", "ντ"), ("μμπ", "μπ"), ("ννγκ", "νγκ"), ("σσ", "σ"), ("λλ", "λ")):
        word = word.replace(doubled, single)
    return word[:-1] + "ς" if word.endswith("σ") else word


# ---------- Greek letters -> Latin letters, for the English voice ----------

_GREEK_DIGRAPHS = [
    ("ου", "ou"), ("αι", "e"), ("ει", "i"), ("οι", "i"), ("υι", "i"), ("μπ", "b"), ("ντ", "d"), ("γκ", "g"),
    ("γγ", "ng"), ("τσ", "ts"), ("τζ", "tz"), ("αυ", "av"), ("ευ", "ev"),
]
_GREEK_LETTERS = dict(zip("αβγδεζηθικλμνξοπρσςτυφχψω", ["a", "v", "g", "th", "e", "z", "i", "th", "i", "k", "l", "m",
                                                          "n", "x", "o", "p", "r", "s", "s", "t", "i", "f", "h", "ps", "o"]))


def greek_to_latin(word: str) -> str:
    plain = "".join(c for c in unicodedata.normalize("NFD", word) if not unicodedata.combining(c))
    capital = plain[:1].isupper()
    text = plain.lower()
    for greek, latin in _GREEK_DIGRAPHS:
        text = text.replace(greek, latin)
    text = "".join(_GREEK_LETTERS.get(c, c) for c in text)
    return text.capitalize() if capital else text


# ---------- prices and times ----------

_PRICE = re.compile(r"€\s?(\d+)(?:[.,](\d{1,2}))?|(\d+)(?:[.,](\d{1,2}))?\s?€")
_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_RANGE = re.compile(r"(?<=\d)\s?[-–]\s?(?=\d)")


def _say_price(match: re.Match, language: str) -> str:
    whole = match.group(1) or match.group(3)
    cents = match.group(2) or match.group(4)
    cents_value = int(cents.ljust(2, "0")) if cents else 0
    if language == "el":
        return f"{int(whole)} ευρώ" + (f" και {cents_value} λεπτά" if cents_value else "")
    return f"{int(whole)} euros" + (f" {cents_value}" if cents_value else "")


def _say_time(match: re.Match, language: str) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    if language == "el":
        return f"{hour}" + (f" και {minute}" if minute else "")
    suffix = "am" if hour < 12 else "pm"
    return f"{hour % 12 or 12}" + (f":{minute:02d}" if minute else "") + f" {suffix}"


def _say_numbers(text: str, language: str) -> str:
    text = _PRICE.sub(lambda m: _say_price(m, language), text)
    text = _RANGE.sub(" έως " if language == "el" else " to ", text)  # before times turn "23:00" into "11 pm"
    return _TIME.sub(lambda m: _say_time(m, language), text)


# ---------- putting it together ----------

_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9']*")
_GREEK_WORD = re.compile(r"[Ͱ-Ͽἀ-῿]+")


def _spelling_pattern(words: dict[str, str]) -> re.Pattern:
    alternatives = sorted(words, key=len, reverse=True)  # "booking engine" before "booking"
    return re.compile(r"(?<!\w)(" + "|".join(re.escape(w) for w in alternatives) + r")(?!\w)", re.IGNORECASE)


_SPELLING_PATTERNS = {language: _spelling_pattern(words) for language, words in SPELLINGS.items()}


class Pronouncer:
    def __init__(self, english_phonemes: Callable[[str], str]) -> None:
        self._english_phonemes = english_phonemes

    def __call__(self, text: str, language: str) -> str:
        """`language` is the voice that will speak the text."""
        text = _say_numbers(text, language)
        if language == "en":
            text = _GREEK_WORD.sub(lambda m: greek_to_latin(m.group()), text)
        words = SPELLINGS[language]
        text = _SPELLING_PATTERNS[language].sub(lambda m: words[m.group().lower()], text)
        if language == "el":
            text = _LATIN_WORD.sub(lambda m: self._to_greek(m.group()), text)
        return text

    @lru_cache(maxsize=2048)
    def _to_greek(self, word: str) -> str:
        if word.lower() == "a":
            return "ε"  # a lone "α" would be read as the letter name
        greek = phonemes_to_greek(self._english_phonemes(word))
        return greek.capitalize() if word[:1].isupper() else greek
