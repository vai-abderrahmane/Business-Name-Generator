import re

BANNED_PATTERNS = [r"\b(fuck|shit|bitch)\b", r"\b(nazi|hitler)\b", r"\b(sex|xxx|porn)\b"]
REGEXES = [re.compile(p, re.IGNORECASE) for p in BANNED_PATTERNS]

def is_blocked(text: str) -> bool:
    return any(rx.search(text) for rx in REGEXES)
