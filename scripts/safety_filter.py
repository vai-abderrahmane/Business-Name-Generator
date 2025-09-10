import re
import sys

BANNED = [r"\b(fuck|shit|bitch)\b", r"\b(nazi|hitler)\b", r"\b(sex|xxx)\b"]

REGEXES = [re.compile(p, re.IGNORECASE) for p in BANNED]

def is_safe(text: str) -> bool:
    return not any(rx.search(text) for rx in REGEXES)

if __name__ == '__main__':
    sample = sys.argv[1] if len(sys.argv) > 1 else "Test"
    print("SAFE" if is_safe(sample) else "BLOCK")
