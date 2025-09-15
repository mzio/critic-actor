"""
Score GPQA responses
"""
import re
from typing import Iterable, Optional, Tuple

CHOICE_LINE = re.compile(r'^\s*\(([A-Z])\)\s', re.MULTILINE)


def extract_allowed_letters(question_text: str) -> Tuple[str, ...]:
    """
    Get allowed letters (A-D/E/…) from the prompt itself.
    """
    # Matches lines like "(A) ...", "(B) ...", etc.
    letters = tuple(dict.fromkeys(CHOICE_LINE.findall(question_text)))
    # Fallback if nothing found: default to A–E
    return letters or tuple("ABCDE")


def parse_model_choice(output_text: str, allowed: Iterable[str]) -> Optional[str]:
    """
    Parse model output for the chosen option, using increasingly liberal rules.
    """
    allowed_set = set(allowed)

    # Preferred: explicit “answer … (X)” style, case-insensitive.
    p1 = re.compile(
        r'(?i)\b(?:the\s+)?(?:correct\s+)?(?:final\s+)?(?:answer|choice|option)'
        r'(?:\s+is|:)?\s*\(?\s*([A-Z])\s*\)?'
    )
    m = None
    matches = p1.findall(output_text)
    if matches:
        # take the last such declaration in case the model changes its mind
        last = matches[-1].upper()
        if last in allowed_set:
            return last

    # Next: parenthesized letter at the very end (e.g., “… Therefore (C)”).
    m = re.search(r'\(([A-Z])\)\s*$', output_text.strip())
    if m and m.group(1) in allowed_set:
        return m.group(1)

    # Next: any parenthesized letter in the last 2–3 lines (avoid equations above).
    tail = "\n".join(output_text.strip().splitlines()[-3:])
    tail_matches = re.findall(r'\(([A-Z])\)', tail)
    for ch in reversed([c.upper() for c in tail_matches]):
        if ch in allowed_set:
            return ch

    # Last resort: bold/markdown around (X)
    m = re.search(r'\*?\(([A-Z])\)\*?', tail)
    if m and m.group(1) in allowed_set:
        return m.group(1)

    return None


def score_response(question: str, correct_answer: str, response: str) -> bool:
    """
    Check correctness
    -> Return True if the model output is correct, False otherwise
    """
    allowed = extract_allowed_letters(question)
    pred = parse_model_choice(response, allowed)
    return (pred is not None) and (pred.upper() == correct_answer.upper())
