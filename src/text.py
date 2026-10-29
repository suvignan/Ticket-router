import re


def clean_text(text: str) -> str:
    """Replace redactions with placeholder tokens, lowercase, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    # Redactions BEFORE .lower(), because the patterns use capital X.
    # Dates first (with or without a visible year), so XXXX doesn't break the date pattern.
    # Spaces around placeholders stop X-runs gluing onto them.
    text = re.sub(r"XX/XX/(XXXX|\d{4})", " redacteddate ", text)
    text = re.sub(r"\{\$[\d,.]+\}", " redactedamount ", text)
    text = re.sub(r"\bX{2,}\b", " redactedinfo ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


if __name__ == "__main__":
    # Self-check: python src/text.py (silent = all passed)
    assert clean_text("I was CHARGED Twice") == "i was charged twice"
    assert clean_text("  too    many\n\nspaces  ") == "too many spaces"
    assert clean_text("     ") == ""
    assert clean_text(None) == ""
    assert clean_text("On XX/XX/2023 I paid {$50.00} to XXXX") == "on redacteddate i paid redactedamount to redactedinfo"
    print("clean_text: all checks passed")
