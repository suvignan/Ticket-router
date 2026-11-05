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

BERT_PLACEHOLDERS = {
    "redactedinfo": "masked",
    "redacteddate": "concealed",
    "redactedamount": "anonymous",
}


def first_350(text: str) -> str:
    """LSTM/BERT input: keep the first 350 words (Day 11: covers ~90% of complaints in full). Apply AFTER clean_text."""
    return " ".join(text.split()[:350])


def format_for_bert(text: str) -> str:
    """BERT-only: swap each placeholder for a single-WordPiece word. Apply AFTER clean_text."""
    for old, new in BERT_PLACEHOLDERS.items():
        text = text.replace(old, new)
    return text



if __name__ == "__main__":
    # Self-check: python src/text.py (silent = all passed)
    assert clean_text("I was CHARGED Twice") == "i was charged twice"
    assert clean_text("  too    many\n\nspaces  ") == "too many spaces"
    assert clean_text("     ") == ""
    assert clean_text(None) == ""
    assert clean_text("On XX/XX/2023 I paid {$50.00} to XXXX") == "on redacteddate i paid redactedamount to redactedinfo"
    assert format_for_bert("on redacteddate i paid redactedamount") == "on concealed i paid anonymous"
    print("clean_text: all checks passed")
