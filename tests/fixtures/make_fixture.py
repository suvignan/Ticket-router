"""Builds tests/fixtures/mini_cfpb.csv (+ params_fixture.yaml): a tiny SYNTHETIC raw file where every
rule of every stage fires. Deterministic (no randomness). Re-run after changing it, commit both outputs.
    python tests/fixtures/make_fixture.py"""
import csv
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Same 18 columns, same order as the real CFPB file
COLUMNS = ["Date received", "Product", "Sub-product", "Issue", "Sub-issue", "Consumer complaint narrative",
           "Company public response", "Company", "State", "ZIP code", "Tags", "Consumer consent provided?",
           "Submitted via", "Date sent to company", "Company response to consumer", "Timely response?",
           "Consumer disputed?", "Complaint ID"]

OLD_CREDIT_REPORTING = "Credit reporting, credit repair services, or other personal consumer reports"
OLD_PAYDAY = "Payday loan, title loan, or personal loan"
OLD_CARD = "Credit card or prepaid card"

# label -> (sub-product, keyword phrase). Synthetic text with clear class keywords: no real complaints
CLASSES = {
    "Checking or savings account": ("Checking account", "my checking account overdraft fee at the branch"),
    "Credit card": ("General-purpose credit card or charge card", "my credit card annual fee and interest charge"),
    "Credit reporting or other personal consumer reports": ("Credit reporting", "the credit bureau report shows an inaccurate late payment"),
    "Debt collection": ("Other debt", "a debt collector keeps calling about a debt i do not owe"),
    "Debt or credit management": ("Debt settlement", "the debt settlement company took fees and did nothing"),
    "Money transfer, virtual currency, or money service": ("Mobile or digital wallet", "my zelle transfer never arrived at the recipient"),
    "Mortgage": ("Conventional home mortgage", "my mortgage escrow payment was calculated wrong"),
    "Payday loan, title loan, personal loan, or advance loan": ("Installment loan", "the payday lender charged a huge loan rollover fee"),
    "Prepaid card": ("General-purpose prepaid card", "my prepaid card balance vanished after reload"),
    "Student loan": ("Federal student loan servicing", "my student loan servicer lost my repayment plan form"),
    "Vehicle loan or lease": ("Loan", "the auto dealer repossessed my car on the vehicle loan"),
}
VARIANTS = ["in january", "again this week", "for the third time", "after i called twice", "with no explanation"]

rows = []


def add(date, product, sub, text, why):
    """date as MM/DD/YY (the real format). `why` is only for the group table, not written to the CSV."""
    rows.append({"Date received": date, "Product": product, "Sub-product": sub, "Consumer complaint narrative": text,
                 "Company": "SYNTHETIC BANK", "Submitted via": "Web", "Complaint ID": 1000 + len(rows), "_why": why})


# --- base: all 11 labels. 3 in train (2024), 1 in val (Feb 2025), 1 in test (Apr 2025)
for label, (sub, phrase) in CLASSES.items():
    for k, date in enumerate(["03/10/24", "06/12/24", "09/14/24"]):
        add(date, label, sub, f"{phrase} {VARIANTS[k]}", "base: train, 3 per label")
    add("02/10/25", label, sub, f"{phrase} {VARIANTS[3]}", "base: val, 1 per label")
    add("04/10/25", label, sub, f"{phrase} {VARIANTS[4]}", "base: test, 1 per label")

# --- renames: the 2 OLD product names (pre 25 Aug 2023) -> new names
add("03/15/23", OLD_CREDIT_REPORTING, "Credit reporting", "old name the credit bureau report has a wrong address", "rename: old credit-reporting name")
add("05/10/23", OLD_PAYDAY, "Personal loan", "old name the payday lender added a surprise loan fee", "rename: old payday name")

# --- card split: old combined card product, split by Sub-product
add("06/05/23", OLD_CARD, "Store credit card", "old card the store credit card charged interest twice", "card split: credit sub-product -> Credit card")
add("07/05/23", OLD_CARD, "Gift card", "old card the gift card was empty when i tried to use it", "card split: prepaid sub-product -> Prepaid card")

# --- credit-repair move: old credit-reporting name + "Credit repair services" -> Debt or credit management
add("04/20/23", OLD_CREDIT_REPORTING, "Credit repair services", "the credit repair company charged upfront and fixed nothing", "credit-repair move (after rename)")

# --- junk: removed by ingestion BEFORE anything else
add("08/01/24", "Debt collection", "Other debt", "", "junk: NaN narrative")
add("08/02/24", "Debt collection", "Other debt", "XXXX XXXX XX/XX/XXXX", "junk: redaction-only")
add("08/03/24", "Mortgage", "Conventional home mortgage", "See attached.", "junk: short 'see attach'")

# --- cutoff: on/after 2025-06-01 is dropped
add("06/15/25", "Mortgage", "Conventional home mortgage", "my mortgage servicer misplaced my escrow check in june", "cutoff: after 2025-06-01, dropped")

# --- split boundaries (half-open): the boundary day belongs to the LATER split
add("12/31/24", "Student loan", "Federal student loan servicing", "my student loan servicer charged me on new years eve", "boundary: last train day")
add("01/01/25", "Student loan", "Federal student loan servicing", "my student loan servicer charged me on new years day", "boundary: exactly val_start -> val")
add("03/01/25", "Student loan", "Federal student loan servicing", "my student loan servicer charged me on the first of march", "boundary: exactly test_start -> test")

# --- train de-dup, same label repeated: 3 copies -> 1 survives
DUP = "my checking account was frozen without any notice"
for date in ["02/01/24", "02/02/24", "02/03/24"]:
    add(date, "Checking or savings account", "Checking account", DUP, "dedup: same text + same label x3 -> 1")

# --- train de-dup, 2-vs-2 TIE: Credit card vs Vehicle loan -> A-Z wins: Credit card
TIE = "they charged my card for a car payment i never authorized"
for date, label, sub in [("05/01/24", "Credit card", "General-purpose credit card or charge card"),
                         ("05/02/24", "Vehicle loan or lease", "Loan"),
                         ("05/03/24", "Credit card", "General-purpose credit card or charge card"),
                         ("05/04/24", "Vehicle loan or lease", "Loan")]:
    add(date, label, sub, TIE, "tie: 2 x Credit card vs 2 x Vehicle loan -> Credit card (A-Z)")

# --- test_dedup: a test text also seen in train is removed
add("04/20/25", "Checking or savings account", "Checking account", DUP, "test_dedup: test text seen in train -> removed")

# --- test_dedup: exact copy inside test (text + label) -> lowest Complaint ID survives
COPY = "my prepaid card was declined at every store this month"
for date in ["05/05/25", "05/06/25"]:
    add(date, "Prepaid card", "General-purpose prepaid card", COPY, "test_dedup: exact copy in test -> lowest ID kept")

# Bad line: text right after a closing quote ("..."x). With usecols + engine="python" this is the shape pandas
# SKIPS; extra or missing fields are silently ACCEPTED (tested 2026-11-05), so they'd prove nothing here
BAD_LINE = '01/15/24,Mortgage,Conventional home mortgage,,,"bad line"broken quote,,SYNTHETIC BANK,,,,,Web,,,,,1999\n'

if __name__ == "__main__":
    with open(HERE / "mini_cfpb.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        f.write(BAD_LINE)                # -> skipped + counted as 1 bad line (max_bad_lines=10)

    # params_fixture.yaml = the real params.yaml + 3 overrides (generated, so it can't drift from the real one)
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    params["model_trainer"]["tfidf"]["min_df"] = 1                # 70 rows: min_df=5 would delete most words
    params["model_evaluation"]["min_macro_f1"] = 0.0              # smoke test checks the chain RUNS, not quality
    params["model_evaluation"]["champion"] = None                 # no frozen champion in CI -> floor only
    (HERE / "params_fixture.yaml").write_text(
        "# GENERATED by tests/fixtures/make_fixture.py from params.yaml: do not edit by hand\n"
        + yaml.safe_dump(params, sort_keys=False, allow_unicode=True), encoding="utf-8")

    groups = {}
    for r in rows:
        groups[r["_why"]] = groups.get(r["_why"], 0) + 1
    for why, n in groups.items():
        print(f"{n:3}  {why}")
    print(f"{len(rows):3}  rows + 1 bad line")
