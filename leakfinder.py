"""LeakFinder engine: find recurring charges, price hikes, duplicates and fees
in a bank/card statement CSV. Pure pandas, runs in memory, no network calls.

Not financial advice - it only highlights patterns in the rows you give it.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Reading the file
# --------------------------------------------------------------------------

DELIMS = [",", ";", "\t", "|"]


def decode_bytes(data: bytes | str) -> str:
    if isinstance(data, str):
        return data
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _pick_delimiter(lines: list[str]) -> str:
    best, best_score = ",", -1.0
    sample = [l for l in lines[:60] if l.strip()]
    for d in DELIMS:
        counts = [len(next(csv.reader([l], delimiter=d))) for l in sample]
        if not counts:
            continue
        modal = max(set(counts), key=counts.count)
        if modal < 2:
            continue
        score = modal * (counts.count(modal) / len(counts))
        if score > best_score:
            best, best_score = d, score
    return best


def read_table(data: bytes | str) -> pd.DataFrame:
    """Parse messy bank CSV text into an all-string DataFrame with a header row."""
    text = decode_bytes(data)
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("The file is empty.")
    delim = _pick_delimiter(lines)
    rows = [r for r in csv.reader(lines, delimiter=delim)]
    wide = [r for r in rows if len(r) >= 3]
    if not wide:
        raise ValueError("Could not find a table with at least 3 columns.")
    lens = [len(r) for r in wide]
    modal = max(set(lens), key=lens.count)
    rows = [[c.strip() for c in r] for r in rows if len(r) == modal]
    if len(rows) < 2:
        raise ValueError("Could not find enough rows in the file.")
    header = rows[0]
    # if the first row looks like data (contains a parseable date), invent headers
    if any(_looks_like_date(c) for c in header):
        header = [f"col{i + 1}" for i in range(modal)]
        body = rows
    else:
        body = rows[1:]
    seen: dict[str, int] = {}
    cols = []
    for h in header:
        h = h or "col"
        seen[h] = seen.get(h, 0) + 1
        cols.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
    return pd.DataFrame(body, columns=cols)


def _looks_like_date(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}(\s.*)?", s.strip()))


# --------------------------------------------------------------------------
# Dates and numbers
# --------------------------------------------------------------------------

_DAY_FIRST = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y", "%d.%m.%y"]
_MONTH_FIRST = ["%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%m/%d/%y", "%m-%d-%y", "%m.%d.%y"]
_OTHER = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d", "%d %b %Y", "%d %B %Y",
          "%d-%b-%Y", "%d-%b-%y", "%b %d, %Y", "%b %d %Y", "%B %d, %Y", "%d/%b/%Y"]


def parse_dates(series: pd.Series, month_first: bool = False):
    """Return (parsed datetimes, format, rate, ambiguous)."""
    s = series.astype(str).str.strip()
    first_tok = s.str.split(r"[T ]", n=1, regex=True).str[0]
    order = _OTHER + (_MONTH_FIRST + _DAY_FIRST if month_first else _DAY_FIRST + _MONTH_FIRST)
    best = (None, None, 0.0)
    results = {}
    for fmt in order:
        for cand in (s, first_tok):
            p = pd.to_datetime(cand, format=fmt, errors="coerce")
            rate = p.notna().mean()
            if rate > best[2] + 1e-9:
                best = (p, fmt, rate)
            results[(fmt, id(cand))] = rate
    parsed, fmt, rate = best
    ambiguous = False
    if fmt in _DAY_FIRST or fmt in _MONTH_FIRST:
        twin = _MONTH_FIRST[_DAY_FIRST.index(fmt)] if fmt in _DAY_FIRST else _DAY_FIRST[_MONTH_FIRST.index(fmt)]
        for cand in (s, first_tok):
            if pd.to_datetime(cand, format=twin, errors="coerce").notna().mean() >= rate - 1e-9:
                ambiguous = True
    return parsed, fmt, rate, ambiguous


_CUR = re.compile(r"[^\d,.\-+()]")


def _decimal_style(values: pd.Series) -> str:
    """Return '.' or ',' as the decimal separator for a column of number strings."""
    v = values.astype(str).str.replace(r"[^\d,.]", "", regex=True)
    eu = v.str.contains(r",\d{1,2}$").sum() + v.str.contains(r"^\d{1,3}(?:\.\d{3})+,\d+$").sum()
    us = v.str.contains(r"\.\d{1,2}$").sum() + v.str.contains(r"^\d{1,3}(?:,\d{3})+\.\d+$").sum()
    return "," if eu > us else "."


def parse_number(x, dec: str = ".") -> float:
    if x is None:
        return np.nan
    s = str(x).strip()
    if not s or s in {"-", "--", "nan", "None"}:
        return np.nan
    up = s.upper()
    neg = ("(" in s and ")" in s) or "-" in s or up.endswith("DR") \
        or up.endswith(" D") or s.startswith("−")
    s = _CUR.sub("", s.replace("−", "-"))
    s = s.replace("(", "").replace(")", "").replace("+", "").replace("-", "")
    if not s:
        return np.nan
    if dec == ",":
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v


def parse_number_series(series: pd.Series) -> pd.Series:
    dec = _decimal_style(series)
    return series.map(lambda x: parse_number(x, dec)).astype(float)


# --------------------------------------------------------------------------
# Column detection
# --------------------------------------------------------------------------

DESC_HINTS = ["description", "merchant", "payee", "narrative", "details", "memo", "name",
              "reference", "text", "particulars", "transaction"]
DEBIT_HINTS = ["debit", "withdraw", "paid out", "money out", "outgoing", "spent", "charge"]
CREDIT_HINTS = ["credit", "deposit", "paid in", "money in", "incoming", "received"]
AMOUNT_HINTS = ["amount", "value", "sum", "amt"]


def _hint(col: str, hints: list[str]) -> int:
    c = col.lower()
    for i, h in enumerate(hints):
        if h in c:
            return i
    return -1


def detect_columns(df: pd.DataFrame) -> dict:
    """Guess which columns hold date, description and amount(s)."""
    cols = list(df.columns)
    date_rates = {}
    for c in cols:
        _, _, rate, _ = parse_dates(df[c])
        date_rates[c] = rate
    date_cands = [c for c in cols if date_rates[c] >= 0.8]
    date_cands.sort(key=lambda c: (_hint(c, ["transaction", "trans", "date"]) < 0,
                                   "post" in c.lower() or "value" in c.lower() or "settle" in c.lower(),
                                   cols.index(c)))
    date = date_cands[0] if date_cands else None

    num_rate = {}
    for c in cols:
        if c == date:
            continue
        nonblank = df[c].astype(str).str.strip().replace({"": np.nan}).dropna()
        if len(nonblank) == 0:
            continue
        parsed = parse_number_series(nonblank)
        # require it to look numeric (not a text with a stray digit)
        looks = nonblank.str.fullmatch(r"[^A-Za-z]*(?:\d[\d.,\s]*)[^A-Za-z]*|[^A-Za-z]*\d[\d.,]*\s?(?:CR|DR)?", case=False)
        num_rate[c] = (parsed.notna().mean() * looks.mean(), len(nonblank) / len(df))
    numeric = [c for c, (r, nb) in num_rate.items() if r >= 0.8 and nb > 0.02]
    numeric = [c for c in numeric if "balance" not in c.lower() and "saldo" not in c.lower()]

    debit = next((c for c in numeric if _hint(c, DEBIT_HINTS) >= 0 and _hint(c, CREDIT_HINTS) < 0), None)
    credit = next((c for c in numeric if _hint(c, CREDIT_HINTS) >= 0 and _hint(c, DEBIT_HINTS) < 0), None)
    amount = None
    if debit is None or credit is None:
        named = [c for c in numeric if _hint(c, AMOUNT_HINTS) >= 0]
        if named:
            amount = named[0]
        elif debit is not None:
            amount = None  # debit-only file
        elif numeric:
            amount = max(numeric, key=lambda c: num_rate[c][1])
        debit = debit if amount is None else None
        credit = credit if amount is None else None
        if amount is None and debit is None and numeric:
            amount = numeric[0]
    else:
        amount = None

    used = {date, amount, debit, credit}
    text_cols = [c for c in cols if c not in used and c not in numeric]
    desc = None
    hinted = sorted([c for c in text_cols if _hint(c, DESC_HINTS) >= 0], key=lambda c: _hint(c, DESC_HINTS))
    if hinted:
        desc = hinted[0]
    elif text_cols:
        desc = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
    return {"date": date, "desc": desc, "amount": amount, "debit": debit, "credit": credit,
            "spend_positive": None, "month_first": None}


def _spend_is_positive(amounts: pd.Series) -> bool:
    a = amounts.dropna()
    a = a[a != 0]
    if len(a) == 0:
        return False
    return (a > 0).mean() > 0.8


# --------------------------------------------------------------------------
# Merchant normalisation
# --------------------------------------------------------------------------

_STOP = {"COM", "WWW", "HTTP", "HTTPS", "NET", "ORG", "INC", "LLC", "LTD", "CO", "CORP", "PAYMENT",
         "PURCHASE", "POS", "CARD", "DEBIT", "VISA", "MASTERCARD", "RECURRING", "ONLINE", "PAYPAL", "SQ",
         "TST", "SP", "PP", "US", "USA", "UK", "GB", "EU", "THE", "TO", "FROM", "DD", "DIRECT", "DEBITS",
         "CHECKCARD", "AUTOPAY", "SUBSCRIPTION", "SUBSCR", "MEMBERSHIP", "BILL", "BILLING", "PMT", "ACH",
         "AB", "GMBH", "SA", "BV", "PLC", "SRL", "AG", "WEB", "ID", "REF", "ON", "AT", "FOR", "NL", "DE", "IE", "SE", "FR", "ES", "IT"}


def normalise_merchant(desc: str) -> str:
    s = str(desc).upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = [t for t in s.split() if not any(ch.isdigit() for ch in t) and t not in _STOP and len(t) > 1]
    if not toks:
        toks = [t for t in re.sub(r"[^A-Z ]", " ", s).split() if t not in _STOP] or ["UNKNOWN"]
    return " ".join(toks[:2])


def _merge_merchants(keys: pd.Series) -> pd.Series:
    names = set(keys.unique())
    singles = {n for n in names if " " not in n}

    def fix(n: str) -> str:
        first = n.split(" ")[0]
        return first if (" " in n and first in singles) else n

    return keys.map(fix)


def pretty(name: str) -> str:
    return name.title()


# --------------------------------------------------------------------------
# Building the transaction table
# --------------------------------------------------------------------------

def build_transactions(df: pd.DataFrame, mapping: dict):
    """Return (spend DataFrame, info dict). spend has date, description, amount (>0), merchant."""
    if not mapping.get("date"):
        raise ValueError("No date column found. Pick one under 'Columns look wrong?'.")
    if not (mapping.get("amount") or mapping.get("debit")):
        raise ValueError("No amount column found. Pick one under 'Columns look wrong?'.")
    dates, fmt, rate, ambiguous = parse_dates(df[mapping["date"]], bool(mapping.get("month_first")))
    desc = df[mapping["desc"]].astype(str) if mapping.get("desc") else pd.Series("UNKNOWN", index=df.index)
    info = {"date_format": fmt, "date_parse_rate": float(rate), "ambiguous_dates": ambiguous,
            "mode": "", "rows_total": len(df)}
    if mapping.get("debit"):
        deb = parse_number_series(df[mapping["debit"]]).abs()
        amt = deb.where(deb > 0)
        info["mode"] = "separate debit/credit columns"
    else:
        raw = parse_number_series(df[mapping["amount"]])
        sp = mapping.get("spend_positive")
        if sp is None:
            sp = _spend_is_positive(raw)
        amt = (raw if sp else -raw)
        amt = amt.where(amt > 0)
        info["mode"] = "single amount column (spending shown as %s numbers)" % ("positive" if sp else "negative")
        info["spend_positive"] = bool(sp)
    out = pd.DataFrame({"date": dates, "description": desc.str.strip(), "amount": amt})
    out = out.dropna(subset=["date", "amount"]).reset_index(drop=True)
    out["date"] = out["date"].dt.normalize()
    out["merchant"] = _merge_merchants(out["description"].map(normalise_merchant))
    info["rows_spend"] = len(out)
    if len(out) == 0:
        raise ValueError("No spending rows could be read. Check the column choices.")
    info["date_min"], info["date_max"] = out["date"].min(), out["date"].max()
    return out.sort_values("date", kind="stable").reset_index(drop=True), info


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------

FEE_RE = re.compile(
    r"\b(FEE|FEES|INTEREST|OVERDRAFT|OVERLIMIT|OVER LIMIT|ATM|FOREIGN|FX|LATE|PENALTY|NSF|"
    r"MAINTENANCE|COMMISSION|SERVICE CHARGE|CHARGE|SURCHARGE|CURRENCY CONVERSION|CONVERSION|"
    r"MONTHLY ACCOUNT|ACCOUNT FEE|RETURNED)\b", re.I)
INTEREST_RE = re.compile(r"\bINTEREST\b", re.I)
# words that look like a fee word but are not (e.g. "CHARGEPOINT" is not matched by \b anyway)
FEE_NEG = re.compile(r"\b(CHARGEBACK REFUND|TRANSFER FEE REFUND)\b", re.I)

CADENCES = [  # name, nominal days, (lo, hi) interval window, periods/year, min charges
    ("weekly", 7, (5, 9), 52, 4),
    ("every 2 weeks", 14, (12, 16), 26, 4),
    ("monthly", 30.4, (26, 35), 12, 3),
    ("quarterly", 91, (84, 98), 4, 3),
    ("yearly", 365, (350, 380), 1, 2),
]


ESSENTIAL_RE = re.compile(r"\b(?:RENT|MORTGAGE|LEASE|LOAN|TAX|TAXES|COUNCIL|PAYROLL|SALARY|UTILITIES|ELECTRIC|WATER|GAS)\b")


def _runs(amounts: list[float], tol: float = 0.03):
    runs = []
    for a in amounts:
        if runs and abs(a - runs[-1]["start"]) / runs[-1]["start"] <= tol:
            runs[-1]["n"] += 1
            runs[-1]["last"] = a
        else:
            runs.append({"start": a, "last": a, "n": 1})
    return runs


def find_recurring(spend: pd.DataFrame, end_date: pd.Timestamp):
    """Return (subscriptions DataFrame, variable-bills DataFrame, price-hikes DataFrame)."""
    subs, var, hikes = [], [], []
    for merchant, g in spend.groupby("merchant"):
        g = g.drop_duplicates(["date", "amount"]).sort_values("date")
        if len(g) < 2:
            continue
        gaps = g["date"].diff().dt.days.dropna().to_numpy(dtype=float)
        if len(gaps) == 0:
            continue
        med = float(np.median(gaps))
        match = None
        for name, nominal, (lo, hi), ppy, min_n in CADENCES:
            if lo <= med <= hi:
                # allow one skipped period (2x gap) to count as consistent
                ok = ((gaps >= lo) & (gaps <= hi)) | ((gaps >= 2 * lo) & (gaps <= 2 * hi))
                if ok.mean() >= 0.7 and len(g) >= min_n:
                    match = (name, nominal, ppy)
                break
        if not match:
            continue
        cadence, nominal, ppy = match
        amounts = g["amount"].tolist()
        dates = g["date"].tolist()
        runs = _runs(amounts)
        biggest = max(r["n"] for r in runs)
        fixed = len(runs) <= 3 and biggest / len(amounts) >= 0.5
        essential = bool(ESSENTIAL_RE.search(merchant))
        cv = float(np.std(amounts) / np.mean(amounts))
        if not fixed and (cv > 0.6 or len(g) < 4):
            continue
        last_gap = (end_date - dates[-1]).days
        active = last_gap <= nominal * 1.6 + 5
        latest = amounts[-1]
        typical = latest if fixed else float(np.mean(amounts))
        row = {
            "merchant": pretty(merchant), "cadence": cadence, "charges": len(g),
            "typical_amount": round(typical, 2), "yearly_cost": round(typical * ppy, 2),
            "first_seen": dates[0].date(), "last_seen": dates[-1].date(),
            "status": "active" if active else "may have ended",
            "amount_type": "fixed" if fixed else "varies",
        }
        if fixed and essential:
            row["amount_type"] = "essential bill"
            var.append(row)
            continue
        if fixed:
            # price hike detection: consecutive runs that go up
            pos = 0
            for i, r in enumerate(runs):
                if i > 0:
                    prev = runs[i - 1]
                    old, new = prev["last"], r["start"]
                    if new > old * 1.03 and (prev["n"] >= 2 or ppy == 1 or prev["n"] >= 1 and cadence == "yearly"):
                        hikes.append({
                            "merchant": pretty(merchant), "cadence": cadence, "old_price": round(old, 2),
                            "new_price": round(new, 2), "increase_pct": round((new / old - 1) * 100, 1),
                            "changed_on": dates[pos].date(), "extra_per_year": round((new - old) * ppy, 2),
                            "still_active": active,
                        })
                pos += r["n"]
        (subs if fixed else var).append(row)
    cols_s = ["merchant", "cadence", "charges", "typical_amount", "yearly_cost", "first_seen",
              "last_seen", "status", "amount_type"]
    cols_h = ["merchant", "cadence", "old_price", "new_price", "increase_pct", "changed_on",
              "extra_per_year", "still_active"]
    mk = lambda rows, cols, by: (pd.DataFrame(rows, columns=cols).sort_values(by, ascending=False)
                                 .reset_index(drop=True) if rows else pd.DataFrame(columns=cols))
    return mk(subs, cols_s, "yearly_cost"), mk(var, cols_s, "yearly_cost"), mk(hikes, cols_h, "extra_per_year")


def find_duplicates(spend: pd.DataFrame, window_days: int = 3, min_amount: float = 1.0) -> pd.DataFrame:
    rows = []
    for (merchant, amount), g in spend[spend["amount"] >= min_amount].groupby(["merchant", "amount"]):
        if len(g) < 2:
            continue
        g = g.sort_values("date")
        d = g["date"].tolist()
        desc = g["description"].tolist()
        for i in range(1, len(d)):
            gap = (d[i] - d[i - 1]).days
            if gap <= window_days:
                rows.append({"merchant": pretty(merchant), "amount": round(float(amount), 2),
                             "first_date": d[i - 1].date(), "second_date": d[i].date(),
                             "days_apart": gap, "description": desc[i]})
    cols = ["merchant", "amount", "first_date", "second_date", "days_apart", "description"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values("amount", ascending=False).reset_index(drop=True)


def find_fees(spend: pd.DataFrame, span_days: int):
    mask = spend["description"].map(lambda s: bool(FEE_RE.search(s)) and not FEE_NEG.search(s))
    fees = spend[mask].copy()
    cols = ["merchant", "kind", "charges", "total", "per_year_estimate", "last_seen"]
    if fees.empty:
        return mask, pd.DataFrame(columns=cols)
    fees["kind"] = np.where(fees["description"].map(lambda s: bool(INTEREST_RE.search(s))), "interest", "fee")
    factor = 365.0 / span_days if span_days >= 28 else 1.0
    g = fees.groupby(["merchant", "kind"]).agg(charges=("amount", "size"), total=("amount", "sum"),
                                               last_seen=("date", "max")).reset_index()
    g["total"] = g["total"].round(2)
    g["per_year_estimate"] = (g["total"] * factor).round(2)
    g["last_seen"] = g["last_seen"].dt.date
    g["merchant"] = g["merchant"].map(pretty)
    return mask, g[cols].sort_values("total", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------

@dataclass
class Result:
    spend: pd.DataFrame
    info: dict
    subscriptions: pd.DataFrame
    variable: pd.DataFrame
    hikes: pd.DataFrame
    duplicates: pd.DataFrame
    fees: pd.DataFrame
    review: pd.DataFrame
    findings: pd.DataFrame
    totals: dict = field(default_factory=dict)


def analyze(spend: pd.DataFrame, info: dict | None = None) -> Result:
    info = info or {}
    start, end = spend["date"].min(), spend["date"].max()
    span = int((end - start).days) + 1
    fee_mask, fees = find_fees(spend, span)
    core = spend[~fee_mask]
    subs, var, hikes = find_recurring(core, end)
    dups = find_duplicates(core)

    active = subs[subs["status"] == "active"] if len(subs) else subs
    sub_yearly = float(active["yearly_cost"].sum()) if len(active) else 0.0
    hike_extra = float(hikes.loc[hikes["still_active"], "extra_per_year"].sum()) if len(hikes) else 0.0
    dup_total = float(dups["amount"].sum()) if len(dups) else 0.0
    fee_total = float(fees["total"].sum()) if len(fees) else 0.0
    fee_year = float(fees["per_year_estimate"].sum()) if len(fees) else 0.0

    rows = []
    hike_by = {r.merchant: r for r in hikes.itertuples()} if len(hikes) else {}
    for r in subs.itertuples():
        h = hike_by.get(r.merchant)
        if r.status != "active":
            continue
        score = r.yearly_cost + (3 * h.extra_per_year if h else 0)
        why = f"{r.cadence} charge of {r.typical_amount:,.2f}; about {r.yearly_cost:,.2f} a year"
        if h:
            why += f"; price rose {h.increase_pct:.0f}% ({h.old_price:,.2f} -> {h.new_price:,.2f})"
        rows.append(("Subscription", r.merchant, why, r.yearly_cost, score,
                     "Still using it? Cancel, downgrade or switch to a cheaper plan."
                     if not h else "Price went up - check if it is still worth it or negotiate/switch."))
    for r in dups.itertuples():
        rows.append(("Possible duplicate", r.merchant,
                     f"{r.amount:,.2f} charged twice within {r.days_apart} day(s) "
                     f"({r.first_date} and {r.second_date})", r.amount, r.amount * 1.5,
                     "Check the receipt; ask the merchant or bank to reverse it if it was a mistake."))
    for r in fees.itertuples():
        rows.append((r.kind.title(), r.merchant,
                     f"{r.charges} charge(s) totalling {r.total:,.2f} (about {r.per_year_estimate:,.2f} a year)",
                     r.per_year_estimate, r.per_year_estimate * 1.2,
                     "Ask the bank to waive it or look for an account/card without this cost."))
    review = pd.DataFrame(rows, columns=["type", "item", "why", "money_at_stake", "score", "what_to_check"])
    review = review.sort_values("score", ascending=False).drop(columns="score").reset_index(drop=True)
    review.insert(0, "rank", range(1, len(review) + 1))
    review["money_at_stake"] = review["money_at_stake"].round(2)

    totals = {"total_spend": float(spend["amount"].sum()), "span_days": span, "start": start.date(),
              "end": end.date(), "n_subs": int(len(active)), "sub_yearly": round(sub_yearly, 2),
              "sub_monthly": round(sub_yearly / 12, 2), "hike_extra_year": round(hike_extra, 2),
              "n_hikes": int(len(hikes)), "n_dups": int(len(dups)), "dup_total": round(dup_total, 2),
              "fee_total": round(fee_total, 2), "fee_year": round(fee_year, 2),
              "n_ended": int((subs["status"] != "active").sum()) if len(subs) else 0}

    parts = []
    for name, df in (("subscription", subs), ("price_hike", hikes), ("possible_duplicate", dups), ("fee", fees)):
        if len(df):
            d = df.copy()
            d.insert(0, "finding_type", name)
            parts.append(d)
    findings = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"finding_type": []})
    return Result(spend, info, subs, var, hikes, dups, fees, review, findings, totals)


def summarize(res: Result, money=lambda x: f"{x:,.2f}") -> list[str]:
    t = res.totals
    out = [f"We read {res.info.get('rows_spend', len(res.spend))} spending transactions from {t['start']} to "
           f"{t['end']} ({t['span_days']} days), totalling {money(t['total_spend'])}."]
    if t["n_subs"]:
        out.append(f"We found {t['n_subs']} recurring charge(s) that look like subscriptions. Together they cost "
                   f"about {money(t['sub_monthly'])} a month, or {money(t['sub_yearly'])} a year.")
        top = res.subscriptions[res.subscriptions["status"] == "active"].head(3)
        out.append("The biggest: " + ", ".join(f"{r.merchant} ({money(r.yearly_cost)}/yr)" for r in top.itertuples()) + ".")
    else:
        out.append("We did not find clear recurring charges. Subscriptions need at least 3 similar charges "
                   "(2 for yearly ones) at regular intervals.")
    if t["n_hikes"]:
        h = res.hikes.iloc[0]
        out.append(f"{t['n_hikes']} recurring charge(s) went up in price, adding about {money(t['hike_extra_year'])} a year. "
                   f"Biggest: {h.merchant} ({money(h.old_price)} -> {money(h.new_price)}, +{h.increase_pct:.0f}%).")
    if t["n_dups"]:
        out.append(f"{t['n_dups']} possible duplicate charge(s) worth {money(t['dup_total'])} - worth a quick check.")
    if t["fee_total"]:
        out.append(f"Fees and interest came to {money(t['fee_total'])} in this period (about {money(t['fee_year'])} a year).")
    if t["n_ended"]:
        out.append(f"{t['n_ended']} recurring charge(s) have not appeared recently and may have ended (not counted above).")
    if not (t["n_hikes"] or t["n_dups"] or t["fee_total"] or t["n_subs"]):
        out.append("Nothing suspicious stood out.")
    return out


def run_pipeline(data: bytes | str, overrides: dict | None = None):
    """Convenience: bytes -> (Result, raw table, mapping). Raises ValueError with a readable message."""
    raw = read_table(data)
    mapping = detect_columns(raw)
    for k, v in (overrides or {}).items():
        mapping[k] = v
    spend, info = build_transactions(raw, mapping)
    return analyze(spend, info), raw, mapping


# --------------------------------------------------------------------------
# Synthetic demo statement
# --------------------------------------------------------------------------

def generate_demo(seed: int = 7, year: int = 2025) -> pd.DataFrame:
    """A fake 12-month statement (signed amounts, US format) with planted leaks."""
    rng = np.random.default_rng(seed)
    rows = []

    def add(date, desc, amount):
        rows.append((pd.Timestamp(date), desc, round(float(amount), 2)))

    for m in range(1, 13):
        add(f"{year}-{m:02d}-01", "PAYROLL ACME CORP", 3200.00)
        add(f"{year}-{m:02d}-02", "RENT - HARBOUR APARTMENTS", -1150.00)
        add(f"{year}-{m:02d}-05", "NETFLIX.COM 8842193", -(12.99 if m < 7 else 15.49))
        add(f"{year}-{m:02d}-08", "SPOTIFY P2A91C3", -10.99)
        add(f"{year}-{m:02d}-11", "PLANET FITNESS #4471", -24.99)
        add(f"{year}-{m:02d}-14", "APPLE.COM/BILL ICLOUD", -2.99)
        add(f"{year}-{m:02d}-09", "ADOBE *CREATIVE CLOUD 800-833", -54.99)
        add(f"{year}-{m:02d}-17", "PAYPAL *DROPBOX 4029357733", -11.99)
        add(f"{year}-{m:02d}-20", "CITY POWER & LIGHT", -rng.uniform(55, 120))
        add(f"{year}-{m:02d}-28", "MONTHLY ACCOUNT FEE", -4.50)
        if m in (3, 7, 10):
            add(f"{year}-{m:02d}-21", "FOREIGN TRANSACTION FEE", -rng.uniform(2, 6))
        if m in (4, 5, 6):
            add(f"{year}-{m:02d}-29", "OVERDRAFT INTEREST CHARGE", -rng.uniform(6, 15))
        for _ in range(int(rng.integers(5, 8))):
            d = int(rng.integers(1, 28))
            shop = rng.choice(["WHOLE FOODS MKT 1021", "TRADER JOES #552", "SHELL OIL 57444", "STARBUCKS STORE 08812",
                               "AMAZON MKTPLACE PMTS", "UBER *TRIP HELP.UBER.COM", "CHIPOTLE 2210", "TARGET 00012345"])
            add(f"{year}-{m:02d}-{d:02d}", f"POS DEBIT {shop}", -rng.uniform(4, 95))
    for w in range(0, 52, 1):
        add(pd.Timestamp(f"{year}-01-03") + pd.Timedelta(days=7 * w), "HELLOFRESH 8854321", -59.94 if w < 30 else -64.94)
    add(f"{year}-05-12", "UBER EATS PENDING 5521", -38.40)
    add(f"{year}-05-12", "UBER EATS PENDING 5521", -38.40)
    add(f"{year}-09-03", "BEST BUY 000912", -129.99)
    add(f"{year}-09-04", "BEST BUY 000912", -129.99)
    df = pd.DataFrame(rows, columns=["Date", "Description", "Amount"]).sort_values("Date", kind="stable")
    df["Date"] = df["Date"].dt.strftime("%m/%d/%Y")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        generate_demo().to_csv("demo_statement.csv", index=False)
        print("wrote demo_statement.csv")
