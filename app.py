"""LeakFinder - find money leaks in a bank/card statement. Runs in memory, stores nothing."""
import pandas as pd
import streamlit as st

import leakfinder as lf

st.set_page_config(page_title="LeakFinder - find money leaks in your statement", page_icon="🔎", layout="wide")

st.title("🔎 LeakFinder")
st.subheader("Upload a bank or card statement CSV. See forgotten subscriptions, price hikes, double charges and fees.")
st.caption("🔒 Private by design: no bank login, your file is processed in memory in this session, never saved, "
           "and no data is sent to any AI service. Works with CSV exports from most banks worldwide.")

with st.sidebar:
    st.header("Options")
    cur = st.text_input("Currency symbol (optional)", value="", max_chars=4, key="cur",
                        help="Only used for display, e.g. $, EUR, GBP.")
    st.markdown("**How do I get my CSV?** In your online banking or card app, look for *Export* or *Download "
                "transactions* and choose CSV. 6-12 months works best.")
    st.markdown("Not financial advice. LeakFinder only highlights patterns; check anything it flags yourself.")


def money(x: float) -> str:
    return f"{cur}{x:,.2f}"


use_demo = st.checkbox("Use demo data (a fake 12-month statement)", key="use_demo")
uploaded = None if use_demo else st.file_uploader("Statement CSV", type=["csv", "txt"], key="upload")

data = None
if use_demo:
    data = lf.generate_demo().to_csv(index=False).encode("utf-8")
elif uploaded is not None:
    data = uploaded.getvalue()

if data is None:
    st.info("Upload a CSV above, or tick **Use demo data** to see what LeakFinder finds.")
    demo_bytes = lf.generate_demo().to_csv(index=False).encode("utf-8")
    st.download_button("Download the demo CSV", demo_bytes, "demo_statement.csv", "text/csv", key="dl_demo")
    st.stop()

try:
    raw = lf.read_table(data)
except ValueError as e:
    st.error(f"Could not read this file: {e}")
    st.stop()

mapping = lf.detect_columns(raw)
with st.expander("Columns look wrong? Adjust how the file is read"):
    cols = ["(none)"] + list(raw.columns)

    def pick(label, key):
        cur_val = mapping.get(key)
        return st.selectbox(label, cols, index=cols.index(cur_val) if cur_val in cols else 0, key=f"map_{key}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        m_date = pick("Date", "date")
    with c2:
        m_desc = pick("Description", "desc")
    with c3:
        m_amt = pick("Amount (single column)", "amount")
    with c4:
        m_deb = pick("Debit / money out", "debit")
    with c5:
        m_cred = pick("Credit / money in", "credit")
    sp_choice = st.radio("Spending appears as", ["Auto-detect", "Negative numbers", "Positive numbers"],
                         horizontal=True, key="sp_choice")
    month_first = st.checkbox("Dates are month/day (US style) when ambiguous", key="month_first")

none = lambda v: None if v == "(none)" else v
mapping.update({"date": none(m_date), "desc": none(m_desc), "amount": none(m_amt), "debit": none(m_deb),
                "credit": none(m_cred), "month_first": month_first,
                "spend_positive": {"Auto-detect": None, "Negative numbers": False, "Positive numbers": True}[sp_choice]})
if mapping["debit"] and mapping["amount"]:
    mapping["debit"] = None  # single amount column wins unless the user clears it

try:
    spend, info = lf.build_transactions(raw, mapping)
except ValueError as e:
    st.error(str(e))
    st.stop()
res = lf.analyze(spend, info)
t = res.totals

if info["date_parse_rate"] < 0.95:
    st.warning(f"Only {info['date_parse_rate']:.0%} of dates could be read - results may be incomplete.")
if info["ambiguous_dates"] and not month_first:
    st.warning("Dates like 03/04/2025 are ambiguous. We assumed day/month. If your bank uses month/day, "
               "tick the box under 'Columns look wrong?'.")

st.header("What we found")
for line in lf.summarize(res, money):
    st.markdown(f"- {line}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Subscriptions / year", money(t["sub_yearly"]), help="Active recurring charges with a steady price")
m2.metric("Subscriptions found", t["n_subs"])
m3.metric("Price hikes", t["n_hikes"], delta=f"+{money(t['hike_extra_year'])}/yr" if t["n_hikes"] else None,
          delta_color="inverse")
m4.metric("Fees + interest", money(t["fee_total"]), help="Total in the period covered by the file")

tabs = st.tabs(["Review these first", "Subscriptions", "Price hikes", "Possible duplicates",
                "Fees & interest", "Other regular bills"])


def show(df, empty_msg, **cfg):
    if df is None or len(df) == 0:
        st.write(empty_msg)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, **cfg)


with tabs[0]:
    st.caption("Ranked by estimated money at stake per year (price rises count extra).")
    show(res.review, "Nothing to review - no recurring charges, duplicates or fees were found.")
with tabs[1]:
    show(res.subscriptions, "No subscriptions found. They need 3+ similar charges at regular intervals (2+ if yearly).")
with tabs[2]:
    show(res.hikes, "No price increases detected on recurring charges.")
with tabs[3]:
    st.caption("Same merchant and amount within 3 days. Often legitimate (two coffees!), so check the receipt.")
    show(res.duplicates, "No possible duplicates found.")
with tabs[4]:
    show(res.fees, "No fees or interest detected.")
with tabs[5]:
    st.caption("Regular but not subscription-like: rent, loans, utilities with changing amounts.")
    show(res.variable, "None found.")

st.divider()
export = res.review.copy()
if len(res.findings):
    export = res.findings
st.download_button("Download findings (CSV)", export.to_csv(index=False).encode("utf-8"),
                   "leakfinder_findings.csv", "text/csv", key="dl_findings")
with st.expander("How this works and its limits"):
    st.markdown(
        "- Merchant names are cleaned (reference numbers, `.COM`, card prefixes removed) and grouped.\n"
        "- A **subscription** is a charge that repeats weekly, every 2 weeks, monthly, quarterly or yearly with a steady amount.\n"
        "- A **price hike** is a recurring charge that stepped up by more than 3%.\n"
        "- **Duplicates** are same merchant + same amount within 3 days.\n"
        "- **Fees** are matched by words like fee, interest, overdraft, ATM, foreign.\n"
        "- Refunds and income are ignored. Merchants that change names, or very short files, can be missed.\n"
        f"- Read as: {info['mode']}; date format {info['date_format']}."
    )
st.caption("LeakFinder is an information tool, not financial, tax or legal advice. "
           "Results are estimates from the file you provide; always verify with your bank.")
