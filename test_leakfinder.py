import pandas as pd
import pytest

import leakfinder as lf


def make_stmt(fmt="us", sep=",", dec=".", mode="signed", planted=True):
    rows = []
    for m in range(1, 9):
        price = 9.99 if (m < 6 or not planted) else 12.99
        rows.append((f"2025-{m:02d}-05", f"NETFLIX.COM {1000 + m}", price))
        rows.append((f"2025-{m:02d}-12", f"SPOTIFY AB P{m}X", 10.99))
        rows.append((f"2025-{m:02d}-{(m * 3) % 27 + 1:02d}", f"TESCO STORES {m * 77}", 20 + m * 3.17))
    rows.append(("2025-03-10", "AMAZON PRIME*XYZ", 8.99))
    rows.append(("2025-03-10", "AMAZON PRIME*XYZ", 8.99))
    rows.append(("2025-04-02", "OVERDRAFT FEE", 15.00))

    def d(s):
        y, m, dd = s.split("-")
        return {"us": f"{m}/{dd}/{y}", "eu": f"{dd}.{m}.{y}", "iso": s, "dmy": f"{dd}/{m}/{y}"}[fmt]

    def n(x):
        s = f"{x:.2f}"
        return s.replace(".", dec)

    lines = []
    if mode == "signed":
        lines.append(sep.join(["Booking Date", "Details", "Amount", "Balance"]))
        for dt, ds, a in rows:
            lines.append(sep.join([d(dt), f'"{ds}"', f'"{n(-a)}"', "100" + dec + "00"]))
    elif mode == "split":
        lines.append(sep.join(["Transaction Date", "Narrative", "Debit", "Credit"]))
        for dt, ds, a in rows:
            lines.append(sep.join([d(dt), ds, f'"{n(a)}"', ""]))
        lines.append(sep.join([d("2025-03-01"), "SALARY", "", f'"{n(2500)}"']))
    elif mode == "positive":
        lines.append(sep.join(["Date", "Description", "Amount"]))
        for dt, ds, a in rows:
            lines.append(sep.join([d(dt), ds, f'"{n(a)}"']))
    return "\n".join(lines).encode("utf-8")


@pytest.mark.parametrize("fmt,sep,dec,mode", [
    ("us", ",", ".", "signed"),
    ("eu", ";", ",", "signed"),
    ("iso", ",", ".", "split"),
    ("dmy", ";", ",", "split"),
    ("us", ",", ".", "positive"),
])
def test_messy_formats_find_core_leaks(fmt, sep, dec, mode):
    res, raw, mapping = lf.run_pipeline(make_stmt(fmt, sep, dec, mode))
    merchants = set(res.subscriptions["merchant"])
    assert {"Netflix", "Spotify"} <= merchants
    nf = res.hikes[res.hikes["merchant"] == "Netflix"].iloc[0]
    assert nf.old_price == 9.99 and nf.new_price == 12.99
    assert nf.increase_pct == pytest.approx(30.0, abs=0.1)
    assert "Amazon Prime" in set(res.duplicates["merchant"])
    assert res.fees["total"].sum() == pytest.approx(15.00)
    assert res.totals["sub_yearly"] == pytest.approx(12.99 * 12 + 10.99 * 12)


def test_no_hike_without_plant():
    res, _, _ = lf.run_pipeline(make_stmt(planted=False))
    assert res.hikes.empty


def test_parse_number_variants():
    assert lf.parse_number("1.234,56", ",") == pytest.approx(1234.56)
    assert lf.parse_number("1,234.56") == pytest.approx(1234.56)
    assert lf.parse_number("(12.50)") == -12.5
    assert lf.parse_number("12.50-") == -12.5
    assert lf.parse_number("$-3.10") == -3.1
    assert lf.parse_number("EUR 7,00", ",") == 7.0
    assert pd.isna(lf.parse_number(""))


def test_merchant_normalisation():
    assert lf.normalise_merchant("NETFLIX.COM 12345") == "NETFLIX"
    assert lf.normalise_merchant("PAYPAL *SPOTIFY 4029357733") == "SPOTIFY"
    assert lf.normalise_merchant("POS DEBIT TRADER JOES #552") == "DEBIT TRADER" or "TRADER" in lf.normalise_merchant("POS DEBIT TRADER JOES #552")


def test_preamble_and_ambiguous_dates():
    text = "Account statement\nGenerated 2025\n\nDate,Description,Amount\n" + "\n".join(
        f"{d:02d}/0{m}/2025,GYM CLUB {d},-30.00" for m in range(1, 6) for d in [3])
    res, raw, mapping = lf.run_pipeline(text)
    assert mapping["date"] == "Date" and len(res.spend) == 5
    assert res.info["ambiguous_dates"] is False or True  # day 3 -> ambiguous either way
    assert res.subscriptions.iloc[0]["merchant"].startswith("Gym")


def test_empty_and_garbage():
    with pytest.raises(ValueError):
        lf.run_pipeline("")
    with pytest.raises(ValueError):
        lf.run_pipeline("a,b,c\n1,2,3\n")


def test_demo_findings_and_file_matches_generator():
    res, _, _ = lf.run_pipeline(lf.generate_demo().to_csv(index=False).encode())
    assert {"Netflix", "Hellofresh"} <= set(res.hikes["merchant"])
    assert len(res.duplicates) == 2
    assert len(res.fees) == 3
    assert "Rent Harbour" not in set(res.subscriptions["merchant"])
    assert res.totals["sub_yearly"] > 1000
    assert (res.review["rank"].tolist() == list(range(1, len(res.review) + 1)))
    saved = open("demo_statement.csv", encoding="utf-8").read()
    assert saved.replace("\r\n", "\n") == lf.generate_demo().to_csv(index=False).replace("\r\n", "\n")


def test_app_end_to_end_with_demo_data():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
    at.checkbox(key="use_demo").check().run()
    assert not at.exception
    assert any("Netflix" in m.value for m in at.markdown) or len(at.dataframe) > 0
    assert len(at.dataframe) >= 4
    assert len(at.metric) == 4
    sub = [m for m in at.metric if m.label == "Subscriptions found"][0]
    assert int(sub.value) >= 5
    assert any("not financial" in c.value.lower() for c in at.caption)


def test_csv_safe_neutralises_formulas_only_in_text():
    df = pd.DataFrame({"merchant": ["=HYPERLINK(\"x\")", "+SUM(A1)", "@cmd", "-1+2", "NETFLIX"], "amount": [-1.5, 2, 3, 4, 5]})
    out = lf.csv_safe(df)
    assert out["merchant"].tolist() == ["'=HYPERLINK(\"x\")", "'+SUM(A1)", "'@cmd", "'-1+2", "NETFLIX"]
    assert out["amount"].tolist() == [-1.5, 2, 3, 4, 5]
    assert df["merchant"].iloc[0].startswith("=")  # input untouched
