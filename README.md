# LeakFinder

Upload a bank or card statement CSV and see where your money quietly leaks:

- **Recurring charges / subscriptions** (weekly, every 2 weeks, monthly, quarterly, yearly) with yearly cost
- **Price hikes** on recurring charges
- **Possible duplicate charges**
- **Fees and interest**
- A plain-English summary and a ranked "review these first" list, plus a CSV download of findings

**Privacy:** no bank login, the file is processed in memory for your session only, nothing is stored, and no AI/API calls are made.
Works with CSVs from most banks: auto-detects date, description and amount columns, signed amounts or separate
debit/credit columns, common date formats, and EU (`1.234,56`) or US (`1,234.56`) numbers. If detection is wrong,
the app lets you pick the columns.

> Not financial advice. LeakFinder only highlights patterns in the file you provide.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
pytest
```
Tick **Use demo data** to try it, or use `demo_statement.csv` (regenerate with `python leakfinder.py --demo`).

## Files
`leakfinder.py` engine · `app.py` Streamlit UI · `test_leakfinder.py` tests · `GO_TO_MARKET.md` launch notes

## Limits
Tested on synthetic and hand-made messy files, not on real bank exports from every bank. Merchants whose names change
between charges can be missed; refunds and income are ignored; fee detection is keyword-based.
