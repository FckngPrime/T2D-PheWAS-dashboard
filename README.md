# T2D PheWAS dashboard

Interactive dashboard for a PheWAS of type 2 diabetes in Danish health
registries. Part of my master's thesis in Data Science.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8050 in a browser.

## Files

- `app.py` - the Dash app
- `requirements.txt` - Python dependencies
- `final_results.parquet` - aggregated Cox results, with privacy
  censoring applied (event counts 1-9 set to 0, 10-19 set to 20)
- `icd102019syst_codes.txt` - ICD-10 names from Sundhedsdatastyrelsen
