#!/usr/bin/env python3
"""
Download country-level covariates from the World Bank WDI API and merge them
into data/country_panel/ilsa_country_covariates.csv (wide: country_code, year, …).

  gdp_per_capita_log             log(NY.GDP.PCAP.PP.KD)  GDP per capita, PPP, constant international $
  education_expenditure_pct_gdp  SE.XPD.TOTL.GD.ZS       Government expenditure on education, % of GDP

Columns already in the file that this script does not fetch (e.g. escs_mean,
ict_home_access, anxiety_mean taken from OECD/IEA country tables) are kept.
Countries default to those in data/country_panel/ilsa_country_means.csv.

Requires network access to api.worldbank.org.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "country_panel"
API = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}?format=json&per_page=20000&date={start}:{end}"

INDICATORS = {
    "gdp_per_capita_log": ("NY.GDP.PCAP.PP.KD", math.log),
    "education_expenditure_pct_gdp": ("SE.XPD.TOTL.GD.ZS", None),
}


def fetch_indicator(indicator: str, countries: list[str] | None, start: int, end: int) -> pd.DataFrame:
    codes = ";".join(countries) if countries else "all"
    url = API.format(countries=codes, indicator=indicator, start=start, end=end)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except OSError as exc:
            if attempt == 3:
                raise SystemExit(f"World Bank API request failed: {exc}")
            time.sleep(2 ** (attempt + 1))
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["country_code", "year", "value"])
    rows = [
        {"country_code": r.get("countryiso3code"), "year": int(r["date"]), "value": r.get("value")}
        for r in payload[1]
        if r.get("countryiso3code") and r.get("value") is not None
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=DATA / "ilsa_country_means.csv")
    parser.add_argument("--out", type=Path, default=DATA / "ilsa_country_covariates.csv")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2024)
    args = parser.parse_args()

    countries = None
    if args.panel.exists():
        countries = sorted(pd.read_csv(args.panel)["country_code"].astype(str).str.upper().unique())
    frames = []
    for column, (indicator, transform) in INDICATORS.items():
        df = fetch_indicator(indicator, countries, args.start, args.end)
        if transform is not None:
            df["value"] = df["value"].map(transform)
        frames.append(df.rename(columns={"value": column}).set_index(["country_code", "year"]))
        print(f"{indicator}: {len(df)} country-years", flush=True)
    wb = pd.concat(frames, axis=1).reset_index()

    if args.out.exists():
        existing = pd.read_csv(args.out)
        keep = [c for c in existing.columns if c not in INDICATORS]
        wb = existing[keep].merge(wb, on=["country_code", "year"], how="outer")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wb.sort_values(["country_code", "year"]).to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(wb)} rows)")


if __name__ == "__main__":
    sys.exit(main())
