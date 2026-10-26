"""Synthetic ILSA-like panel for tests only — never used for reported results."""

from __future__ import annotations

import numpy as np
import pandas as pd

PISA_CYCLES = [2000, 2003, 2006, 2009, 2012, 2015, 2018, 2022]
TIMSS_CYCLES = [1995, 1999, 2003, 2007, 2011, 2015, 2019, 2023]


def make_panel(n_countries: int = 25, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel where next-cycle change depends on ESCS change (+) and anxiety (−)."""
    rng = np.random.default_rng(seed)
    countries = [f"C{i:02d}" for i in range(n_countries)]
    years = sorted(set(PISA_CYCLES + TIMSS_CYCLES))
    cov_rows = []
    for c in countries:
        escs = rng.normal(0, 0.4)
        anx = rng.normal(0, 0.3)
        gdp = rng.normal(10, 0.5)
        for y in range(1993, 2024):
            escs += rng.normal(0.01, 0.03)
            anx += rng.normal(0, 0.02)
            gdp += rng.normal(0.015, 0.02)
            cov_rows.append({"country_code": c, "year": y, "escs_mean": escs, "anxiety_mean": anx,
                             "gdp_per_capita_log": gdp, "noise_cov": rng.normal()})
    cov = pd.DataFrame(cov_rows)
    cov_idx = cov.set_index(["country_code", "year"])
    rows = []
    specs = [("PISA", d, "all", PISA_CYCLES) for d in ("Mathematics", "Reading", "Science")]
    specs += [("TIMSS", d, g, TIMSS_CYCLES) for d in ("Mathematics", "Science") for g in ("4", "8")]
    for c in countries:
        base = rng.normal(480, 40)
        for prog, dom, grade, cycles in specs:
            level = base + rng.normal(0, 10)
            prev_year = None
            for y in cycles:
                if prev_year is not None:
                    d_escs = cov_idx.loc[(c, prev_year), "escs_mean"] - cov_idx.loc[(c, max(prev_year - 3, 1993)), "escs_mean"]
                    level += 25 * d_escs - 4 * cov_idx.loc[(c, prev_year), "anxiety_mean"] + rng.normal(0, 4)
                rows.append({"program": prog, "domain": dom, "grade": grade, "country_code": c,
                             "year": y, "mean_score": round(level, 1), "se": 3.0})
                prev_year = y
    return pd.DataFrame(rows), cov
