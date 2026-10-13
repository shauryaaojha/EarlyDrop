"""Raw Google Form export -> clean modelling table. Same code path for real and synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def read_form(path) -> pd.DataFrame:
    """Read a form export and rename the 50 question headers to snake_case (by position,
    because the live form repeats one question header)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding_errors="replace")
    extra = list(df.columns[len(C.SNAKE):])
    df.columns = C.SNAKE + extra
    return df.apply(lambda s: s.str.strip())


def _to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace("%", "", regex=False).replace("", np.nan), errors="coerce")


def encode_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Text answers -> numbers. No cleaning or imputation; blanks stay NaN."""
    out = df.copy().replace("", np.nan)
    for col in ["previous_sem_sgpa", "current_cgpa", "internal_assessment_average",
                "attendance_percentage", "courses_completed", "current_semester"]:
        out[col] = _to_number(df[col])
    for col, mapping in C.ORDINAL_MAPS.items():
        out[col] = df[col].map(mapping)
    for col in C.LIKERT:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Inspection -> cleaning -> transformation, with a log of every change."""
    log = []
    answers = [c for c in C.SNAKE if c != "timestamp"]
    n0 = len(raw)
    dedup = raw.drop_duplicates(subset=answers, keep="first").reset_index(drop=True)
    log.append({"step": "Duplicate submissions removed", "count": n0 - len(dedup)})

    pct = sum(dedup[c].str.contains("%", regex=False).sum()
              for c in ["attendance_percentage", "internal_assessment_average"])
    log.append({"step": "Percent-sign answers parsed (e.g. '80.50%')", "count": int(pct)})

    df = encode_frame(dedup)
    blank_before = int(df[C.LATE_ADDED].isna().any(axis=1).sum())
    log.append({"step": "Rows missing questions added mid-survey", "count": blank_before})

    for col, (lo, hi) in C.VALID_RANGES.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        if bad.any():
            log.append({"step": f"Out-of-range {col} set to missing",
                        "count": int(bad.sum()), "values": dedup.loc[bad, col].tolist()})
            df.loc[bad, col] = np.nan

    numeric = list(C.VALID_RANGES)
    ordinal = list(C.ORDINAL_MAPS) + C.LIKERT
    imputed = {}
    for col in numeric + ordinal:
        miss = int(df[col].isna().sum())
        if miss:
            fill = df[col].median()
            df[col] = df[col].fillna(round(fill) if col in ordinal and col not in
                                     ("daily_study_hours", "weekly_online_learning_hours") else fill)
            imputed[col] = miss
    log.append({"step": "Median imputation (numeric) / median category (ordinal)",
                "count": int(sum(imputed.values())), "columns": imputed})

    for col in C.NOMINAL:
        df[col] = df[col].fillna("Unknown")
    for col in C.TEXT_FIELDS:
        df[col] = df[col].fillna("")
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    return df, log


def build_synthetic() -> tuple[pd.DataFrame, list[dict]]:
    raw = read_form(C.SYN_FORM_CSV)
    df, log = clean(raw)
    outcomes = pd.read_csv(C.SYN_OUTCOMES_CSV)
    df = df.merge(outcomes[["student_id", "end_sem_sgpa", "at_risk"]], on="student_id", how="left")
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN_CSV, index=False)
    return df, log


def build_real() -> tuple[pd.DataFrame, list[dict]] | tuple[None, list]:
    if not C.REAL_FORM_CSV.exists():
        return None, []
    df, log = clean(read_form(C.REAL_FORM_CSV))
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.REAL_CLEAN_CSV, index=False)
    return df, log


def load_clean() -> pd.DataFrame:
    return pd.read_csv(C.CLEAN_CSV, parse_dates=["timestamp"])


def load_history() -> pd.DataFrame:
    return pd.read_csv(C.SYN_HISTORY_CSV)
