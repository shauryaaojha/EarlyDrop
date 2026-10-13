"""The 15 real survey responses: calibration check and scoring with the trained models."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import config as C
from .unit2_models import BayesianLogisticRegression


def run(syn: pd.DataFrame, real: pd.DataFrame | None, real_log: list) -> dict:
    if real is None:
        return {"available": False}
    truth = json.loads(C.GROUND_TRUTH_JSON.read_text())
    targets = truth["calibration_targets"]
    high = syn[syn["current_cgpa"] >= 8.2]
    from .preprocess import encode_frame, read_form
    raw_real = encode_frame(read_form(C.REAL_FORM_CSV))
    calib = [{"item": k, "real_mean": round(v, 3), "synthetic_high_mean": round(float(high[k].mean()), 3),
              "synthetic_all_mean": round(float(syn[k].mean()), 3), "n_real": int(raw_real[k].notna().sum())}
             for k, v in targets.items() if k in syn]

    sc = StandardScaler().fit(syn[C.FEATURES])
    blr = BayesianLogisticRegression().fit(sc.transform(syn[C.FEATURES]), syn[C.TARGET].to_numpy())
    mean, lo, hi = blr.predict_with_uncertainty(sc.transform(real[C.FEATURES]))
    km = KMeans(3, n_init=20, random_state=C.SEED).fit(sc.transform(syn[C.FEATURES]))
    syn_lab = km.labels_
    y = syn[C.TARGET].to_numpy()
    eng = syn["overall_academic_engagement"].to_numpy()
    first = max(range(3), key=lambda k: y[syn_lab == k].mean())
    rest = sorted((k for k in range(3) if k != first), key=lambda k: eng[syn_lab == k].mean())
    name = {first: "High risk", rest[0]: "Coasting", rest[1]: "Engaged achievers"}
    real_lab = km.predict(sc.transform(real[C.FEATURES]))

    return {
        "available": True, "n": len(real),
        "profile": {"cgpa_min": float(real["current_cgpa"].min()), "cgpa_max": float(real["current_cgpa"].max()),
                    "backlogs_total": int(real["active_backlogs"].sum()),
                    "years": real["current_year"].value_counts().sort_index().to_dict(),
                    "departments": real["department"].value_counts().to_dict()},
        "cleaning_log": real_log,
        "calibration": calib,
        "calibration_mean_abs_gap": round(float(np.mean([abs(c["real_mean"] - c["synthetic_high_mean"])
                                                         for c in calib])), 4),
        "scores": [{"respondent": i + 1, "cgpa": float(real["current_cgpa"].iloc[i]),
                    "p_at_risk": round(float(mean[i]), 4), "p90_low": round(float(lo[i]), 4),
                    "p90_high": round(float(hi[i]), 4), "persona": name[int(real_lab[i])]}
                   for i in range(len(real))],
        "persona_counts": pd.Series([name[int(k)] for k in real_lab]).value_counts().to_dict(),
    }
