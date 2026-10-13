"""EarlyDrop: run the whole pipeline end to end.

    python run_all.py              generate data, run Units 1-4, write results/ figs/ deck data
    python run_all.py --no-generate   reuse the existing data/raw files
"""
import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from earlydrop import config as C
from earlydrop import generate, preprocess, realcheck, unit1_stats, unit2_models, unit3_clustering, unit4_hmm

warnings.filterwarnings("ignore")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-generate", action="store_true")
    args = ap.parse_args()
    timings, t0 = {}, time.perf_counter()

    def step(name, fn):
        t = time.perf_counter()
        print(f"> {name} ...", flush=True)
        out = fn()
        timings[name] = round(time.perf_counter() - t, 2)
        return out

    if not args.no_generate:
        step("generate synthetic cohort", lambda: generate.write(generate.generate()))
    syn, syn_log = step("clean synthetic responses", preprocess.build_synthetic)
    real, real_log = step("clean real responses", preprocess.build_real)
    history = preprocess.load_history()
    outcomes = pd.read_csv(C.SYN_OUTCOMES_CSV)

    u1 = step("Unit 1: statistics, probability, Bayes", lambda: unit1_stats.run(syn, history))
    u2 = step("Unit 2: regression + classification", lambda: unit2_models.run(syn))
    u3 = step("Unit 3: clustering, PCA, factor analysis", lambda: unit3_clustering.run(syn, outcomes))
    u4 = step("Unit 4: HMM + Kalman filter", lambda: unit4_hmm.run(syn, history))
    rc = step("real responses: calibration + scoring", lambda: realcheck.run(syn, real, real_log))

    checks = {
        "schema_49_modelling_columns": len([c for c in C.SNAKE if c not in C.TEXT_FIELDS]) + 1 == 49,
        "no_missing_after_cleaning": int(syn[C.FEATURES + [C.TARGET]].isna().sum().sum()) == 0,
        "ols_normal_equations_match_sklearn": u2["regression"]["ols_matches_sklearn"],
        "bayesian_lr_auc_close_to_sklearn": abs(
            next(r["roc_auc"] for r in u2["classification"]["test"] if r["model"].startswith("Bayesian"))
            - next(r["roc_auc"] for r in u2["classification"]["test"] if r["model"] == "Logistic Regression")) < 0.02,
        "scratch_gmm_matches_sklearn_ari>=0.95": u3["scratch_vs_sklearn"]["gmm_ari"] >= 0.95,
        "scratch_pca_eigenvalues_match_sklearn": u3["scratch_vs_sklearn"]["pca_eigvals_match"],
        "scratch_hmm_loglik_matches_hmmlearn": abs(u4["loglik"] - u4["hmmlearn"]["loglik_same_params"]) < 1e-6,
        "scratch_hmm_as_good_as_hmmlearn_fit": u4["loglik"] >= u4["hmmlearn"]["loglik_own_fit"] - 0.5,
    }
    metrics = {
        "meta": {"project": "EarlyDrop", "course": "21CSC305P Machine Learning", "section": "E2",
                 "data_is_synthetic": C.DATA_IS_SYNTHETIC, "seed": C.SEED, "n_synthetic": len(syn),
                 "n_real": 0 if real is None else len(real), "generated_at": time.strftime("%Y-%m-%d %H:%M"),
                 "timings_sec": timings, "checks": checks},
        "cleaning": syn_log,
        "unit1": u1, "unit2": u2, "unit3": u3, "unit4": u4, "real": rc,
    }
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metrics, default=_json_default, indent=1)
    C.METRICS_JSON.write_text(payload)
    deck = C.ROOT / "deck"
    deck.mkdir(exist_ok=True)
    (deck / "deck-data.js").write_text("window.EARLYDROP = " + json.dumps(metrics, default=_json_default) + ";\n")

    print(f"\nDone in {time.perf_counter() - t0:.1f}s -> {C.METRICS_JSON.relative_to(C.ROOT)}, figs/, deck/deck-data.js")
    for k, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
