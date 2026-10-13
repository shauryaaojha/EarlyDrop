"""Synthetic EarlyDrop cohort, written in the exact raw format of the live Google Form.

How it works
------------
1. Every student gets latent traits: ability, engagement, motivation and support.
2. Their academic state (Struggling / Stable / Strong) evolves semester by semester
   through a Markov chain (A_TRUE). Each semester's SGPA is emitted from that state.
   This is a genuine hidden Markov process, so Unit 4 has a ground truth to recover.
3. Survey answers are generated from the latent traits and the current-semester state.
   Likert answers are calibrated so that synthetic high performers (CGPA >= 8.2) match
   the means of the 15 real responses, which are all high performers.
4. The label `at_risk` is the outcome of the in-progress semester (end-semester SGPA < 6.5).
   It is not visible on the survey, exactly like a real early-warning setting.
5. The quirks seen in the real export are reproduced: "80.50%" strings, a "7%" typo,
   questions added mid-survey (blank for early rows), and duplicate submissions.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config as C

# ---- latent academic-state process (ground truth for Unit 4) -------------------------
A_TRUE = np.array([[0.78, 0.19, 0.03],
                   [0.10, 0.78, 0.12],
                   [0.03, 0.15, 0.82]])
STATE_MU = np.array([5.6, 7.3, 8.9])  # mean SGPA per state
ABILITY_W = 0.35                      # stable student-level offset on SGPA
SGPA_NOISE = 0.45

YEARS = {"2nd": (3, 2), "3rd": (5, 4), "4th": (7, 6)}  # year -> (current sem, completed sems)
YEAR_P = [0.32, 0.43, 0.25]
DEPARTMENTS = (["Computer Science & Engineering", "Information Technology",
                "Electronics & Communication Engineering", "Electrical Engineering",
                "Mechanical Engineering", "Other"], [0.55, 0.10, 0.12, 0.08, 0.08, 0.07])
PROGRAMMES = (["B.Tech", "M.Tech", "Other"], [0.90, 0.04, 0.06])

# Ordinal / Likert answer generators: categories + loadings on latent factors.
L5 = ["1", "2", "3", "4", "5"]
ITEMS = {
    "assignment_completion_consistency": (L5, {"e": .7, "pz": .4}),
    "class_attendance_frequency": (L5, {"att": .9, "e": .2}),
    "class_participation": (L5, {"e": .8}),
    "doubt_resolution_frequency": (L5, {"e": .6, "sup": .3}),
    "faculty_academic_interaction": (L5, {"e": .5, "sup": .4}),
    "study_consistency": (L5, {"e": .8, "pz": .3}),
    "exam_cramming_frequency": (L5, {"e": -.6, "pz": -.3}),
    "self_learning_frequency": (L5, {"m": .6, "e": .3}),
    "revision_frequency": (L5, {"e": .7, "pz": .3}),
    "additional_resource_usage": (L5, {"m": .6, "e": .2}),
    "lms_usage_frequency": (L5, {"e": .5}),
    "online_learning_material_usage": (L5, {"e": .4, "m": .4}),
    "online_activity_completion": (L5, {"e": .8}),
    "late_assignment_frequency": (L5, {"e": -.8, "pz": -.3}),
    "test_preparation_consistency": (L5, {"e": .7, "pz": .3}),
    "course_interest": (L5, {"m": .8}),
    "academic_motivation": (L5, {"m": 1.0}),
    "career_goal_clarity": (L5, {"m": .6, "a": .3}),
    "academic_progress_satisfaction": (L5, {"pz": .9, "tr": .4}),
    "overall_academic_engagement": (L5, {"e": .8, "m": .3}),
    "programme_completion_confidence": (L5, {"pz": .6, "m": .5}),
    "peer_support": (L5, {"sup": .8}),
    "faculty_support": (L5, {"sup": .8}),
    "academic_help_accessibility": (L5, {"sup": .7}),
    "academic_guidance_frequency": (L5, {"sup": .7}),
    "academic_workload_difficulty": (L5, {"st": .8}),
    "academic_performance_concern": (L5, {"st": .8, "tr": -.4}),
    "programme_completion_certainty": (L5, {"pz": .6, "m": .5}),
    "daily_study_hours": (list(C.ORDINAL_MAPS["daily_study_hours"]), {"e": .7, "m": .3}),
    "weekly_online_learning_hours": (list(C.ORDINAL_MAPS["weekly_online_learning_hours"]), {"e": .5}),
    "missed_assessment_frequency": (list(C.ORDINAL_MAPS["missed_assessment_frequency"]), {"e": -.7, "pz": -.4}),
    "attendance_change": (list(C.ORDINAL_MAPS["attendance_change"]), {"tr": .6}),
    "digital_engagement_change": (list(C.ORDINAL_MAPS["digital_engagement_change"]), {"tr": .3, "e": .3}),
    "assessment_performance_trend": (list(C.ORDINAL_MAPS["assessment_performance_trend"]), {"tr": .9}),
    "overall_academic_situation_change": (list(C.ORDINAL_MAPS["overall_academic_situation_change"]), {"tr": .9}),
    "academic_performance_change": (list(C.ORDINAL_MAPS["academic_performance_change"]), {"dif": .9}),
    "additional_academic_support_needed": (["No", "Not sure", "Yes"], {"st": .9}),
}
ITEM_NOISE = 0.7
DEFAULT_BASE = {"attendance_change": 1.0, "additional_academic_support_needed": 0.6}

CHALLENGES = ["Time management", "Attendance", "Too many assignments", "Understanding core subjects",
              "Coding practice", "Placement preparation alongside academics", "Maths heavy courses",
              "Lack of practical exposure", "Backlog clearance", "Staying consistent", "Nil", "na"]
SUPPORT = ["More practical classes", "Online video support", "Mentoring from seniors",
           "Doubt clearing sessions", "Learning industry based skills", "Previous year question practice",
           "Extra lab hours", "Peer study groups", "nil", "na"]


def band_of(sgpa):
    return np.digitize(sgpa, C.BAND_EDGES)  # 0 Poor, 1 Average, 2 Good


def _z(x):
    return (x - x.mean()) / x.std()


def _latent(rng: np.random.Generator, n: int) -> dict:
    """Latent traits + semester trajectories for n students."""
    a = rng.normal(size=n)
    e = 0.5 * a + np.sqrt(0.75) * rng.normal(size=n)
    m = _z(0.4 * e + 0.3 * a + 0.8 * rng.normal(size=n))
    sup = _z(rng.normal(size=n) + 0.2 * e)

    years = rng.choice(list(YEARS), size=n, p=YEAR_P)
    hist_states, hist_sgpa, s_cur = [], [], np.empty(n, int)
    for i in range(n):
        k = YEARS[years[i]][1]
        logits = np.array([-1.1 * a[i] - 0.2, 0.3, 1.1 * a[i] - 0.1])
        p0 = np.exp(logits) / np.exp(logits).sum()
        s = rng.choice(3, p=p0)
        st = []
        for _ in range(k):
            st.append(s)
            s = rng.choice(3, p=A_TRUE[s])
        s_cur[i] = s  # state during the semester in progress
        hist_states.append(st)
        sg = STATE_MU[st] + ABILITY_W * a[i] + rng.normal(0, SGPA_NOISE, k)
        hist_sgpa.append(np.clip(sg, 3.0, 10.0).round(2))

    perf_cur = STATE_MU[s_cur] + ABILITY_W * a
    end_sgpa = np.clip(perf_cur + rng.normal(0, SGPA_NOISE, n), 3.0, 10.0).round(2)
    s_prev = np.array([h[-1] for h in hist_states])
    tr = _z((STATE_MU[s_cur] - STATE_MU[s_prev]) / 1.6 + 0.4 * rng.normal(size=n))
    dif = _z(np.array([h[-1] - h[-2] for h in hist_sgpa]) / 0.6)
    pz = _z(perf_cur)
    st = _z(-0.6 * pz + 0.8 * rng.normal(size=n))
    att = np.clip(48 + 4.6 * perf_cur + 4 * e + rng.normal(0, 4.5, n), 35, 100).round(1)
    return dict(a=a, e=e, m=m, sup=sup, pz=pz, st=st, tr=tr, dif=dif, att_val=att, att=_z(att),
                years=years, hist_states=hist_states, hist_sgpa=hist_sgpa, s_cur=s_cur,
                perf_cur=perf_cur, end_sgpa=end_sgpa)


def _item_index(L: dict, loadings: dict, noise: np.ndarray, base: float, k: int):
    z = sum(w * L[f] for f, w in loadings.items()) + noise
    return np.clip(np.round(base + z), 0, k - 1).astype(int)


def _real_targets() -> dict:
    """Mean encoded answer per item in the real responses (None if the file is absent)."""
    if not C.REAL_FORM_CSV.exists():
        return {}
    from .preprocess import encode_frame, read_form
    real = encode_frame(read_form(C.REAL_FORM_CSV))
    return {c: float(real[c].mean()) for c in ITEMS if c in real and real[c].notna().sum() >= 3}


def _calibrate(L, noises, high_mask, targets) -> dict:
    """Bisection on each item's base so synthetic high performers match the real mean."""
    bases = {}
    for name, (cats, load) in ITEMS.items():
        k = len(cats)
        enc = C.ORDINAL_MAPS.get(name, {c: i + 1 for i, c in enumerate(cats)})
        values = np.array([enc[c] for c in cats], float)
        if name not in targets:
            bases[name] = DEFAULT_BASE.get(name, (k - 1) / 2)
            continue
        lo, hi = -3.0, k + 2.0
        for _ in range(40):
            mid = (lo + hi) / 2
            idx = _item_index(L, load, noises[name], mid, k)
            if values[idx[high_mask]].mean() < targets[name]:
                lo = mid
            else:
                hi = mid
        bases[name] = (lo + hi) / 2
    return bases


def generate(n: int = C.N_STUDENTS, seed: int = C.SEED) -> dict:
    rng = np.random.default_rng(seed)
    L = _latent(rng, n)
    cgpa = np.array([h.mean() for h in L["hist_sgpa"]]).round(2)
    prev = np.array([h[-1] for h in L["hist_sgpa"]])
    high = cgpa >= 8.2

    noises = {name: rng.normal(0, ITEM_NOISE, n) for name in ITEMS}
    targets = _real_targets()
    bases = _calibrate(L, noises, high, targets)

    rows = {s: [None] * n for s in C.SNAKE}
    t0 = datetime(2026, 8, 20, 9, 0, 0)
    gaps = rng.exponential(scale=3.2 * 3600, size=n)
    stamps = [t0 + timedelta(seconds=float(s)) for s in np.cumsum(gaps)]
    rows["timestamp"] = [t.strftime("%d/%m/%Y %H:%M:%S") for t in stamps]
    rows["programme"] = list(rng.choice(PROGRAMMES[0], size=n, p=PROGRAMMES[1]))
    rows["department"] = list(rng.choice(DEPARTMENTS[0], size=n, p=DEPARTMENTS[1]))
    rows["current_year"] = list(L["years"])
    rows["current_semester"] = [str(YEARS[y][0]) for y in L["years"]]
    rows["previous_sem_sgpa"] = [f"{v:g}" for v in prev]
    rows["current_cgpa"] = [f"{v:g}" for v in cgpa]

    internal = np.clip(9.2 * L["perf_cur"] + rng.normal(0, 6, n), 15, 100).round(0)
    rows["internal_assessment_average"] = [f"{v:g}" for v in internal]
    rows["attendance_percentage"] = [f"{v:g}" for v in L["att_val"]]
    k_done = np.array([YEARS[y][1] for y in L["years"]])
    rows["courses_completed"] = [str(int(v)) for v in np.maximum(1, 5 * k_done + rng.integers(-2, 3, n))]
    lam = np.array([1.4 * np.maximum(0, 6.8 - h[-2:]).mean() for h in L["hist_sgpa"]])
    backlogs = np.minimum(rng.poisson(lam), 4)
    rows["active_backlogs"] = ["4 or more" if b == 4 else str(b) for b in backlogs]

    for name, (cats, load) in ITEMS.items():
        idx = _item_index(L, load, noises[name], bases[name], len(cats))
        rows[name] = [cats[i] for i in idx]

    for col, pool in (("biggest_academic_challenge", CHALLENGES), ("required_academic_support", SUPPORT)):
        rows[col] = [pool[rng.integers(len(pool))] if rng.random() < 0.6 else "" for _ in range(n)]

    form = pd.DataFrame({s: rows[s] for s in C.SNAKE})
    form.insert(0, "student_id", [f"S{i + 1:04d}" for i in range(n)])

    # ---- reproduce the real export's quirks ------------------------------------------
    early = int(round(0.08 * n))
    form.loc[: early - 1, C.LATE_ADDED] = ""                       # questions added mid-survey
    pct = rng.choice(n, size=int(0.05 * n), replace=False)
    form.loc[pct, "attendance_percentage"] = [f"{float(v):.2f}%" for v in form.loc[pct, "attendance_percentage"]]
    typo = rng.choice(np.arange(early, n), size=2, replace=False)
    form.loc[typo, "internal_assessment_average"] = [f"{int(float(v) // 10)}%" for v in form.loc[typo, "internal_assessment_average"]]
    bad = rng.choice(n, size=2, replace=False)
    form.loc[bad[0], "attendance_percentage"] = "108"
    form.loc[bad[1], "current_cgpa"] = str(round(float(form.loc[bad[1], "current_cgpa"]) * 10, 1))
    dup_src = rng.choice(n, size=6, replace=False)
    dups = form.loc[dup_src].copy()
    dups["timestamp"] = [(datetime.strptime(t, "%d/%m/%Y %H:%M:%S") + timedelta(seconds=int(rng.integers(40, 200))))
                         .strftime("%d/%m/%Y %H:%M:%S") for t in dups["timestamp"]]
    form = pd.concat([form, dups]).sort_values("timestamp", key=lambda s: pd.to_datetime(s, dayfirst=True))
    form = form.reset_index(drop=True)

    outcomes = pd.DataFrame({
        "student_id": [f"S{i + 1:04d}" for i in range(n)],
        "end_sem_sgpa": L["end_sgpa"],
        "at_risk": (L["end_sgpa"] < C.BAND_EDGES[0]).astype(int),
        "true_state_current": L["s_cur"],
    })
    hist = pd.DataFrame([
        {"student_id": f"S{i + 1:04d}", "semester": t + 1, "sgpa": float(L["hist_sgpa"][i][t]),
         "band": int(band_of(L["hist_sgpa"][i][t])), "true_state": int(L["hist_states"][i][t])}
        for i in range(n) for t in range(len(L["hist_sgpa"][i]))
    ])
    truth = {
        "A_true": A_TRUE.tolist(), "state_mu": STATE_MU.tolist(), "sgpa_noise": SGPA_NOISE,
        "ability_weight": ABILITY_W, "band_edges": list(C.BAND_EDGES),
        "pi_empirical": np.bincount(hist.query("semester == 1").true_state, minlength=3).tolist(),
        "calibration_targets": targets, "calibrated_bases": bases, "seed": seed, "n": n,
    }
    return {"form": form, "outcomes": outcomes, "history": hist, "truth": truth}


def write(out: dict) -> None:
    C.RAW.mkdir(parents=True, exist_ok=True)
    f = out["form"].copy()
    # export with the real Google Form headers; student_id stays as a trailing join key
    sid = f.pop("student_id")
    f.columns = C.HEADERS
    f["student_id"] = sid
    f.to_csv(C.SYN_FORM_CSV, index=False)
    out["outcomes"].to_csv(C.SYN_OUTCOMES_CSV, index=False)
    out["history"].to_csv(C.SYN_HISTORY_CSV, index=False)
    C.GROUND_TRUTH_JSON.write_text(json.dumps(out["truth"], indent=2))


if __name__ == "__main__":
    o = generate()
    write(o)
    print(o["form"].shape, "at_risk rate", o["outcomes"].at_risk.mean().round(3))
