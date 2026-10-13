# EarlyDrop: build phases

| Phase | Scope | Exit criteria | Status |
|---|---|---|---|
| P1 | Synthetic generator in the live form's exact schema, calibrated on the 15 real responses; linked semester history; ground truth | `generate.py` writes the export, outcomes, history and truth; at-risk rate ≈ 28 % | done |
| P1 | Cleaning pipeline shared by real and synthetic data | duplicates, `%` strings, mid-survey blanks and impossible values handled and logged | done |
| P2 | Units 1–2: EDA, Bayes, curve fitting, 4 regressors, 7 classifiers, Bayesian LR (scratch) | CV metrics in `metrics.json`; OLS and Bayesian LR match the library | done |
| P3 | Unit 3: K-Means and EM (scratch), hierarchical, PCA (scratch), FA, personas | scratch EM ARI = 1.00 against sklearn; PCA eigenvalues match | done |
| P4 | Unit 4: HMM (scratch, scaled), Baum–Welch, Viterbi, BIC over K, Kalman filter, baselines | log-likelihood equals hmmlearn's; A recovered within 0.07 | done |
| P5 | Streamlit dashboard: 7 pages, "Blackboard" theme, motion engine from the Animations library | all views pass `AppTest`; works with `?static` | done |
| P6 | Docker: one image that runs the pipeline at build time; `START.bat` for non-technical users | `docker compose up` serves :8501 (dashboard) and :8080 (deck) | done |
| P7 | Review deck: 25 slides, data-bound, animated, PDF backup | every number from `deck-data.js`; PDF has one page per slide | done |
| P8 | README, viva cheat sheet | — | done |
| Next | PBL-III (Unit 5): CART, bagging, random forest, boosting, stacking on the same cohort | — | todo |
| Next | Add "SGPA of each completed semester" to the Google Form so real trajectories replace the synthetic history | — | todo |

Re-run everything with `python run_all.py`. The dashboard and deck read only `results/metrics.json` and `deck/deck-data.js`, so they always show the latest run.
