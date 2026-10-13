# EarlyDrop

**Student success and dropout-risk analysis using machine learning.**
21CSC305P Machine Learning · Section E2 · Faculty: Dr. Santhiya Parivallal

| S.No | Name | Register Number |
|---|---|---|
| 1 | Yash Arun Mulajkar | RA2411003011562 |
| 2 | Bignesh Sahu | RA2411003011563 |
| 3 | Shaurya Ojha | RA2411003011565 |
| 4 | Aditya Bhutada | RA2411003011607 |

EarlyDrop asks whether mid-semester survey signals can flag students whose **end-semester SGPA will fall below 6.5**, early enough for help to matter. This repo covers **PBL-I (Units 1–2)** and **PBL-II (Units 3–4)** end to end:
- a data pipeline
- from-scratch implementations, each checked against a library
- a Streamlit dashboard
- an animated review deck with a PDF backup

---

## Quick start

**With Docker (recommended, and what to send a friend).** Install Docker Desktop and start it, then:

- **Windows:** double-click `START.bat`.
- **macOS / Linux:** run `./start.sh`.

Then open:

| What | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| Review deck | http://localhost:8080 (press `F` for fullscreen) |

`STOP.bat` (or `docker compose down`) stops everything. The first build downloads about 1 GB and runs the whole pipeline once.

**Without Docker** (Python 3.12+):

```bash
pip install -r requirements.txt
python run_all.py                       # ~1–2 min on a laptop CPU: data → Units 1–4 → results/ figs/ deck data
streamlit run app/streamlit_app.py      # dashboard
start deck/index.html                   # deck (or open it in any browser)
powershell -File deck/export_pdf.ps1    # deck → deck/EarlyDrop_Deck.pdf
```

**Hardware.** Everything is classical ML on about 400 students, so no GPU is needed. The full pipeline runs on an Intel Core Ultra 5 laptop CPU in under two minutes. A GPU (e.g. an RTX 4080) would not speed anything up.

---

## Data: real form, synthetic cohort (read this first)

- **The live Google Form** has 15 real responses so far (`data/real/`). All 15 are high performers: CGPA 8.28–10, zero backlogs, and no outcome label. They cannot train a risk model.
- **So the modelling cohort is synthetic.** `earlydrop/generate.py` writes 400 responses in the **exact 50-column format of the live form**. It also reproduces the form's real quirks:
  - `80.50%` strings
  - a `7%` typo
  - 9 questions added mid-survey, so early rows are blank
  - duplicate submissions
- **Calibration.** Likert answers are calibrated so that synthetic high performers (CGPA ≥ 8.2) match the real respondents' means. The mean gap is 0.01 Likert points over 28 items.
- **A hidden process with known truth.** Behind the data sits a real hidden process:
  - Each student has latent traits (ability, engagement, motivation, support).
  - An academic state (Struggling / Stable / Strong) evolves as a Markov chain.
  - Each semester's SGPA is emitted from the state.
  - The survey answers come from the traits and the current state.
  - The label `at_risk` is the outcome of the semester in progress (end-sem SGPA < 6.5), which the survey cannot see.
- **Why this helps:** because we know the truth, every Unit 3–4 method can be checked against it.
- **One cleaning pipeline.** The same `earlydrop/preprocess.py` cleans the real export and the synthetic one.

Numbers below describe this synthetic cohort, not SRM students. The pipeline runs unchanged on real responses once outcome labels exist.

---

## What is implemented

| Unit | Topic (course plan) | Where | From scratch? |
|---|---|---|---|
| 1 | EDA, discrete and continuous random variables, quantiles, expectation, covariance | `unit1_stats.py` | yes |
| 1 | Bayes' rule, independence and conditional independence | `unit1_stats.py` | yes |
| 1 | Polynomial curve fitting (under-fit, over-fit, regularised) | `unit1_stats.py` | yes |
| 2 | Least squares / MLE (normal equations), ridge, robust (Huber) | `unit2_models.py` | OLS yes |
| 2 | Bayesian linear regression with evidence maximisation | `unit2_models.py` | yes |
| 2 | Discriminant / generative models (LDA, Gaussian NB), logistic regression | `unit2_models.py` | – |
| 2 | Laplace approximation + Bayesian logistic regression (IRLS, probit predictive) | `unit2_models.py` | yes |
| 2 | Kernels and SVMs (linear, RBF, polynomial, grid search) | `unit2_models.py` | – |
| 3 | K-Means (k-means++, Lloyd), choosing k (elbow, silhouette, BIC) | `unit3_clustering.py` | yes |
| 3 | Mixtures of Gaussians via EM | `unit3_clustering.py` | yes |
| 3 | Hierarchical clustering (4 linkages), dissimilarity measures, cluster evaluation | `unit3_clustering.py` | – |
| 3 | PCA (eigen-decomposition), choosing latent dimensions, factor analysis | `unit3_clustering.py` | PCA yes |
| 4 | Markov chain, HMM: scaled forward–backward, Baum–Welch, Viterbi, BIC over K | `unit4_hmm.py` | yes |
| 4 | Linear dynamical system (Kalman filter, MLE of noise variances) | `unit4_hmm.py` | yes |

`run_all.py` runs 8 self-checks and fails loudly if any break:
- the from-scratch OLS, GMM, PCA and HMM match sklearn and hmmlearn
- the Bayesian LR matches logistic regression
- the schema is intact and nothing is missing after cleaning

## Results (synthetic cohort, seed 42)

| Unit | Question | Method | Result |
|---|---|---|---|
| 1 | How much do two signals change the odds? | Bayes' rule | P(risk) 0.28 → 0.71 (attendance < 75 %) → 0.96 (+ internals < 60 %) |
| 1 | Curve fitting | Polynomial, 12 points | best M = 1; M = 9 test RMSE 8.56 → 0.21 with ridge |
| 2 | Predict end-semester SGPA | Ridge | R² 0.798 · RMSE 0.654; Bayesian LR 95 % band covers 93 % |
| 2 | Robustness | Huber vs OLS | 8 % corrupted targets: OLS RMSE 0.668 → 0.839, Huber 0.678 → 0.675 |
| 2 | Classify at-risk | Logistic / Bayesian LR | 5-fold CV ROC-AUC 0.953; threshold 0.4 → recall 0.86 |
| 3 | How many groups? | GMM BIC | minimum at k = 3 |
| 3 | Groups without labels | K-Means | High risk (74 % at risk), Coasting (9 %), Engaged achievers (11 %) |
| 3 | Scratch vs library | EM / PCA | EM = sklearn (ARI 1.00, same log-likelihood); eigenvalues match |
| 3 | Latent dimensions | PCA / FA | 10 PCs for 90 %; 5 PCs already give 0.905 CV accuracy; FA recovers the 4 planted traits |
| 4 | Number of hidden states | HMM BIC | K = 3 |
| 4 | Parameter recovery | Baum–Welch | max abs(Â − A_true) = 0.063; P(stay Struggling) = 0.84 |
| 4 | Decoding | Viterbi | true hidden state recovered 92.5 % of the time; identical to hmmlearn |
| 4 | Next-semester SGPA | Kalman filter | RMSE 0.897 (persistence 0.924) |
| 4 | Early warning | grades only vs survey | ROC-AUC 0.862 (HMM on grade history) vs 0.953 (survey) |

Read honestly:
- The HMM's next-band accuracy ties a plain Markov chain (76.1 %), because the emission matrix is nearly diagonal. It wins only on log-loss.
- Adding the HMM forecast to the survey does not lift AUC, since the survey already contains previous SGPA and CGPA.

## Repository layout

```
earlydrop/          config · generate · preprocess · unit1_stats · unit2_models · unit3_clustering · unit4_hmm · realcheck · plots
run_all.py          the whole pipeline + self-checks → results/metrics.json, figs/*.png, deck/deck-data.js
app/                Streamlit dashboard (streamlit_app.py, views/, fx/ motion engine + "Blackboard" theme)
deck/               animated review deck (index.html, deck.js, deck-data.js) + export_pdf.ps1 → EarlyDrop_Deck.pdf
data/real/          the 15 real form responses        data/raw/ synthetic export, outcomes, semester history, ground truth
docs/               build-phases.md · viva-cheatsheet.md
Dockerfile · docker-compose.yml · START.bat · STOP.bat · start.sh · requirements.txt
```

## Presenting

**The deck:** open `deck/index.html` (or http://localhost:8080 under Docker).

| Keys | Action |
|---|---|
| `→` / `Space` / clicker | next slide |
| `←` | back |
| `N` | speaker notes, with a suggested 4-way speaker split |
| `O` | slide overview |
| `F` | fullscreen |

`index.html?print#s13` shows one slide with no motion. `deck/EarlyDrop_Deck.pdf` is the backup.

**The dashboard:** add `?static` to the URL to turn motion off on slow projector PCs. The **Predict a student** page is the live demo. Move attendance and watch the risk and its credible band change.

## Limitations and ethics

- The cohort is synthetic. The real responses so far have no outcome label.
- Survey answers are self-reported. Correlation is not causation.
- Semester history comes from the generator. The form does not yet ask for per-semester SGPA, which is the obvious next question to add.
- A risk score should only trigger an **offer of support**, never a label, a stream or a penalty.
