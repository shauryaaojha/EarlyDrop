# EarlyDrop: viva cheat sheet

Short answers to the questions a panel is likely to ask, using this project's own numbers.
All results come from the synthetic cohort (n = 400, seed 42). Say "synthetic" out loud if asked about real-world accuracy.

## Data

**Why synthetic data?** The live form has 15 real responses, all high performers with CGPA 8.28–10, zero backlogs and no outcome label. A risk model cannot be trained on that. So:
- We generate 400 responses in the form's exact 50-column format.
- Their Likert answers are calibrated to match the real respondents (mean gap 0.01 points).
- The real 15 are scored by the trained model as a sanity check. All come out low risk, as they should.

**What does synthetic buy us?** Ground truth. We know the true transition matrix, the true hidden states and the 4 latent traits, so we can check whether HMM, PCA/FA and clustering *recover* them. With real data you can only check fit.

**What is the label?** `at_risk` = the end-of-semester SGPA of the semester in progress is below 6.5. The survey is taken mid-semester, so the label is genuinely in the future.

**What did cleaning do?**
- 6 duplicate submissions removed.
- 22 `80.50%`-style strings parsed.
- 32 rows predate 9 mid-survey questions.
- 4 impossible values set to missing: CGPA 78.6, internal "6%" twice, attendance 108.
- 292 cells median-imputed.

## Unit 1

**Expectation of a discrete RV?** E[X] = Σ x·P(x). For active backlogs, E[X] = 0.52.

**Bayes' rule here?** H = at risk, E1 = attendance below the 75 % rule, E2 = internals below 60 %.
- The prior is 0.28.
- After E1 it rises to 0.71.
- After E1 and E2 it is 0.96.

The sequential update assumes E1 ⟂ E2 | H. The exact posterior from the data is 0.963, so the assumption costs almost nothing here.

**Independence vs conditional independence?** Low attendance and low study hours are dependent overall: P(A∩B) = 0.175 vs P(A)P(B) = 0.117. Given risk status they are much closer to independent. Risk is the common cause.

**Why does M = 9 overfit?** With 12 training points a degree-9 polynomial has 10 parameters. It threads the training points (train RMSE 0.015) but test RMSE explodes to 8.56. A ridge penalty (ln λ = −8) brings it back to 0.21.

## Unit 2

**OLS as maximum likelihood?** Under Gaussian noise, maximising the likelihood is the same as minimising squared error. We solve the normal equations directly, and the coefficients match sklearn exactly.

**Ridge vs Huber?**
- **Ridge** shrinks the weights (L2 penalty, α chosen by CV). It gave the best R², 0.798.
- **Huber** uses a loss that is linear for large residuals. With 8 % of training targets corrupted, OLS RMSE goes 0.668 → 0.839, while Huber goes 0.678 → 0.675.

**Bayesian linear regression?**
- Prior w ~ N(0, α⁻¹I), noise precision β.
- α and β are set by evidence maximisation, which gives γ ≈ 16 effective parameters.
- The predictive variance is 1/β + φᵀS_Nφ.
- 93 % of test students fall inside the 95 % band.

**Laplace approximation?**
1. Find the MAP weights by Newton–Raphson (IRLS).
2. Approximate the posterior as N(w_MAP, H⁻¹), where H is the Hessian of the negative log posterior.
3. For predictions, use the probit approximation σ(κ μₐ) with κ = (1 + πσₐ²/8)^(−½).

Bayesian LR's CV AUC is 0.953, the same as plain logistic regression, plus every weight has a posterior std.

**Generative vs discriminative?** LDA and Naive Bayes model p(x | class), which makes them generative. Logistic regression models p(class | x) directly, which makes it discriminative. Naive Bayes is weakest (CV AUC 0.925) because its independence assumption is violated: the features are correlated.

**Kernel trick?** An SVM only needs inner products, so a kernel k(x, x′) replaces φ(x)·φ(x′) without ever building φ. The RBF grid search picked C = 10, γ = 0.003. The small γ means a nearly linear boundary, consistent with linear models doing just as well.

**Why tune the threshold?** Early warning cares about missing at-risk students. Lowering the threshold to 0.4 lifts recall from 0.75 to 0.86 for a small loss in precision.

## Unit 3

**How did you choose k?** We used three criteria:
- The elbow is soft.
- Silhouette prefers 2.
- GMM BIC has a clear minimum at 3.

We chose k = 3 because BIC is likelihood-based and penalises complexity, and because the three groups are interpretable.

**What is EM doing?** It alternates two steps until the log-likelihood stops rising:
- **E-step:** compute the responsibilities γ(z_nk).
- **M-step:** re-estimate π, μ and Σ from those responsibilities.

EM never decreases the likelihood, and our curve shows it. It converged in 32 iterations and matches sklearn exactly (ARI 1.00, log-lik −4796.34).

**K-Means vs GMM?** K-Means is the hard-assignment, equal-spherical-covariance limit of EM. GMM allows soft assignments and full covariances.

**Why are silhouettes low (~0.16)?** Risk is a continuum, not separate blobs. The clusters are still meaningful:
- **High risk:** 74 % at risk.
- **Coasting:** decent grades but low effort, 9 % at risk.
- **Engaged achievers:** 11 % at risk.

**Average linkage has the best ARI. Why not use it?** Its sizes are 89 / 310 / 1. It isolates one outlier and splits the rest in two. Single linkage chains everything into one cluster (398 / 1 / 1).

**How many PCA components?**
- 10 components keep 90 % of the variance.
- PC1 alone carries 42 %; it is the academic-performance axis.
- 5 components already give the same CV accuracy as all 15.

**Factor analysis vs PCA?** FA models each item as loadings × latent factors + item-specific noise. PCA just rotates to directions of maximum variance. On the 28 Likert items, FA recovers the 4 traits the generator used: engagement, support, performance outlook and motivation.

## Unit 4

**What is the HMM here?**
- Hidden state: Struggling, Stable or Strong.
- Observation: that semester's SGPA band (Poor < 6.5, Average < 8, Good).
- Parameters: π, A (3×3 transitions) and B (3×3 emissions).

**Forward–backward and scaling?** The forward pass computes α̂ₜ = p(zₜ | o₁..ₜ), normalised by the scaling factors cₜ, so log p(O) = Σ log cₜ. Without scaling the raw probabilities underflow. The backward pass uses the same cₜ. Then γ = α̂β̂ and the ξ terms drive Baum–Welch.

**Viterbi vs forward filtering?**
- **Viterbi** finds the single most likely whole path, using max-product in log space.
- **Filtering** gives the marginal of each state at each time.

On the declining student, P(Struggling) jumps from about 0 to 0.68 in the semester the grades drop.

**How do you know it's right?**
- Our log-likelihood equals hmmlearn's with the same parameters.
- The learned A is within 0.063 of the true A.
- Viterbi recovers the true state 92.5 % of the time, with paths identical to hmmlearn's.
- BIC picks K = 3 over 2, 4 and 5.

**Why does the HMM not beat a Markov chain on next-band accuracy?** Emissions are nearly diagonal (0.97 / 0.91 / 0.98), so the band almost *is* the state. The HMM then behaves like a Markov chain on bands. It still has a better log-loss (0.667 vs 0.671).

**Kalman filter?** It is a linear dynamical system with a local-level model, xₜ = xₜ₋₁ + w and yₜ = xₜ + v. The noise variances q = 0.21 and r = 0.43 are fitted by maximum likelihood. It gives the best next-SGPA RMSE, 0.897 against 0.924 for persistence.

**The key takeaway?** Struggling is sticky: P(stay Struggling) = 0.84. A student who starts struggling usually keeps struggling unless something changes, and that is the case for intervening early.
