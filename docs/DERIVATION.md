# Deriving the model and the estimator

## 1. Why the geometric distribution

Each play of the claw machine is a Bernoulli trial: it either grabs a tub
(success) or it does not (failure).  Denote the per-play success probability
by $p$ — the machine's *effective* grab probability, which bundles claw
strength, machine settings, and how well the tubs are packed.

Let $T$ be the number of plays needed until (and including) the first
successful grab.  If the trials are independent and $p$ is constant, then

$$
P(T = t) = (1-p)^{t-1} p, \qquad t = 1, 2, 3, \dots
$$

which is the **geometric distribution**, $T \sim \mathrm{Geo}(p)$.  Its key
quantities are

$$
\mathbb{E}[T] = \frac{1}{p}, \qquad
\mathrm{Var}(T) = \frac{1-p}{p^2}, \qquad
P(T \le k) = 1 - (1-p)^k .
$$

The expected cost of one successful grab is therefore $\$1 \cdot 1/p$ (at
\$1 per play).

**Why not the other usual suspects?**

| Distribution | Would model | Why it is not the right default here |
|---|---|---|
| Binomial | number of wins in a *fixed* number of plays | the number of plays is random — we stop at the first win |
| Negative binomial | plays until the $r$-th win | the special case $r=1$ *is* the geometric; use it if you play until you collect several tubs |
| Poisson | number of wins in a fixed time window | there is no fixed window; we count trials until a success, not events per window |
| Exponential (continuous) | waiting time | our waiting variable is discrete (plays), so the geometric — its discrete analogue — is the right one |

A distinctive property of the geometric is the **memoryless property**:

$$
P(T > s + t \mid T > s) = P(T > t),
$$

i.e. after $s$ failed plays the remaining expected effort is still $1/p$.
This holds exactly when the machine's per-play behaviour is constant.  If
the claw strength ramps up with every failed attempt (common on claw
machines), the success probability is no longer constant and the geometric
is only a baseline — see the model checks below.

## 2. Maximum-likelihood estimation of $p$

### 2.1 Complete data (every session ends with a grab)

Collect $n$ sessions.  Session $i$ needed $t_i$ plays (the last one a
success).  The likelihood is

$$
L(p) = \prod_{i=1}^{n} p(1-p)^{t_i - 1}
     = p^{n} (1-p)^{\sum_i t_i - n},
$$

and the log-likelihood is

$$
\ell(p) = n \log p + \Big(\textstyle\sum_i t_i - n\Big) \log(1-p).
$$

Setting the score to zero:

$$
\frac{d\ell}{dp} = \frac{n}{p} - \frac{\sum_i t_i - n}{1-p} = 0
\;\;\Longrightarrow\;\;
 n(1-p) = p\Big(\textstyle\sum_i t_i - n\Big)
\;\;\Longrightarrow\;\;
\boxed{\;\hat p_{\mathrm{MLE}} = \frac{n}{\sum_i t_i} = \frac{1}{\bar t}\;}
$$

The MLE is simply the reciprocal of the sample mean number of tries — the
same estimator that the method of moments gives.

### 2.2 Right-censored sessions (player gives up)

Realistically, a player sometimes quits after $u_j$ failed plays without a
grab.  Such a session contributes $P(T > u_j) = (1-p)^{u_j}$ to the
likelihood.  With $m$ successes and $c$ censored sessions:

$$
L(p) = p^{m} (1-p)^{\sum_i(t_i - 1) + \sum_j u_j}.
$$

Solving $d\ell/dp = 0$ gives

$$
\boxed{\;\hat p_{\mathrm{MLE}} = \frac{m}{\sum_i t_i + \sum_j u_j}
       = \frac{\text{number of successful grabs}}{\text{total number of plays}}\;}
$$

The complete-data result is the special case with no censoring.  In words:
**the MLE of the per-play win probability is just successes divided by
total plays.**

## 3. Variance and confidence intervals

### 3.1 Fisher information (complete data)

The second derivative of the log-likelihood is

$$
\ell''(p) = -\frac{n}{p^2} - \frac{\sum_i t_i - n}{(1-p)^2},
$$

and since $\mathbb{E}[\sum_i t_i] = n/p$,

$$
I(p) = -\mathbb{E}[\ell''(p)] = \frac{n}{p^2(1-p)} .
$$

By the asymptotic normality of the MLE, $\hat p \approx N\big(p,\,
I(p)^{-1}\big)$, so

$$
\mathrm{SE}(\hat p) \approx \sqrt{\frac{\hat p^{\,2}(1-\hat p)}{n}} .
$$

### 3.2 Binomial view (any data)

Pool all plays: $m$ successes out of $N$ total plays is a binomial count, so
$\hat p = m/N$ and

$$
\mathrm{SE}(\hat p) = \sqrt{\frac{\hat p(1-\hat p)}{N}} .
$$

For complete data $N = n/p$, and this equals the Fisher-information
standard error — the two views agree.  Because $\hat p$ is a binomial
proportion, a **Wilson score interval** is used rather than the Wald
interval (which behaves poorly near $p=0$ or $1$ and for small $N$):

$$
\frac{\hat p + \dfrac{z^2}{2N} \pm z\sqrt{\dfrac{\hat p(1-\hat p)}{N}
+ \dfrac{z^2}{4N^2}}}{1 + \dfrac{z^2}{N}},
\qquad z = 1.96 \text{ for } 95\% .
$$

## 4. Decision analysis

At \$1 per play with a tub worth about \$6:

- **Expected spend per successful grab:** $\mathbb{E}[\text{cost}] = 1/p$.
- **Break-even win probability:** $p^{*} = 1/6 \approx 0.1667$, where the
  expected spend equals the tub's retail value.
- **Probability of grabbing within $k$ plays:** $1 - (1-p)^k$.  At
  $p = 1/6$: within 3 plays ≈ 42%, within 6 ≈ 67%, within 10 ≈ 84%.

So the practical question "is this machine worth it?" becomes a hypothesis
test of $H_0: p = 1/6$ against the observed $\hat p$.

## 5. Model checks

1. **Memoryless / constant-hazard check.**  Under the geometric model the
   survival curve $S(t) = P(T > t) = (1-p)^t$ is exactly log-linear.
   Plotting the empirical survival curve on a log scale should follow a
   straight line.  Systematic curvature means $p$ changes with $t$
   (e.g. claw strength increasing after failures).
2. **Pearson chi-square goodness-of-fit.**  Compare observed try-count
   frequencies with the fitted geometric PMF, merging tail cells so every
   expected count is ≥ 5, and losing one degree of freedom for the
   estimated parameter.
3. **KS caveat.**  A one-sample Kolmogorov–Smirnov test with an
   *estimated* parameter is invalid (its null distribution changes); the
   repository uses the chi-square test plus the survival diagnostic
   instead.  A Monte-Carlo null distribution would be the rigorous
   alternative.
