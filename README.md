# Worst-of Option Pricer

Monte Carlo pricer for **Worst-of Options** (call & put) on correlated multi-asset baskets, built from scratch in Python.

---

## Table of Contents

1. What is a Worst-of Option?
2. Mathematical Model
3. Variance Reduction
4. Greeks
5. Results
6. Installation & Usage
7. Tests
8. Limitations & Extensions

---

## What is a Worst-of Option?

A **Worst-of Option** is a multi-asset exotic derivative whose payoff at maturity $T$ depends on the worst-performing asset in a basket of $n$ underlyings.

$$\text{Worst-of Call:} \quad Payoff = N \cdot \max\left(\min_{i \in \{1,\dots,n\}} \frac{S_i(T)}{S_i(0)} - K,\ 0\right)$$

$$\text{Worst-of Put:} \quad Payoff = N \cdot \max\left(K - \min_{i \in \{1,\dots,n\}} \frac{S_i(T)}{S_i(0)},\ 0\right)$$

where *K* is a **performance ratio** (1.0 = at-the-money), and *N* is the notional.

**Why is a Worst-of Call cheaper than a vanilla call?**

Since $\min_i(S_i(T)/S_i(0)) \leq S_j(T)/S_j(0)$ for any asset $j$, the worst-of payoff is always dominated by a vanilla call payoff:

$$E\left[\text{payoff}_{\text{WoC}}\right] \leq E\left[\text{payoff}_{\text{vanilla}}\right]$$

Lower correlation makes assets diverge more, pushing the worst performer further down and making the call cheaper. Worst-of Call sellers are therefore short correlation, they profit when correlation drops and the option expires worthless.

---

## Mathematical Model

### Multi-Asset

Each asset $i$ follows a Brownian Motion. Therefore, based on Black-Scholes dynamic we have:

$$dS_i(t) = r \, S_i(t) \, dt + \sigma_i \, S_i(t) \, dW_i(t)$$

By Itô's lemma:

$$S_i(T) = S_i(0) \cdot \exp\left[\left(r - \frac{\sigma_i^2}{2}\right)T + \sigma_i\sqrt{T}\,Z_i\right]$$

### Correlation via Cholesky Decomposition

Asset returns are correlated through a correlation matrix $C$ (symmetric, positive definite, with $C_{ii} = 1$). Cholesky decomposition factorises it as $C = LL^\top$ where $L$ is lower triangular matrix.

Correlated Gaussian vectors are generated as:

$$Z = \varepsilon \cdot L^\top, \quad \varepsilon \sim \mathcal{N}(0, I_n)$$

### Monte Carlo Pricing Pipeline

The price estimator is the mean value of the discounted payoffs :

$$Price = e^{-rT} \cdot \frac{1}{M}\sum_{k=1}^{M} payoff_k$$

Its standard error is:

$$\text{SE} = \frac{{\sigma}_{payoff}}{\sqrt{M}} \cdot e^{-rT}$$

The 95% confidence interval signifies that over 100 independent runs, 95 of the computed intervals contain the true price.

$$CI_{95} = [{Price} - 1.96 \cdot SE, \quad {Price} + 1.96 \cdot SE]$$

---

## Variance Reduction with Antithetic Variates

For each draw $\varepsilon \sim \mathcal{N}(0, I)$, the method also simulates with $-\varepsilon$. Payoffs are averaged in pairs:

$$w_i = \frac{payoff_i + payoff_i'}{2}$$

Since $\text{Cov}(payoff_i, payoff_i') < 0$, therefore the variance of $w_i$ is smaller than the variance of $payoff_i$:

$$\text{Var}\left(w_i\right) = \frac{\text{Var}(payoff) + \text{Var}(payoff') + 2\text{Cov}(payoff, payoff')}{4}$$

The negative covariance term reduces the variance without increasing simulation count. The standard error is computed on the pair means:

$$\text{SE}_{\text{antithetic}} = \frac{{\sigma}_w}{\sqrt{M/2}} \cdot e^{-rT}$$

**Observed result:** ~20% SE reduction, equivalent to 1.57× more simulations at no computational cost.

---

## Greeks with Bump-and-Revalue method

| Greek | Formula |
|---|---|
| Delta $\Delta_i$ | $\frac{V(S_i+h) - V(S_i-h)}{2h}$ |
| Gamma $\Gamma_i$ | $\frac{V(S_i+h) - 2V(S_i) + V(S_i-h)}{h^2}$ |
| Vega $\mathcal{V}_i$ | $\frac{V(\sigma_i+h) - V(\sigma_i-h)}{2h}$ |
| Theta $\Theta$ | $\frac{V(T-h) - V(T)}{h}$ |
| Rho (rate) | $\frac{V(r+h) - V(r-h)}{2h}$ |
| Rho (corr) | $\frac{V(\rho+h) - V(\rho-h)}{2h}$ |

A seed is fixed before each pricer call. This allows to avoid noise created by Monte Carlo method.

---

## Results

### Convergence of the MC Estimator

![Convergence](assets/Monte%20Carlo%20Convergence%20Worst-of%20call%20pricing.png)

### Price Sensitivity to Correlation

Worst-of Call price increases with $\rho$. Higher correlation means assets move together, so the worst performer is less likely to diverge far below the strike. Worst-of Put price shows the strictly inverse behavior.

![Correlation](assets/Monte%20Carlo%20Worst-of%20Price%20as%20a%20Function%20of%20Correlation.png)

---

## Installation & Usage

```bash
git clone https://github.com/Leonard-ROGER/worst-of-option-pricer.git
cd worst-of-option-pricer
pip install -r requirements.txt
```

```python
import numpy as np
from src.pricer import pricer_worst_of_antithetic

S0 = np.array([100.0, 100.0])
r = 0.05
sigma = np.array([0.20,  0.20])
correlation_matrix  = np.array([[1.0, 0.8], [0.8, 1.0]])
T = 1.0
K = 1.0
N = 1000
n_simulations=1000000
option_type="call"

price, se, ci = pricer_worst_of_antithetic(S0, r, sigma, T, K, N, correlation_matrix,n_simulations, option_type)

print(price, se ,ci)

```

---

## Tests

```bash
pytest tests/
```

Three tests validate mathematical properties of the pricer:

- **Price positivity**
- **Worst-of Call Price ≤ Vanilla Call Price**
- **Worst-of Call Price increase with correlation**

---

## Limitations

This pricer is built on the six Black-Scholes assumptions:

1. **Log-normal returns**: Log-returns ln(Sᵢ(T)/Sᵢ(0)) follow a normal distribution
2. **Constant risk-free rate and volatility**: r and σᵢ are fixed
3. **No arbitrage**: No risk-free profit opportunity exists in the market
4. **No transaction costs**: Markets are frictionless
5. **Short selling allowed**: Assets can be sold without being owned beforehand
6. **Divisible assets**: Ability to buy and sell any amount, even fractional, of the stock

In addition, the multi-asset extension introduces one other assumption:

- **Constant correlation**: the correlation matrix ρ is fixed over time
