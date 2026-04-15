# Quantyx — A Python Quantitative Finance Library

A ground-up implementation of quantitative finance models, option pricing engines, and stochastic process simulators. Built from first principles — from the Black-Scholes PDE derivation to exotic option Monte Carlo pricing — this library covers the full stack of modern derivatives pricing.

---

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Installation](#installation)
- [Library Structure](#library-structure)
- [Models](#models)
  - [Abstract Framework](#abstract-framework)
  - [Arithmetic Brownian Motion (Bachelier)](#arithmetic-brownian-motion-bachelier)
  - [Geometric Brownian Motion](#geometric-brownian-motion)
  - [Stochastic Variance Model (Heston)](#stochastic-variance-model-heston)
- [Option Pricing — Analytical](#option-pricing--analytical)
  - [Black-Scholes Call](#black-scholes-call)
  - [Black-Scholes Put](#black-scholes-put)
- [Option Pricing — Monte Carlo](#option-pricing--monte-carlo)
  - [Vanilla Call](#vanilla-call)
  - [Binary Call](#binary-call)
  - [Barrier Call](#barrier-call)
  - [Asian Call & Put](#asian-call--put)
  - [Extendible Call & Put](#extendible-call--put)
- [The Greeks](#the-greeks)
- [Usage Examples](#usage-examples)
- [Mathematical Reference](#mathematical-reference)
- [Model Comparison](#model-comparison)
- [Limitations & Future Work](#limitations--future-work)
- [Dependencies](#dependencies)

---

## Overview

This library implements a complete derivatives pricing framework spanning:

- **Closed-form analytical pricing** via Black-Scholes and Bachelier models
- **Monte Carlo simulation** for vanilla, binary, barrier, Asian, and extendible options
- **Two stochastic process engines** — Geometric Brownian Motion and Heston Stochastic Variance
- **Full Greeks computation** — Delta, Gamma, Vega, Theta for calls and puts
- **Abstract model framework** for extensible model development and calibration

The library is built to be educational and transparent — every formula maps directly to its mathematical derivation. It is also practically useful — models are interchangeable through a common interface, and Monte Carlo pricing works across GBM and Heston backends for every exotic option type.

---

## Mathematical Foundation

The library is grounded in the following theoretical pillars:

### 1. Geometric Brownian Motion

Stock price dynamics under the real-world measure:

$$dS = \mu S \, dt + \sigma S \, dW$$

Where μ is the drift, σ is volatility, and dW is a Wiener process increment.

### 2. Ito's Lemma

For any function C(S, t), the stochastic chain rule gives:

$$dC = \frac{\partial C}{\partial t}dt + \frac{\partial C}{\partial S}dS + \frac{1}{2}\frac{\partial^2 C}{\partial S^2}\sigma^2 S^2 dt$$

The extra term ½σ²S²(∂²C/∂S²) arises because (dW)² = dt — the foundational insight of stochastic calculus.

### 3. Delta Hedging & No Arbitrage

A portfolio Π = C − ΔS with Δ = ∂C/∂S eliminates all randomness (dW cancels). A riskless portfolio must earn the risk-free rate r, leading directly to the Black-Scholes PDE:

$$\frac{\partial C}{\partial t} + rS\frac{\partial C}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} - rC = 0$$

### 4. Black-Scholes Formula

Solving the PDE with boundary condition C(S,T) = max(S−K, 0):

$$C = SN(d_1) - Ke^{-rT}N(d_2)$$

$$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

### 5. Bachelier Formula

Under Arithmetic Brownian Motion dF = σ dW:

$$C = (F_0 - X) \cdot N(d) + \sigma\sqrt{T} \cdot N'(d), \qquad d = \frac{F_0 - X}{\sigma\sqrt{T}}$$

### 6. Heston Stochastic Variance

Two coupled SDEs with correlated Brownian motions:

$$dS = (r - q)S \, dt + S\sqrt{v_t} \, dW_1$$
$$dv_t = \alpha(\beta - v_t)dt + \xi\sqrt{v_t} \, dW_2, \qquad dW_1 \cdot dW_2 = \rho \, dt$$

---

## Installation

```bash
git clone https://github.com/yourusername/quantlib.git
cd quantlib
pip install -r requirements.txt
```

### Requirements

```
numpy
scipy
```

---

## Library Structure

```
quantlib/
│
├── models/
│   ├── base.py                    # Abstract StochasticModel framework
│   ├── arithmetic_brownian.py     # Bachelier / ABM model
│   ├── geometric_brownian.py      # GBM simulator
│   └── stochastic_variance.py     # Heston model
│
├── analytical/
│   ├── black_scholes_call.py      # BS call price + Greeks
│   └── black_scholes_put.py       # BS put price + Greeks
│
├── monte_carlo/
│   ├── vanilla.py                 # MonteCarloCall
│   ├── binary.py                  # MonteCarloBinaryCall
│   ├── barrier.py                 # MonteCarloBarrierCall
│   ├── asian.py                   # MonteCarloAsianCall/Put
│   └── extendible.py              # MonteCarloExtendibleCall/Put
│
└── README.md
```

---

## Models

### Abstract Framework

`StochasticModel` is the abstract base class that all models inherit from. It enforces a consistent interface across every model in the library.

```python
from models.base import StochasticModel

class StochasticModel:
    @abstractmethod
    def vanilla_pricing(self, F0, X, T, op_type="CALL"): pass
    def calibrate(self, impl_vol, T, op_type="CALL"): pass
    @abstractmethod
    def simulate(self): pass
    def __init__(self, params): self.params = params
```

Every model must implement:
- `vanilla_pricing` — closed-form price for European calls/puts
- `simulate` — path generation for Monte Carlo use

The `calibrate` method is optional and model-specific — it fits model parameters to observed implied volatility surfaces.

---

### Arithmetic Brownian Motion (Bachelier)

The oldest option pricing model (Bachelier, 1900). Price moves by fixed dollar amounts, not percentages. Used for interest rates, spreads, and any instrument that can go negative.

```python
from models.arithmetic_brownian import ArithmeticBrownianMotion

# params = [sigma] where sigma is absolute dollar volatility
abm = ArithmeticBrownianMotion(params=[2.0])

# Price a call option
price = abm.vanilla_pricing(F0=100, X=102, T=0.5, op_type="CALL")

# Price a put (via put-call parity automatically)
price = abm.vanilla_pricing(F0=100, X=102, T=0.5, op_type="PUT")

# Simulate 1000 paths
paths, n, dt, T = abm.simulate(F0=100, n=1000, dt=1/252, T=1.0)
```

**Model equation:** dF = σ dW

**Key property:** Constant absolute volatility. Prices can go negative.

---

### Geometric Brownian Motion

The standard Black-Scholes model for equity prices. Volatility scales proportionally with price — a 20% vol stock at $100 moves ~$20/year; at $200 it moves ~$40/year.

```python
from models.geometric_brownian import GeometricBrownianMotion

gbm = GeometricBrownianMotion(S=100, mu=0.1, sigma=0.2, dt=1/252, T=1.0)
gbm.simulated_path   # list of daily prices over 1 year
```

**Model equation:** dS = μS dt + σS dW

**Key property:** Log-normally distributed prices. Cannot go negative.

---

### Stochastic Variance Model (Heston)

The Heston (1993) model — volatility is itself random and mean-reverting. Two correlated random shocks drive the system: one for price, one for variance.

```python
from models.stochastic_variance import StochasticVarianceModel

heston = StochasticVarianceModel(
    S=100,          # Current stock price
    mu=0.1,         # Real-world drift
    r=0.05,         # Risk-free rate
    div=0.02,       # Dividend yield
    alpha=2.0,      # Mean reversion speed
    beta=0.04,      # Long-run variance (= 0.2² → 20% long-run vol)
    rho=-0.7,       # Correlation: negative → crashes cause vol spikes
    vol_var=0.3,    # Vol of vol
    inst_var=0.04,  # Starting variance
    dt=1/252,
    T=1.0
)
heston.simulated_path   # 252 daily prices with stochastic vol
```

**Model equations:**

$$dS = (r-q)S\,dt + S\sqrt{v_t}\,dW_1$$
$$dv_t = \alpha(\beta - v_t)dt + \xi\sqrt{v_t}\,dW_2$$
$$\text{Corr}(dW_1, dW_2) = \rho$$

**Key properties:** Volatility clustering, mean reversion, fat tails, volatility skew. The most realistic equity model in the library.

**Parameters explained:**

| Parameter | Symbol | Typical Range | Effect |
|-----------|--------|---------------|--------|
| alpha | α | 1–5 | Higher = variance reverts faster |
| beta | β | 0.01–0.09 | Long-run variance level |
| rho | ρ | −0.9 to 0 | Negative = vol spikes on crashes |
| vol_var | ξ | 0.1–0.6 | Vol of vol — how wildly σ moves |
| inst_var | v₀ | any positive | Today's starting variance |

---

## Option Pricing — Analytical

### Black-Scholes Call

Exact closed-form pricing and Greeks for European call options.

```python
from analytical.black_scholes_call import BlackScholesCall

call = BlackScholesCall(
    asset_price=100,
    asset_volatility=0.2,
    strike_price=105,
    time_to_expiration=0.5,
    risk_free_rate=0.05
)

call.price   # Option fair value
call.delta   # ∂C/∂S — hedge ratio, always in [0,1]
call.gamma   # ∂²C/∂S² — curvature, rate of delta change
call.vega    # ∂C/∂σ — sensitivity to volatility
call.theta   # ∂C/∂t — time decay, always negative
```

**Formula:**

$$C = SN(d_1) - Ke^{-rT}N(d_2)$$

---

### Black-Scholes Put

Exact closed-form pricing and Greeks for European put options.

```python
from analytical.black_scholes_put import BlackScholesPut

put = BlackScholesPut(
    asset_price=100,
    asset_volatility=0.2,
    strike_price=105,
    time_to_expiration=0.5,
    risk_free_rate=0.05
)

put.price   # Option fair value
put.delta   # Always in [−1, 0] — moves opposite to stock
put.gamma   # Same as equivalent call
put.vega    # Same as equivalent call
put.theta   # Usually negative; can be positive for deep ITM puts
```

**Formula:**

$$P = Ke^{-rT}N(-d_2) - SN(-d_1)$$

---

## Option Pricing — Monte Carlo

All Monte Carlo pricers support two backends: **GBM** (simple, fast) and **Heston** (realistic, stochastic vol). Pass Heston parameters to activate the Heston backend.

### Vanilla Call

Standard European call priced by simulation.

```python
from monte_carlo.vanilla import MonteCarloCall

# GBM backend
call = MonteCarloCall(strike=105, n=10000, r=0.05,
                      S=100, mu=0.1, sigma=0.2, dt=1/252, T=1.0)

# Heston backend (pass alpha, beta, rho, div, vol_var)
call_heston = MonteCarloCall(strike=105, n=10000, r=0.05,
                              S=100, mu=0.1, sigma=0.2, dt=1/252, T=1.0,
                              alpha=2.0, beta=0.04, rho=-0.7,
                              div=0.02, vol_var=0.3)

call.price   # Monte Carlo estimate ≈ Black-Scholes price
```

**Payoff:** max(S_T − K, 0) · e^{−rT}

---

### Binary Call

Fixed payout if stock finishes above strike. Price = probability of finishing ITM × discounted payout.

```python
from monte_carlo.binary import MonteCarloBinaryCall

binary = MonteCarloBinaryCall(
    strike=105, n=10000, payout=1000,
    r=0.05, S=100, mu=0.1, sigma=0.2, dt=1/252, T=1.0
)
binary.price   # ≈ 1000 · e^{-rT} · N(d₂) under GBM
```

**Payoff:** payout · e^{−rT} if S_T ≥ K, else 0

---

### Barrier Call

Path-dependent option that activates (knock-in) or deactivates (knock-out) when the stock touches a barrier at any point during the option's life.

```python
from monte_carlo.barrier import MonteCarloBarrierCall

# Up-and-Out: option dies if stock hits $130 (default)
barrier = MonteCarloBarrierCall(
    strike=105, n=10000, barrier=130,
    r=0.05, S=100, mu=0.1, sigma=0.2, dt=1/252, T=1.0,
    up=True, out=True
)

# Down-and-In: option activates only if stock falls to $80
barrier = MonteCarloBarrierCall(
    strike=105, n=10000, barrier=80,
    r=0.05, S=100, mu=0.1, sigma=0.2, dt=1/252, T=1.0,
    up=False, out=False
)

barrier.price
```

**Parameters:**

| Parameter | Type | Meaning |
|-----------|------|---------|
| barrier | float | The price level that triggers the condition |
| up | bool | True = barrier above current price |
| out | bool | True = touching kills option (knock-out) |

**Four option types:**

| up | out | Name | Behaviour |
|----|-----|------|-----------|
| True | True | Up-and-Out | Dies if stock rises to barrier |
| True | False | Up-and-In | Lives only if stock rises to barrier |
| False | True | Down-and-Out | Dies if stock falls to barrier |
| False | False | Down-and-In | Lives only if stock falls to barrier |

---

### Asian Call & Put

Payoff based on the **average stock price** over the option's life, not the final price. Cheaper than vanilla, manipulation-resistant, widely used in commodities.

```python
from monte_carlo.asian import MonteCarloAsianCall, MonteCarloAsianPut

asian_call = MonteCarloAsianCall(
    strike=105, n=10000, r=0.05,
    S=100, mu=0.1, sigma=0.2, dt=1/252, T=1.0
)

asian_put = MonteCarloAsianPut(
    strike=105, n=10000, r=0.05,
    S=100, mu=0.1, sigma=0.2, dt=1/252, T=1.0
)

asian_call.price   # Always ≤ equivalent vanilla call price
asian_put.price
```

**Call payoff:** max(S̄ − K, 0) · e^{−rT}

**Put payoff:** max(K − S̄, 0) · e^{−rT}

where S̄ = arithmetic mean of all simulated prices.

**Why cheaper than vanilla:** The average of a GBM process has variance σ²T/3 vs σ²T for the terminal price — roughly 42% lower volatility.

---

### Extendible Call & Put

If the option is out of the money at original expiry T, the life is extended by an additional `extension` period. A second chance to recover.

```python
from monte_carlo.extendible import MonteCarloExtendibleCall, MonteCarloExtendiblePut

ext_call = MonteCarloExtendibleCall(
    strike=105, n=10000, r=0.05,
    S=100, mu=0.1, sigma=0.2,
    dt=1/252, T=0.5,
    extension=0.25      # 3-month extension if OTM at 6 months
)

ext_put = MonteCarloExtendiblePut(
    strike=95, n=10000, r=0.05,
    S=100, mu=0.1, sigma=0.2,
    dt=1/252, T=0.5,
    extension=0.25
)

ext_call.price   # Always ≥ equivalent vanilla call price
```

**Logic:**
- Phase 1: Simulate path from S to T
- If ITM at T → collect payoff immediately
- If OTM at T → launch Phase 2 from S_T for `extension` more time
- If ITM at T+extension → collect payoff
- If still OTM → expire worthless

**Pricing identity:**

Extendible Call = Vanilla Call + E[e^{−rT} · Call(S_T, K, extension) · **1**_{S_T < K}]

---

## The Greeks

All analytical pricers compute the full set of Greeks automatically on initialization.

| Greek | Symbol | Formula | Meaning |
|-------|--------|---------|---------|
| Delta | Δ | ∂C/∂S | $ change in option per $1 move in stock |
| Gamma | Γ | ∂²C/∂S² | Rate of change of Delta |
| Vega | ν | ∂C/∂σ | $ change per 1% rise in volatility |
| Theta | Θ | ∂C/∂t | $ change per day passing |

**Greeks relationship (Black-Scholes PDE rewritten):**

$$\Theta + \frac{1}{2}\sigma^2 S^2 \Gamma + rS\Delta - rC = 0$$

The Greeks are not independent — knowing three determines the fourth.

**Call vs Put Greeks:**

| Greek | Call | Put |
|-------|------|-----|
| Delta | N(d₁) ∈ [0,1] | N(d₁)−1 ∈ [−1,0] |
| Gamma | Same for both | Same for both |
| Vega | Same for both | Same for both |
| Theta | Always negative | Usually negative; can be positive |

---

## Usage Examples

### Price a vanilla call analytically and via Monte Carlo

```python
from analytical.black_scholes_call import BlackScholesCall
from monte_carlo.vanilla import MonteCarloCall

S, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.20

# Analytical
bs = BlackScholesCall(S, sigma, K, T, r)
print(f"Black-Scholes price: {bs.price:.4f}")
print(f"Delta: {bs.delta:.4f}, Gamma: {bs.gamma:.4f}")
print(f"Vega: {bs.vega:.4f}, Theta: {bs.theta:.4f}")

# Monte Carlo GBM
mc = MonteCarloCall(K, 50000, r, S, 0.1, sigma, 1/252, T)
print(f"Monte Carlo (GBM) price: {mc.price:.4f}")

# Monte Carlo Heston
mc_h = MonteCarloCall(K, 50000, r, S, 0.1, sigma, 1/252, T,
                       alpha=2.0, beta=0.04, rho=-0.7, div=0.0, vol_var=0.3)
print(f"Monte Carlo (Heston) price: {mc_h.price:.4f}")
```

---

### Compare all exotic option prices

```python
from monte_carlo.vanilla import MonteCarloCall
from monte_carlo.binary import MonteCarloBinaryCall
from monte_carlo.barrier import MonteCarloBarrierCall
from monte_carlo.asian import MonteCarloAsianCall
from monte_carlo.extendible import MonteCarloExtendibleCall

params = dict(strike=105, n=10000, r=0.05, S=100,
              mu=0.1, sigma=0.2, dt=1/252, T=1.0)

vanilla    = MonteCarloCall(**params).price
binary     = MonteCarloBinaryCall(**params, payout=1).price
barrier    = MonteCarloBarrierCall(**params, barrier=130, up=True, out=True).price
asian      = MonteCarloAsianCall(**params).price
extendible = MonteCarloExtendibleCall(**params, extension=0.25).price

print(f"Vanilla:    ${vanilla:.4f}")
print(f"Binary($1): ${binary:.4f}")   # ≈ e^{-rT}·N(d₂)
print(f"Barrier:    ${barrier:.4f}")  # < vanilla (can be knocked out)
print(f"Asian:      ${asian:.4f}")    # < vanilla (avg < final)
print(f"Extendible: ${extendible:.4f}") # > vanilla (second chance)
```

---

### Run Heston simulation and inspect paths

```python
from models.stochastic_variance import StochasticVarianceModel
import numpy as np

heston = StochasticVarianceModel(
    S=100, mu=0.1, r=0.05, div=0.0,
    alpha=2.0, beta=0.04, rho=-0.7,
    vol_var=0.3, inst_var=0.04,
    dt=1/252, T=1.0
)

path = heston.simulated_path
print(f"Start: ${path[0]:.2f}")
print(f"End:   ${path[-1]:.2f}")
print(f"Min:   ${min(path):.2f}")
print(f"Max:   ${max(path):.2f}")
print(f"Steps: {len(path)}")
```

---

## Mathematical Reference

### Black-Scholes Formula Components

| Symbol | Meaning |
|--------|---------|
| S | Current stock price |
| K | Strike price |
| T | Time to expiry (years) |
| r | Continuously compounded risk-free rate |
| σ | Annual volatility |
| N(x) | Standard normal CDF |
| N'(x) | Standard normal PDF |

### d₁ and d₂

$$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}$$

$$d_2 = d_1 - \sigma\sqrt{T} = \frac{\ln(S/K) + (r - \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}$$

**Interpretation:**
- N(d₂) = risk-neutral probability of finishing in the money
- N(d₁) = delta of the call = hedge ratio

### Monte Carlo Convergence

Error scales as 1/√n. To halve the error, quadruple the simulations.

| n | Approximate Error |
|---|------------------|
| 1,000 | ~3.2% |
| 10,000 | ~1.0% |
| 100,000 | ~0.32% |
| 1,000,000 | ~0.10% |

### Heston Feller Condition

For variance to remain strictly positive in continuous time:

$$2\alpha\beta \geq \xi^2$$

The code enforces a variance floor of 10⁻⁷ to handle violations in discrete simulation.

---

## Model Comparison

| Model | Vol Type | Negative Prices | Smile | Closed Form | Best For |
|-------|----------|-----------------|-------|-------------|----------|
| Bachelier (ABM) | Constant absolute | Yes | No | Yes | Rates, spreads |
| Black-Scholes (GBM) | Constant % | No | No | Yes | Equity vanilla |
| Heston | Stochastic | No | Yes | Partial | Equity exotics |

### Option Type Comparison

| Option | Path Dependent | Cheaper than Vanilla | Closed Form |
|--------|---------------|---------------------|-------------|
| Vanilla | No | — | Yes (BS) |
| Binary | No | Often | Yes (e^{-rT}·N(d₂)) |
| Barrier | Yes (trigger) | Yes (knock-out) | Yes (GBM only) |
| Asian | Yes (average) | Yes | No |
| Extendible | Yes (conditional) | No — more expensive | No |

---

## Limitations & Future Work

### Current Limitations

**Models:**
- Constant volatility in GBM (addressed by Heston)
- No jump processes (Merton jump-diffusion not implemented)
- Heston uses Euler-Maruyama discretization — more accurate schemes (Milstein, exact simulation) not yet implemented
- No local volatility model (Dupire)

**Monte Carlo:**
- No variance reduction techniques (antithetic variates, control variates, importance sampling)
- No quasi-Monte Carlo (Sobol sequences)
- Convergence is O(1/√n) — slow for high accuracy requirements

**Scope:**
- European options only in analytical pricing (no American options / early exercise)
- No interest rate models (Vasicek, CIR, Hull-White)
- No credit models

### Planned Extensions

- [ ] Milstein scheme for improved SDE discretization
- [ ] Antithetic variates for Monte Carlo variance reduction
- [ ] American option pricing via Longstaff-Schwartz LSM
- [ ] SABR stochastic volatility model
- [ ] Merton jump-diffusion model
- [ ] Calibration implementation for Heston to implied vol surface
- [ ] Greeks via Monte Carlo (finite difference bumping)
- [ ] Multi-asset correlation models (basket options)

---

## Dependencies

```
numpy >= 1.21.0
scipy >= 1.7.0
```

Install via:

```bash
pip install numpy scipy
```

No other dependencies required. The library is pure Python with standard scientific computing packages.

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Acknowledgements

This library implements classical results from:

- **Fischer Black, Myron Scholes** (1973) — *The Pricing of Options and Corporate Liabilities*
- **Robert Merton** (1973) — *Theory of Rational Option Pricing*
- **Louis Bachelier** (1900) — *Théorie de la spéculation*
- **Steven Heston** (1993) — *A Closed-Form Solution for Options with Stochastic Volatility*

The mathematical derivations follow the delta-hedging / no-arbitrage approach: construct a riskless portfolio, invoke no-arbitrage to set its return equal to the risk-free rate, and derive the Black-Scholes PDE. The solution uses the equivalence to the heat equation from physics.

---

*Built from first principles. Every line of code traces back to a mathematical derivation.*
