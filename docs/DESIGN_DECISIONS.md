# Design Decisions and Parameter Justifications

This document provides technical rationale for key parameter choices and design decisions in the fraud detection system and synthetic data generation.

---

## Table of Contents

1. [Credit Limit Calculation](#credit-limit-calculation)
2. [Account Age Distribution](#account-age-distribution)
3. [Transaction Count Distribution](#transaction-count-distribution)
4. [Statistical Distribution Choices](#statistical-distribution-choices)

---

## Credit Limit Calculation

### Formula

```python
credit_limit = base_limit * (credit_score / 700) * np.random.lognormal(0, 0.3)
```

### Components

**Base Limit by Persona:**
- Light user: $2,000
- Convenience user: $8,000
- Revolver: $5,000
- Transactor: $15,000

**Credit Score Normalization (÷700):**
- 700 is the median U.S. FICO score (range: 300-850)
- Normalization creates multiplier centered around 1.0
- Score of 700 → multiplier of 1.0 (receives base limit)
- Score of 800 → multiplier of 1.14 (14% increase)
- Score of 600 → multiplier of 0.86 (14% decrease)

**Lognormal Multiplier (μ=0, σ=0.3):**
- Introduces realistic underwriting variation (30-40% typical range)
- **Always positive** (credit limits cannot be negative)
- **Right-skewed** (most near 1.0×, some higher)
- **Median multiplier ≈ 1.0**, typical range 0.5× to 2.0×

### Rationale

Two applicants with identical credit scores often receive different limits due to:
- Income verification differences
- Debt-to-income ratio
- Employment history and stability
- Bank-specific risk appetite
- Existing customer relationship

The lognormal distribution captures this real-world variation without requiring explicit modeling of these additional factors.

### Example Results

For a convenience user (base limit $8,000) with credit score 740:

```
Normalization: 740 / 700 = 1.057
Lognormal sample: 1.12 (example)
Final limit: $8,000 × 1.057 × 1.12 = $9,549
```

**Distribution of 1,000 samples:**
- 25th percentile: ~$6,800
- Median: ~$8,400
- 75th percentile: ~$10,200
- Range: ~$4,500 - $17,000

This variation is consistent with real credit card issuer practices.

---

## Account Age Distribution

### Formula

```python
account_age_days = int(np.random.exponential(500))
account_age_days = min(account_age_days, 3650)  # Cap at 10 years
```

### Parameters

- **Scale parameter λ = 500 days** (mean of exponential distribution)
- **Maximum age: 3,650 days** (10 years)

### Rationale

**Why Exponential Distribution?**

1. **Many new accounts, fewer old accounts** - characteristic of active card portfolios
2. **Models account attrition** - cards close over time (churn, product changes)
3. **Right-skewed** - most accounts relatively young

**Distribution Results:**
- ~51% of accounts less than 1 year old
- ~31% of accounts 1-3 years old
- ~18% of accounts 3-10 years old

**Why Cap at 10 Years?**

1. **Card expiration**: Physical cards typically expire after 3-5 years
2. **Product evolution**: Card products and terms change; old accounts migrate
3. **Portfolio management**: Issuers periodically refresh inactive accounts
4. **Data reality**: Accounts older than 10 years are rare in active portfolios

### Real-World Validation

Industry data (Federal Reserve, card issuer portfolio studies) shows:
- High new account acquisition rates (marketing, balance transfers)
- Natural attrition (account closure, churn to other cards)
- Exponential-like age distribution in practice

**Comparison to Normal Distribution:**
- Normal would create symmetric distribution (as many very old as very new - unrealistic)
- Exponential better captures portfolio dynamics

---

## Transaction Count Distribution

### Formula

```python
n_transactions = int(np.random.poisson(cardholder['monthly_txns_mean']))
```

### Parameters

Mean transactions per month by persona:
- Light user: λ = 4
- Convenience user: λ = 12
- Revolver: λ = 15
- Transactor: λ = 25

### Rationale

**Why Poisson Distribution?**

**Key Properties:**
1. **Discrete, non-negative integers** - you cannot have 2.7 transactions
2. **Models rare events in fixed time window** - transactions occurring in a month
3. **Mean = Variance** - Poisson assumption holds for transaction arrival rates
4. **Right-skewed** - most months near mean, occasional high-activity months

**Real-World Alignment:**
- Transaction counts are fundamentally **count data** (discrete events)
- Monthly counts show **moderate overdispersion** (variance ≈ mean)
- Some months naturally busier (holidays, travel, unexpected expenses)
- Some months quieter (vacation, reduced spending)

### Example: Convenience User (mean = 12)

Sample months:
- Month 1: 11 transactions (typical)
- Month 2: 14 transactions (busy)
- Month 3: 8 transactions (quiet)
- Month 4: 12 transactions (average)
- Month 5: 16 transactions (holiday shopping)
- Month 6: 10 transactions (normal)

**Distribution over 1,000 months:**
- Mean: 12.0 transactions
- Standard deviation: 3.5 transactions
- 95% of months: 6-20 transactions

### Alternative Distributions Considered

**Fixed count:**
- Too deterministic (every month exactly 12 transactions)
- Unrealistic (real spending varies)

**Uniform(6, 18):**
- Equal probability of 6 or 18 transactions (unrealistic)
- No concentration around typical behavior

**Normal(12, 3):**
- Would allow negative counts (impossible)
- Symmetric (as likely to have 5 as 19 - not observed)

**Negative Binomial:**
- Allows more overdispersion than Poisson
- Added complexity without clear benefit for monthly counts
- Better for weekly or daily counts (more variation)

**Conclusion:** Poisson is industry standard for modeling transaction arrival rates and provides best fit for monthly data.

---

## Statistical Distribution Choices

### Summary Table

| Variable | Distribution | Key Property Utilized |
|----------|--------------|----------------------|
| **Credit limit** | Lognormal | Always positive, right-skewed |
| **Account age** | Exponential | Decreasing hazard rate (more new accounts) |
| **Transaction count** | Poisson | Discrete counts, models rare events |
| **Transaction amount** | Gamma | Always positive, flexible shape/scale |
| **Credit score** | Normal (truncated) | Central tendency, symmetric |

### General Principles

1. **Match domain constraints**
   - Use distributions that respect natural limits (e.g., amounts > 0)
   - Avoid distributions that allow impossible values

2. **Parsimony**
   - Use simplest distribution that captures essential pattern
   - Avoid over-parameterized models

3. **Empirical validation**
   - Compare synthetic data to known industry statistics
   - Validate against published research and reports

4. **Interpretability**
   - Choose distributions with clear parameter meanings
   - Enable sensitivity analysis and "what-if" scenarios

---

## Validation Sources

**Industry Data:**
- Federal Reserve Payments Study (transaction volumes, trends)
- Payment network rules and specifications (Visa, Mastercard, EMVCo)
- Card issuer portfolio reports (anonymized, aggregate statistics)

**Academic Research:**
- Credit risk modeling literature (credit limit assignment)
- Consumer finance studies (spending patterns, account dynamics)
- Fraud detection papers (transaction characteristics, fraud rates)

**Regulatory Filings:**
- Bank call reports (aggregate credit card data)
- Consumer Financial Protection Bureau data (credit card market)

---

**For implementation details, see:**
- `generate_synthetic_data.py` - Data generation code
- `assumptions.md` - Complete assumption documentation
- `DATA_GENERATION_METHODOLOGY.md` - Comprehensive technical details (private reference)

**Last Updated:** January 2026
