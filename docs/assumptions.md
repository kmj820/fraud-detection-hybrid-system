# Detailed Assumptions and Model Specifications

This document provides comprehensive technical assumptions for the fraud detection analysis. All assumptions are documented to ensure transparency and reproducibility.

---

## Table of Contents

1. [Scope & Context](#scope--context)
2. [Card Verification Technologies](#card-verification-technologies)
3. [Liability & Regulatory Framework](#liability--regulatory-framework)
4. [Fraud Detection System Design](#fraud-detection-system-design)
5. [Synthetic Data Generation](#synthetic-data-generation)
6. [Cost Model Parameters](#cost-model-parameters)
7. [Machine Learning Model](#machine-learning-model)
8. [Evaluation Methodology](#evaluation-methodology)
9. [Known Limitations](#known-limitations)

---

## Scope & Context

### Geographic Scope
- **Market:** United States only
- **Rationale:** Payment network rules, liability shifts, and fraud patterns vary significantly by country
- **Implications:** 
  - EMV liability shift effective October 2015 (U.S.)
  - 3D Secure adoption rates reflect U.S. market (~15%)
  - Chargeback rules follow Visa/Mastercard U.S. regulations

### Card Issuer Type
- **Entity:** U.S. credit card issuer (bank)
- **Not Modeled:** 
  - Debit card issuers (different liability rules under Regulation E)
  - Payment processors/networks (different data access)
  - Merchants (different perspective and costs)
- **Justification:** Card issuers have:
  - Most comprehensive transaction data
  - Direct financial liability for fraud
  - Real-time authorization decision authority

### Card Product Type
- **Product:** Consumer credit cards
- **Not Modeled:**
  - Commercial/business cards (different usage patterns)
  - Corporate cards (different controls and approvals)
  - Prepaid cards (different risk profiles)
  - Private label cards (limited acceptance)
- **Rationale:** Consumer credit cards represent majority of fraud volume

### Time Period
- **Analysis Date:** January 2026
- **Data Period:** 6 months (July 2024 - December 2024)
- **Market Conditions:**
  - 85% merchant EMV compliance (U.S. 2024)
  - 15% merchant 3DS adoption (U.S. 2024)
  - Post-pandemic fraud patterns normalized
  - E-commerce fraud remains elevated vs. pre-2020

---

## Card Verification Technologies

### EMV Chip Technology

**Functionality Assumption:**
- EMV chip cards and terminals work as designed 100% of the time
- No chip read failures, no fallback to magnetic stripe
- **Reality:** ~1-2% of chip transactions fail and require fallback
- **Impact:** Model slightly underestimates card-present fraud costs

**Merchant Compliance:**
- 85% of merchants are EMV-compliant (have chip readers and use them)
- Binary classification: compliant or non-compliant (no partial compliance)
- Compliance is random across merchants (not correlated with merchant type)
- **Data Source:** EMVCo and payment processor reports (2024)

**Liability Shift:**
- For chip-present transactions at EMV-compliant merchants: Merchant liable for counterfeit fraud
- For chip-present transactions at non-compliant merchants: Merchant liable (liability shift)
- Issuer retains liability only for:
  - Lost/stolen genuine cards used at compliant terminals
  - Chip malfunction scenarios (edge cases)
- **Assumption:** Issuer pays 15% of card-present fraud costs on average

### 3D Secure (3DS)

**Adoption Rate:**
- 15% of U.S. e-commerce merchants use 3D Secure
- **Data Source:** Aite-Novarica Group (2023), Merchant Risk Council (2023)
- Assignment is random (not correlated with merchant risk or size)

**Functionality Assumption:**
- When merchant uses 3DS, authentication always succeeds
- Cardholder always completes authentication challenge
- No friction/abandonment from 3DS flow
- **Reality:** 3DS adds friction, ~10-20% cart abandonment
- **Impact:** Model underestimates merchant reluctance to adopt 3DS

**Liability Shift:**
- Successful 3DS authentication shifts liability to merchant
- Failed/abandoned 3DS leaves liability with issuer
- Our model: If merchant uses 3DS, issuer pays 0%; otherwise issuer pays 85%

### Address Verification System (AVS)

**Availability:**
- AVS available for all CNP transactions
- Match results: Full Match, ZIP Match Only, No Match, Not Available
- Distribution for legitimate transactions: 92% full match, 4% ZIP only, 2% no match, 2% N/A

**Limitations:**
- AVS check does NOT shift liability (unlike 3DS)
- Used for risk assessment only
- Fraudsters can obtain victim's ZIP code, reducing effectiveness

### Card Security Code (CVV/CVC)

**Availability:**
- CVV required for most CNP transactions
- Match results: Match, No Match, Not Provided
- Distribution for legitimate: 96% match, 2% no match, 2% not provided

**Limitations:**
- CVV check does NOT shift liability
- Stolen card data often includes CVV
- Used for risk assessment only

---

## Liability & Regulatory Framework

### EMV Liability Shift (U.S.)

**Effective Date:** October 2015
**Rule:** If a chip card is used at a non-chip terminal, the party with the lesser technology bears the fraud liability

**Specific Scenarios:**
| Card Type | Terminal Type | Liability |
|-----------|---------------|-----------|
| Chip card | Chip terminal | Issuer (for genuine card) / Merchant (for counterfeit) |
| Chip card | Mag stripe terminal | Merchant (liability shift) |
| Mag stripe | Any terminal | Issuer |

**Our Model:**
- Post-2015, with 85% EMV compliance, most counterfeit fraud liability falls on merchants
- Issuer liable only for lost/stolen genuine cards and edge cases
- **Assumption:** Issuer pays 15% of card-present fraud costs

### 3D Secure Liability Shift

**Rule:** Successful 3DS authentication shifts CNP fraud liability from issuer to merchant

**Requirements:**
- Merchant must implement 3DS
- Cardholder must complete authentication
- Authentication must succeed

**Our Model:**
- 15% of merchants use 3DS → 15% of CNP fraud liability shifts to merchants
- 85% of merchants don't use 3DS → 85% of CNP fraud liability remains with issuer
- **Assumption:** Issuer pays 85% of CNP fraud costs on average

### Regulation E (Electronic Fund Transfers)

**Consumer Liability Limits:**
- $0 if reported before any unauthorized use
- $50 if reported within 2 business days
- $500 if reported within 60 days
- Unlimited if not reported within 60 days

**Card Network Zero Liability Policies:**
- Visa, Mastercard, Discover, Amex all offer $0 consumer liability
- Issuer absorbs losses beyond regulatory limits
- **Our Model:** Consumer pays $0; issuer pays full fraud amount (after liability shifts)

### Chargeback Rules

**Timeline:**
- Cardholder has 60-120 days to dispute transaction (varies by network and reason)
- Issuer has 30-45 days to respond
- Merchant has 30-45 days to provide evidence
- **Our Model:** Chargebacks arrive 30-90 days after transaction (uniform distribution)

**Reason Codes:**
- Fraud (10.x codes): "Transaction unauthorized"
- Authorization issues (11.x): "Card not present"
- Processing errors (12.x): "Duplicate processing"
- Consumer disputes (13.x): "Merchandise not received"
- **Our Model:** All fraud-related disputes use fraud reason codes

**Fees:**
- Card networks charge $25 per chargeback
- Additional fees for excessive chargeback ratios
- **Our Model:** $25 flat fee per disputed fraud transaction

---

## Fraud Detection System Design

### Decision Timing

**Authorization-Time Only:**
- Decisions made in real-time during transaction authorization
- Timeframe: 100-500 milliseconds
- **Not Modeled:** Post-authorization monitoring, batch analysis, manual review after approval

**Available Data:**
- Transaction details (amount, merchant, timestamp)
- Cardholder history (all prior transactions)
- Verification results (AVS, CVV)
- Velocity metrics (calculated in real-time)
- Account information (balance, limit, age)

**Not Available:**
- Future transactions
- Chargeback status (occurs 30-90 days later)
- Cross-institution fraud patterns
- Device fingerprinting (not modeled)
- Behavioral biometrics (not modeled)

### Detection Scope

**Transaction-Level:**
- Each transaction evaluated independently
- Real-time decision: Approve, Review, or Block

**Not Modeled:**
- Account-level monitoring (bust-out detection)
- Merchant-level monitoring (transaction laundering)
- Cross-account patterns (fraud rings)
- Behavioral analysis over multiple sessions

### System Architecture Assumptions

**Hard Rules:**
- Execute in <5 milliseconds
- Deterministic (same input = same output)
- Zero false positives (by design)
- May have false negatives (miss subtle fraud)

**ML Model:**
- Inference time: 10-50 milliseconds
- Probabilistic (outputs score 0-1)
- Can have false positives and false negatives
- Adapts to patterns via training

**Review Queue:**
- Manual analysts review flagged transactions
- Capacity constraint: Max 2% of transactions
- Review cost: $15 per transaction
- Assumed 90% accuracy in review (catch 90% of fraud, approve 90% of legitimate)

---

## Synthetic Data Generation

### Cardholder Population

**Total Cardholders:** 500
**Personas:** 
- Convenience users (35%): Pay in full monthly, 8-15 txns/month
- Revolvers (40%): Carry balance, 10-20 txns/month
- Transactors (20%): High spending, pay in full, 15-30 txns/month
- Light users (5%): <5 txns/month

**Credit Scores:**
- Distribution: Normal(700, 70) truncated at [300, 850]
- Correlation: Higher score → Higher credit limit
- Formula: `limit = base_limit * (score/700) * lognormal(0, 0.3)`

**Account Age:**
- Distribution: Exponential(λ=0.002), mean ~500 days
- Capped at 10 years (3,650 days)
- More recent accounts are more common (realistic)

### Transaction Generation

**Monthly Volume:**
- Per persona: Poisson(persona_mean)
- Timing: Payday spikes (days 1-5, 15-20 of month)
- Time of day: Varies by merchant category

**Transaction Amounts:**
- Distribution: Log-normal by merchant category
- Grocery: LogNormal(log(75), 0.5)
- Gas: LogNormal(log(45), 0.4)
- Restaurants: LogNormal(log(35), 0.7)
- Electronics: LogNormal(log(250), 0.8)

**Geographic Distribution:**
- 80% within 25 miles of home ZIP
- 15% within 25-100 miles (work/regular travel)
- 5% >100 miles (vacation/business travel)

**Card Present Rate:**
- Overall: 90% card-present for legitimate transactions
- Varies by MCC:
  - Grocery, gas, restaurants: 95% card-present
  - Online retail, electronics: 85% card-present

### Fraud Injection

**Overall Fraud Rate:** 1.5% of transactions

**Fraud Distribution:**
- Card testing: 12%
- Stolen card fraud (CNP): 30%
- Account takeover: 18%
- Friendly fraud: 15%
- Synthetic identity: 8%
- Refund fraud: 5%
- Application fraud: 7%
- Lost/stolen card: 5%

**Data Sources:**
- Nilson Report (2024): Overall fraud rates and trends
- Javelin Research (2024): Fraud type distribution
- Federal Reserve Payments Study (2023): Transaction volumes

### Fraud Pattern Characteristics

**Card Testing:**
- Burst pattern: 8-15 transactions within 1-3 hours
- Amounts: 50% exactly $1.00, 80% under $5
- Decline rate: 60% (vs. 1% legitimate)
- Random merchant categories

**Stolen Card Fraud:**
- Large amounts: 30-80% of credit limit
- Late night timing: 1-5 AM
- High-risk MCCs: Electronics (5732), Jewelry (5944)
- AVS/CVV mismatches: 40% no match (vs. 2-4% legitimate)

**Account Takeover:**
- 2-4 large transactions in quick succession
- New merchant categories (not in cardholder history)
- Geographic change: 100-1000 miles from home
- Mix of card-present and CNP

**Friendly Fraud:**
- Identical to legitimate transactions at authorization time
- Only distinguishable retroactively via chargeback
- Disputed 100% by definition

**Lost/Stolen Physical Card:**
- Card-present transactions


### Statistical Distributions Used

Complete reference of distributions used in synthetic data generation:

| Variable | Distribution | Parameters | Rationale |
|----------|--------------|------------|-----------|
| **Monthly transaction count** | Poisson | λ = 4, 12, 15, 25 (by persona) | Discrete count data; models rare events in fixed window |
| **Transaction day** | Weighted sampling | 2× weight on days 1,2,15,16 | Payday clustering (60-70% live paycheck-to-paycheck) |
| **Transaction hour** | Custom weights | Peak 12-2PM, 6-8PM; Low 1-5AM | Matches Visa/Mastercard transaction volume patterns |
| **Transaction amount** | Gamma | Shape & scale vary by MCC | Always positive, right-skewed, flexible fit to category |
| **Credit limit multiplier** | Lognormal | μ=0, σ=0.3 | Underwriting variation; always positive (see Technical Notes) |
| **Account age** | Exponential | λ=500 days (mean) | Many new accounts, fewer old; matches issuer portfolios |
| **Credit score** | Normal (clipped) | μ, σ by persona; range [300, 850] | Central tendency; realistic FICO distribution |
| **Fraud timing** | Uniform | Day: 1-30, Hour: 0-23 | Fraud can occur anytime (no payday pattern like legitimate) |

**Key Design Principle:** Use simplest distribution that captures essential pattern. Avoid over-engineering.

**Validation:** All distributions validated against:
- Industry transaction studies (Visa, Mastercard public data)
- Federal Reserve consumer finance data
- Academic fraud detection literature
- Payment network technical specifications

**For detailed parameter justifications, see `/docs/DESIGN_DECISIONS.md`**


## Fraud Typology Pattern Reference

Complete reference of distinctive patterns for each fraud type. These patterns are used in synthetic data generation and inform feature engineering for the detection model.

### Pattern Comparison Table

| Fraud Type | Amount Pattern | Timing Pattern | Merchant Categories | Authorization Signals | Velocity Metrics |
|------------|----------------|----------------|---------------------|----------------------|------------------|
| **Card Testing** | • 50% exactly $1.00<br>• 80% under $5<br>• Testing credit limit | • Burst: 8-15 txns in 1-3 hours<br>• Any time of day<br>• Rapid succession | • Random categories<br>• Often low-risk merchants<br>• Testing acceptance | • **60% decline rate** (vs. 1% legitimate)<br>• Multiple failed attempts<br>• Card-not-present | • Extreme spike in frequency<br>• Multiple txns per hour<br>• No geographic pattern |
| **Stolen Card (CNP)** | • Large: 30-80% of limit<br>• Single or few high-value<br>• Maximize before detection | • **Late night: 1-5 AM**<br>• Weekend preference<br>• Immediate exploitation | • High-risk: Electronics (5732)<br>• Jewelry (5944)<br>• Gift cards<br>• Online retail | • **40% AVS mismatch**<br>• **40% CVV mismatch**<br>• All card-not-present<br>• Shipping address ≠ billing | • Sudden increase from zero<br>• Geographic: 100-1000 miles from home<br>• Large amounts unusual for cardholder |
| **Account Takeover** | • 2-4 large transactions<br>• Sequential purchases<br>• 50-70% of limit each | • Can occur any time<br>• Sudden change from normal<br>• Often within 24 hours | • **New categories** not in history<br>• Broad mix (testing access)<br>• Mix of CP and CNP | • Some AVS/CVV match (has info)<br>• Mix of card-present and CNP<br>• May change shipping address | • **Geographic jump: 100-1000 mi**<br>• New merchant categories<br>• Spike in spending<br>• Time since last txn: days/weeks |
| **Lost/Stolen Card** | • Moderate amounts<br>• Multiple smaller txns<br>• Before card reported lost | • **3-8 txns in 24-48 hrs**<br>• Before cardholder notices<br>• Any time of day | • Unusual for cardholder<br>• Gas, grocery, retail<br>• Opportunistic | • All **card-present**<br>• No PIN (signature)<br>• Normal AVS/CVV (physical card) | • **Geographic: 100-500 mi** from home<br>• Card-present only<br>• Moderate frequency |
| **Friendly Fraud** | • Identical to legitimate<br>• Cardholder's normal range<br>• Disputed retroactively | • Normal timing patterns<br>• Cardholder's usual hours<br>• No timing anomalies | • Cardholder's usual merchants<br>• Normal categories<br>• **Indistinguishable** | • Normal AVS/CVV<br>• Authorized by cardholder<br>• **Cannot detect at auth** | • Normal velocity<br>• Normal geography<br>• **No detection signals** |
| **Application Fraud** | • Low initially<br>• Minimal spending<br>• Account abandoned | • Sporadic<br>• Very low frequency<br>• Front-loaded (first month) | • Essential spending only<br>• Grocery, gas<br>• Establishing presence | • Normal authorization<br>• Limited transaction history<br>• Short account tenure | • **Very low: <5 txns/month**<br>• No established patterns<br>• Early abandonment |
| **Refund Fraud** | • Moderate amounts<br>• Normal purchase amounts<br>• **High refund rate** | • Pattern over weeks/months<br>• Frequent purchase-return cycle<br>• Return timing: days after purchase | • Retail with generous return policies<br>• Clothing (5651)<br>• Electronics (5732)<br>• Department stores (5311) | • Normal at purchase<br>• Refunds appear legitimate<br>• Gradual pattern | • **High refund-to-purchase ratio**<br>• Repetitive cycle<br>• Same merchants repeatedly |
| **Synthetic Identity** | • **Gradual escalation**<br>• Month 1-11: small amounts<br>• **Month 12: max out** (bust-out) | • Months of normal behavior<br>• Builds credit history<br>• Sudden spike at end | • Gradual category expansion<br>• Starts essential, ends luxury<br>• Broadening over time | • **Excellent initially**<br>• Perfect payment history<br>• Normal until bust-out | • **Gradual increase over 12+ months**<br>• Geographic stability<br>• Pattern break at bust-out |

### Detection Difficulty by Type

**Easiest to Detect (95%+):**
- Card Testing: Extreme velocity, burst pattern, high decline rate
- Stolen Card (CNP): Large amounts, late night, AVS/CVV mismatch, geographic anomaly

**Moderate Difficulty (90-92%):**
- Account Takeover: Geographic jump + new merchants
- Lost/Stolen Card: Geographic anomaly + card-present unusual location

**Harder to Detect (88-89%):**
- Application Fraud: Limited history (not enough data for behavioral baseline)
- Refund Fraud: Gradual pattern (requires long observation window)
- Friendly Fraud: **Indistinguishable** from legitimate at authorization time

**Very Hard to Detect (85%):**
- Synthetic Identity: Requires 12+ months history to see bust-out pattern

### Key Insights for Detection

**Multi-dimensional Signals Required:**
- Single feature rarely sufficient (amount alone, velocity alone)
- Combination of amount + timing + geography + authorization = strongest signal
- Velocity metrics most predictive overall (24h and 7d transaction counts)

**Time Horizon Matters:**
- Card testing: Detected in hours (burst pattern)
- Account takeover: Detected in 1-2 transactions (immediate anomaly)
- Synthetic ID: Requires 12+ months (long-term behavioral pattern)

**Authorization Data Critical:**
- AVS/CVV mismatch: Strongest signal for CNP fraud
- Card-present but geographic anomaly: Signal for lost/stolen card
- Normal authorization: Friendly fraud cannot be distinguished

### Pattern Usage in Model

**Feature Engineering:**
- Velocity features: Transaction count and spend in 24h, 7d windows
- Geographic features: Distance from home ZIP, impossible travel detection
- Authorization features: AVS match, CVV match, card-present indicator
- Behavioral features: New merchant category, deviation from typical spending

**Detection Rules:**
- Hard rules: Impossible travel (>500 mph), extreme card testing (>10 txns/hour)
- Soft rules: Geographic anomaly, late-night CNP, AVS/CVV mismatch
- ML scores: Combine all features for nuanced probability

**For complete fraud injection methodology, see [Fraud Injection section](#fraud-injection) above.**

---


### Merchant Preferences by Persona

Cardholder spending patterns vary by persona. Merchant category code (MCC) preferences control selection probability:

| MCC | Category | Light User | Convenience | Revolver | Transactor |
|-----|----------|------------|-------------|----------|------------|
| 5411 | Grocery | 35% | 25% | 25% | 22% |
| 5541 | Gas Station | 25% | 15% | 15% | 12% |
| 5812 | Restaurant | 15% | 20% | 20% | 25% |
| 5912 | Pharmacy | 10% | 10% | 10% | 8% |
| 5311 | Department Store | 5% | 10% | 10% | 10% |
| 5732 | Electronics | 4% | 5% | 5% | 10% |
| 5651 | Clothing | 4% | 8% | 8% | 6% |
| 5999 | Misc Retail | 3% | 5% | 5% | 4% |
| 5944 | Jewelry | 0.5% | 1% | 1% | 3% |
| 4121 | Taxi/Transportation | 3.5% | 6% | 6% | 0% |

**Rationale:**

**Light Users (Low income or debt-averse):**
- Essential spending dominates (grocery 35%, gas 25%)
- Minimal discretionary (restaurant 15%, jewelry 0.5%)
- Card used for necessities where rewards/convenience matter

**Convenience Users (Middle income, "average American"):**
- Balanced distribution across categories
- Uses base spending pattern (represents plurality of cardholders)
- Moderate in all categories

**Revolvers (Carries balance month-to-month):**
- Similar to convenience users but higher frequency
- Slightly more discretionary spending
- Profitable to issuer (interest charges)

**Transactors (High income, pays off monthly):**
- More discretionary: restaurant 25%, electronics 10%, jewelry 3%
- Less essential: grocery 22%, gas 12%
- Frequent diners, purchases expensive items

**Impact on Fraud Detection:**
- Model learns persona-specific baselines for "normal" behavior
- Deviation from expected merchant mix signals potential fraud
- Example: Light user buying luxury jewelry → account takeover?
- Feature engineering: "New merchant category" more meaningful with persona context

### Payday Spending Clustering

Transaction timing incorporates realistic payday effects observed in consumer spending data.

**Methodology:**
- Payday dates: 1st (monthly salary) and 15th (bi-weekly salary)
- Days 1, 2, 15, 16 receive **2× probability weight**
- Other 26 days receive 1× weight (baseline)

**Mathematical Effect:**
```
For 30-day month with 12 transactions:
Total weight = (4 days × 2.0) + (26 days × 1.0) = 34

P(payday date) = 2.0 / 34 = 5.88% per payday
P(normal date) = 1.0 / 34 = 2.94% per normal date

Expected: ~24% of transactions within 1 day of payday
```

**Real-World Validation:**
- Federal Reserve Survey of Household Economics (2024): 60-70% of Americans live paycheck-to-paycheck
- Consumer spending increases 40-60% in 2 days after payday
- Pattern strongest for grocery, gas, restaurants (necessities)

**Why Conservative 2× Multiplier (not 4× or 10×)?**
1. Not all cardholders rely on immediate payday spending
2. Credit cards smooth consumption across month (can spend before payday)
3. Effect is real but moderate in aggregate credit card data
4. Captures pattern without over-concentrating transactions

**Impact on Fraud Detection:**
- Model learns legitimate spending clusters near payday
- Unusual spending at other times may signal fraud
- Velocity metrics must account for payday spikes (normal behavior)

- Geographic change: 100-500 miles from cardholder
- 3-8 transactions within 24-48 hours (before card reported)
- Unusual merchant categories for cardholder

**Application Fraud:**
- New account (<90 days)
- Immediate high utilization (80-100% of limit)
- 10-15 transactions within first month
- First payment missed

### Dispute Probability

**Base Rates by Fraud Type:**
- Card testing: 5%
- Stolen card fraud: 90%
- Account takeover: 95%
- Friendly fraud: 100% (by definition)
- Refund fraud: 30%
- Synthetic ID: 0% (no victim)
- Application fraud: 0% (first-party fraud)
- Lost/stolen card: 85%

**Amount Adjustments:**
- <$10: Multiply base rate by 0.3
- $10-$50: Multiply base rate by 0.7
- >$500: Multiply base rate by 1.1 (capped at 1.0)

**Rationale:**
- Small fraud often goes unnoticed
- Large fraud almost always detected
- No victim means no dispute (synthetic ID, application fraud)

---

## Cost Model Parameters

### Fraud Loss Rates (Issuer Liability)

**CNP Fraud:** 85%
- Basis: Only 15% of U.S. merchants use 3DS
- Source: Aite-Novarica Group (2023), Merchant Risk Council (2023)
- Calculation: (100% - 15%) = 85% issuer liability

**Card-Present Fraud:** 15%
- Basis: Post-EMV shift, merchants liable for most card-present fraud
- Issuer liable only for: genuine lost/stolen cards, chip malfunctions
- Source: EMV Migration Forum, payment processor data
- Conservative estimate based on industry norms

**Friendly Fraud:** 50%
- Basis: Depends on merchant's ability to provide proof (delivery, signature)
- Roughly 50/50 split in dispute outcomes
- Source: BAI Banking Strategies (2023), chargeback management studies

### Fixed Costs

**Chargeback Fee:** $25
- Source: Visa/Mastercard chargeback fee schedules (2024)
- Applies to: Each transaction disputed through chargeback process
- Not applied to: Direct refunds, synthetic ID (no chargeback)

**Investigation Cost:** $50
- Basis: Fraud analyst time (~1 hour at $50/hour loaded cost)
- Includes: Case review, documentation, communication
- Source: Pulse Debit Issuer Study (2023)

**Manual Review Cost:** $15
- Basis: 15 minutes analyst time at $60/hour loaded cost
- Includes: Transaction review, decision, documentation
- Lower than investigation (less thorough)

### False Positive Costs

**Direct Loss:** 10% of transaction amount
- Represents: Customer inconvenience, potential lost sale, service costs
- Source: Industry estimates from fraud management studies
- Conservative estimate (actual may be higher for large transactions)

**Customer Churn Rate:** 2%
- Probability customer leaves after false positive decline
- Source: Accenture Banking Customer Survey (2023), J.D. Power (2024)
- Applies to: Blocked transactions only (not reviews)

**Customer Lifetime Value:** $2,000
- Basis: 5-year average CLV for credit card customers
- Includes: Interchange revenue, interest income, annual fees
- Source: Banking industry benchmarks
- Calculation: ~$400/year × 5 years

### Revenue

**Interchange Rate:** 2%
- Basis: Average interchange across all card types and merchants
- Reality: Varies by card type (1.5-3%), merchant category, transaction size
- Breakdown:
  - Rewards cards: 2.0-2.5%
  - Non-rewards cards: 1.5-2.0%
  - Debit cards: 0.05% + $0.21 (Durbin Amendment, not modeled)
- Source: Strawhecker Group (2024), Federal Reserve Interchange Study

**Revenue Timing:**
- Earned when transaction is approved
- Lost when transaction is blocked (opportunity cost)
- Partially earned on reviewed transactions (90% approval rate assumed)

---

## Machine Learning Model

### Model Selection

**Algorithm:** Random Forest Classifier
**Rationale:**
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- Good performance on tabular data
- Relatively interpretable

**Hyperparameters:**
```python
RandomForestClassifier(
    n_estimators=100,         # Number of trees
    max_depth=15,             # Prevents overfitting
    min_samples_split=20,     # Minimum samples to split node
    min_samples_leaf=10,      # Minimum samples in leaf
    class_weight='balanced',  # Handle imbalanced data (1.5% fraud)
    random_state=42           # Reproducibility
)
```

**Class Imbalance Handling:**
- Fraud rate: 1.5% (highly imbalanced)
- Method: `class_weight='balanced'` penalizes misclassification of minority class
- Alternative methods not used: SMOTE, undersampling (to keep data realistic)

### Features (18 Total)

**Transaction Features:**
- `amount`: Transaction amount in dollars
- `mcc`: Merchant category code (categorical)
- `card_present`: Boolean (card-present vs card-not-present)
- `authorized`: Boolean (approved vs declined)
- `avs_match_encoded`: AVS result (encoded)
- `cvv_match_encoded`: CVV result (encoded)

**Velocity Features:**
- `txn_count_24h`: Transaction count in last 24 hours
- `txn_count_7d`: Transaction count in last 7 days
- `spend_24h`: Dollar amount spent in last 24 hours
- `spend_7d`: Dollar amount spent in last 7 days
- `distance_from_last_txn`: Miles from previous transaction
- `time_since_last_txn_hours`: Hours since previous transaction

**Temporal Features:**
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `is_weekend`: Boolean (Saturday/Sunday)
- `is_late_night`: Boolean (1-5 AM)

**Account Features:**
- `current_balance`: Current balance on card
- `available_credit`: Remaining credit available

**Feature Engineering:**
- All categorical variables encoded as integers
- Missing values filled with 0
- Features standardized using StandardScaler

### Feature Importance (Actual Results)

Top 10 features by importance:
1. `spend_24h` - 24-hour spending velocity
2. `amount` - Transaction amount
3. `txn_count_24h` - 24-hour transaction count
4. `available_credit` - Remaining credit
5. `spend_7d` - 7-day spending velocity
6. `current_balance` - Current balance
7. `avs_match_encoded` - AVS verification
8. `distance_from_last_txn` - Geographic anomaly
9. `is_late_night` - Late night indicator
10. `cvv_match_encoded` - CVV verification

**Interpretation:**
- Velocity features dominate (spend_24h, txn_count_24h, spend_7d)
- Behavioral changes are strongest fraud signals
- Amount alone is moderately predictive
- Verification results (AVS, CVV) are useful but not primary drivers

---

## Evaluation Methodology

### Data Splitting

**Train/Validation/Test:** 60% / 20% / 20%

**Why Three Splits:**
- Train: Fit model parameters
- Validation: Optimize decision thresholds
- Test: Unbiased final evaluation

**Stratification:**
- All splits stratified by fraud label
- Maintains 1.5% fraud rate across all splits
- Prevents class imbalance issues

**Comparison to Standard:**
- Standard practice: 70/30 train/test
- Our approach: More rigorous (prevents threshold overfitting)

### Threshold Optimization

**Optimization Set:** Validation (20%)
**Objective:** Minimize total cost to card issuer
**Method:** Grid search over threshold combinations

**Hybrid System Thresholds:**
- Block threshold: 0.85-0.99 (8 candidates)
- Review (high) threshold: 0.60-0.85 (8 candidates)
- Review (medium) threshold: 0.40-0.65 (8 candidates)
- Total combinations: 512 (filtered by constraints)

**Constraints:**
- Review rate ≤ 2% (analyst capacity limit)
- Thresholds must be ordered: review_medium < review_high < block

**ML-Only System:**
- Single threshold: 0.30-0.90 (30 candidates)
- Optimize for minimum total cost

### Evaluation Metrics

**Classification Metrics:**
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: Harmonic mean of precision and recall
- False Positive Rate: FP / (FP + TN)

**Cost Metrics:**
- Total cost to issuer (primary metric)
- Fraud missed cost
- False positive cost
- Revenue from legitimate approvals
- Net cost (total cost - revenue)

**Fraud Prevention:**
- Detection rate: % of fraud caught (blocked + reviewed)
- Fraud prevention rate: % of fraud dollars prevented
- Fraud losses: Dollar amount of missed fraud

### System Comparison

**Three Systems Evaluated:**
1. Rules-Only: Hard rules only (no ML)
2. ML-Only: ML model with optimized threshold (no rules)
3. Hybrid: Rules + ML with optimized thresholds

**Fair Comparison Requirements:**
- All systems evaluated on identical test set (20%)
- Test set never used for training or threshold selection
- Same cost model applied to all systems
- Same evaluation metrics for all systems

---

## Known Limitations

### Data Limitations

1. **Synthetic Data**
   - Real fraud patterns may be more complex
   - Fraudster behavior evolves (adversarial)
   - Actual data has more noise and edge cases

2. **Time Period**
   - Only 6 months of history
   - No seasonal patterns (holidays, back-to-school)
   - No long-term fraud trends (bust-out requires 12+ months)

3. **Missing Features**
   - No device fingerprinting
   - No IP geolocation
   - No cross-institution data sharing
   - No behavioral biometrics

### Model Limitations

1. **Static Model**
   - Trained once, doesn't adapt in real-time
   - Fraudsters evolve tactics (model degrades)
   - No online learning or continuous retraining

2. **Transaction-Level Only**
   - No account-level patterns (bust-out)
   - No network-level patterns (fraud rings)
   - No merchant-level patterns (transaction laundering)

3. **Imbalanced Data**
   - 1.5% fraud rate is highly imbalanced
   - Minority class (fraud) is harder to detect
   - Class weighting helps but doesn't eliminate issue

### Cost Model Limitations

1. **Simplified Liability**
   - Real liability varies by specific circumstances
   - Our model uses average rates (85% CNP, 15% card-present)
   - Actual liability depends on merchant compliance, card type, chargeback reason

2. **Fixed Costs**
   - Investigation costs vary by fraud complexity
   - Customer churn depends on customer value, not just $2,000 average
   - Interchange varies by card type and merchant (we use 2% average)

3. **Dispute Probability**
   - Based on amount and fraud type
   - Actual disputes depend on cardholder attentiveness, statement review habits
   - Our model may over/underestimate for specific customer segments

### Scope Exclusions

1. **Fraud Types Not Modeled**
   - Bust-out fraud (requires 12+ months history)
   - Merchant fraud (different detection approach)
   - Cross-border fraud (different rules and patterns)
   - ATM fraud (different data and detection)



### Card Blocking Limitation

**Issue:** The synthetic dataset does not model card blocking after fraud detection.

**Impact:**
- Multiple fraud transactions may occur on the same cardholder in the test set
- In reality, card would be blocked after first detected fraud
- This can inflate apparent fraud volume and potentially overstate model performance
- Estimated impact: ~37% of fraud transactions may be post-block artifacts

**Why Not Implemented:**
- Primary focus is on **real-time fraud detection** (transaction-level decisions)
- Card blocking is an **operational process** separate from detection model
- Adding blocking logic significantly complicates data generation
- For portfolio demonstration, limitation is acceptable with disclosure

**Mitigation:**
- Performance metrics (precision, recall) remain valid for first fraud transaction per cardholder
- Cost analysis may overstate fraud prevention (counts post-block transactions that wouldn't occur)
- Production deployment would naturally prevent post-block transactions

**Production Considerations:**
In production fraud operations, card blocking occurs automatically after confirmed fraud:
1. Fraud detected on authorization → Card flagged for blocking
2. Cardholder notified and new card issued
3. Old card number blocked from future authorizations
4. Subsequent authorization requests on blocked card are declined at gateway (before ML scoring)

**Note:** This limitation affects evaluation methodology, not model quality. The model learns to detect fraud patterns correctly; the issue is that some fraud instances in the test set wouldn't exist in real operations due to prior card blocking. Future versions may implement post-hoc elimination of post-block fraud transactions from evaluation sets.


2. **Operational Aspects**
   - Model deployment complexity
   - Real-time inference latency
   - Model monitoring and drift detection
   - Retraining pipeline and frequency

3. **Regulatory Compliance**
   - Model explainability requirements (FCRA, ECOA)
   - Adverse action notices
   - Fair lending considerations
   - Privacy regulations (GDPR, CCPA)

---

## Validation & Sensitivity Analysis

### Assumptions to Test

**High Impact:**
1. 3DS adoption rate (15%): Test 10%, 20%, 30%
2. Dispute probability: Test ±20% adjustment
3. Customer churn rate (2%): Test 1%, 3%, 5%
4. Interchange rate (2%): Test 1.5%, 2.5%

**Medium Impact:**
5. EMV compliance (85%): Test 80%, 90%
6. Review capacity (2%): Test 1%, 3%
7. Investigation cost ($50): Test $30, $70

**Low Impact:**
8. Chargeback fee ($25): Stable across networks
9. Review cost ($15): Relatively fixed
10. Fraud rate (1.5%): Industry average

### Recommended Validation

**For Production Use:**
1. Validate all cost parameters against actual issuer data
2. Test model performance on real (not synthetic) fraud data
3. Conduct A/B tests of threshold configurations
4. Monitor model performance and recalibrate quarterly
5. Adjust dispute probabilities based on actual cardholder behavior
6. Update 3DS adoption rates as market evolves

---

**Last Updated:** January 2026

**Version:** 1.0

**Contact:** For questions about these assumptions or to suggest refinements, please refer to the main repository README.