# Credit Card Fraud Detection: Hybrid ML + Rules System

## Executive Summary

This project demonstrates a fraud detection system for credit card transactions, combining rule-based logic with machine learning to balance fraud detection and customer experience.

**Results on Test Set:**
- 💰 Cost reduction compared to rules-only approach
- 🎯 94%+ fraud detection rate with <0.5% false positive rate
- ⚖️ Threshold optimization using proper train/validation/test methodology
- 📊 8 fraud typologies with realistic patterns

---


### Key Results (Simulation on Synthetic Data)

**Cost Comparison:**

| System | Total Cost | vs. Rules-Only | vs. ML-Only | Key Advantage |
|--------|------------|----------------|-------------|---------------|
| **Rules-Only** | $314,245 | Baseline | +$152,389 | High precision (98%), explainable |
| **ML-Only** | $180,334 | -43% | Baseline | High recall (92%), nuanced |
| **Hybrid** | **$161,856** | **-48%** | **-10%** | **Best of both worlds** |

*Note: Results from simulation on 47,109 synthetic transactions (1.63% fraud rate), optimized on validation set, evaluated on hold-out test set.*

**Detection by Fraud Type:**

| Fraud Type | % of Fraud | Detection Rate | Primary Detection Signal |
|------------|-----------|----------------|-------------------------|
| Card Testing | 12% | 98% | Burst pattern, high decline rate |
| Stolen Card (CNP) | 30% | 95% | Large amounts, late night, AVS mismatch |
| Account Takeover | 18% | 92% | Geographic anomaly, new merchants |
| Lost/Stolen Card | 5% | 90% | Card-present, unusual location |
| Application Fraud | 7% | 89% | First-party, limited history |
| Friendly Fraud | 15% | 88% | Indistinguishable at authorization |
| Refund Fraud | 5% | 88% | Pattern emerges over time |
| Synthetic Identity | 8% | 85% | Bust-out pattern (12+ months) |

*For detailed fraud pattern specifications (amount, timing, merchant categories, velocity), see [Fraud Typology Reference](docs/assumptions.md#fraud-typology-pattern-reference).*

**Threshold Optimization Impact:**

| Threshold Configuration | Total Cost | Improvement |
|------------------------|------------|-------------|
| Arbitrary defaults (0.5, 0.7, 0.95) | $202,143 | Baseline |
| Cost-optimized (0.68, 0.82, 0.94) | $161,856 | **-$40,287 (-20%)** |

*Optimization: Grid search over 512 combinations on validation set to minimize total issuer cost (fraud losses + false positive costs + operations).*


---

## Project Context: Building on Prior Work

This project extends my previous fraud detection explorations by focusing on production deployment aspects and cost modeling that are critical for real-world systems but often overlooked in academic approaches.

### Previous Projects: ML Fundamentals

My prior fraud detection projects covered core machine learning challenges:
- **Imbalanced learning techniques**: SMOTE, undersampling, class weights for handling 1-2% fraud rates
- **Prequential validation**: Time-series cross-validation to avoid data leakage
- **Hyperparameter optimization**: Grid search, random search
- **Standard ML evaluation**: Precision, recall, F1-score, ROC-AUC

These projects established strong ML foundations but left production deployment questions unanswered.

### This Project: Production Realities

This analysis addresses gaps between academic machine learning and production fraud systems:

**1. Hybrid Rule-ML Architecture**
- *Why*: Production systems rarely use ML-only or rules-only approaches
- *Exploration*: How to optimally combine deterministic rules with probabilistic ML
- *Finding*: Hybrid achieves 52% cost reduction vs. rules-only, 18% vs. ML-only

**2. Fraud Typology Detection**
- *Why*: Different fraud types exhibit different patterns and detection difficulty
- *Exploration*: Performance breakdown across 8 common fraud typologies
- *Finding*: Card testing (98% detection) vs. friendly fraud (88% detection)

**3. Cost-Based Threshold Optimization**
- *Why*: Academic metrics (accuracy, F1) ignore business costs
- *Exploration*: Optimize decision thresholds to minimize total cost to card issuer
- *Finding*: $40K cost savings vs. arbitrary thresholds (0.5, 0.7, 0.95)

**4. Realistic Issuer Cost Components**
- *Why*: Fraud cost ≠ transaction amount due to liability rules and business impact
- *Exploration*: Model CNP liability (85%), EMV shift (15%), interchange loss (2%), customer churn
- *Finding*: False positives can cost more than fraud on large transactions

**5. Operational Constraints: Card Blocking**
- *Why*: Real systems block cards after fraud detection, affecting subsequent transactions
- *Exploration*: Post-hoc elimination of fraud that wouldn't occur due to blocking
- *Finding*: ~37% of fraud transactions in test set are post-block artifacts (inflate metrics)

**6. Dispute Probability Modeling**
- *Why*: Not all fraud is disputed by cardholders; small amounts often go unreported
- *Exploration*: Amount-based dispute probability (30% for <$10, 95% for >$500)
- *Finding*: Changes chargeback cost estimates and decision thresholds significantly

### What This Project Adds to Portfolio

**Demonstrates:**
- Understanding of production ML deployment beyond algorithm selection
- Domain knowledge of payment industry (liability rules, EMV, 3D Secure, chargebacks)
- Cost-benefit analysis and business metrics (not just technical metrics)
- Systems thinking: interactions between rules, ML, operations, and economics

**Complements prior work by showing:**
- How to bridge academic ML → production systems
- How to incorporate payment network rules and liability standards
- How to evaluate models on realistic operational scenarios
- How to optimize for business outcomes (cost) vs. ML metrics (F1)

---

## Installation

### Option 1: Flexible Installation (Recommended)

```bash
pip install -r requirements.txt
```

**Use this to get the latest compatible versions of all libraries.**

✅ **Benefits:**
- Security patches and bug fixes
- Performance improvements
- Maintained compatibility

⚠️ **Trade-off:** Results may differ slightly from original analysis due to minor version differences (typically <1% variance in metrics).

### Option 2: Exact Reproduction

```bash
pip install -r requirements-lock.txt
```

**Use this to replicate the exact development environment.**

✅ **Benefits:**
- Identical numerical results
- Perfect reproducibility for academic validation
- No unexpected behavior from library updates

⚠️ **Trade-off:** Won't receive latest bug fixes and security patches until lock file is regenerated.

### Recommendation

- **For development and exploration:** Use `requirements.txt`
- **For exact paper reproduction:** Use `requirements-lock.txt`
- **For production deployment:** Generate your own lock file from tested environment

### Setup Instructions

```bash
# 1. Clone repository
git clone https://github.com/yourusername/fraud-detection-portfolio.git
cd fraud-detection-portfolio

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies (choose one)
pip install -r requirements.txt          # Flexible (recommended)
# OR
pip install -r requirements-lock.txt     # Exact reproduction

# 5. Verify installation
python -c "import pandas, numpy, sklearn; print('Success!')"
```


---

## Analysis Overview

### System Architecture

**Three-Layer Approach:**

1. **Hard Rules** (Immediate block)
   - Impossible travel (>500 mph between transactions)
   - Card testing patterns (burst of small transactions, high decline rate)

2. **ML Model** (Random Forest)
   - Trained on transaction features, velocity metrics, verification results
   - Outputs fraud probability score (0-1)
   - 18 features including amount, MCC, temporal patterns, velocity

   - **Hyperparameters:** Default scikit-learn values with `class_weight='balanced'` to handle 1.5% fraud rate
   - **Rationale:** Defaults achieved excellent performance (94% precision/recall); focus is on production architecture and cost optimization, not hyperparameter tuning (covered in prior projects)

3. **Hybrid Decision Engine**
   - Combines rules + ML scores with optimized thresholds
   - Block: Hard rule violated OR ML score ≥ optimized threshold
   - Review: ML score above review threshold OR (medium score AND soft rule)
   - Approve: Otherwise

### Methodology

**Proper Train/Validation/Test Split:**

```
Dataset Split:
├── Train (60%) ────────→ Train ML model
├── Validation (20%) ───→ Optimize decision thresholds
└── Test (20%) ─────────→ Final unbiased evaluation
```

Decision thresholds are optimized on the validation set to minimize total cost to the card issuer, then all three systems (Rules-Only, ML-Only, Hybrid) are evaluated on the identical test set for fair comparison.

---

## Fraud Typologies Implemented

### Transaction Data Includes:

1. **Card Testing** (12% of fraud) - Bursts of small transactions to verify stolen cards
2. **Stolen Card Fraud** (30% of fraud) - CNP fraud using stolen card credentials
3. **Account Takeover** (18% of fraud) - Compromised login credentials, unusual spending
4. **Friendly Fraud** (15% of fraud) - Legitimate cardholder disputes valid charge
5. **Synthetic Identity** (8% of fraud) - Fake identity using real/fake info mix
6. **Refund Fraud** (5% of fraud) - Excessive returns of purchased items
7. **Application Fraud** (7% of fraud) - First-party fraud with no intent to pay
8. **Lost/Stolen Physical Card** (5% of fraud) - Physical card theft before reported

Each fraud type exhibits distinctive patterns in transaction amount, timing, merchant category, authorization rates, and velocity metrics.

---

## Cost Model (Card Issuer Perspective)

### Liability Rules

Based on U.S. payment network rules and EMV liability shift (2015):

| Fraud Type | Issuer Pays | Basis |
|------------|-------------|-------|
| CNP Fraud | 85% | Only ~15% of U.S. merchants use 3D Secure |
| Card-Present | 15% | Post-EMV shift, merchants liable for most fraud |
| Friendly Fraud | 50% | Split based on dispute outcomes |

### Cost Components

**Per Transaction:**
- **Fraud losses**: Varies by type (see table above)
- **Chargeback fee**: $25 per chargeback
- **Investigation**: $50 per fraud case
- **Interchange revenue**: 2% of transaction amount (earned on approved transactions)

**False Positives:**
- Direct cost: 10% of transaction amount (customer inconvenience)
- Customer churn: 2% probability × $2,000 lifetime value
- Lost interchange: 2% of transaction amount

**Operational:**
- Manual review: $15 per transaction

### Dispute Probability

Not all fraud is disputed by cardholders:
- Small amounts (<$10): 30% dispute rate
- Medium amounts ($10-$500): 70% dispute rate  
- Large amounts (>$500): 95% dispute rate
- Synthetic ID/Application fraud: 0% (no victim)

Only disputed fraud incurs chargeback fees and investigation costs.

---

## Technical Details

### Feature Engineering

**Transaction-Level:**
- Amount, merchant category code (MCC), card present/not present
- Authorization result, AVS/CVV match
- Time of day, day of week, late night indicator

**Velocity Features (Most Important):**
- Transaction count (24h, 7d windows)
- Spend amount (24h, 7d windows)  
- Time since last transaction
- Distance from last transaction

**Account-Level:**
- Current balance, available credit
- Account age, credit limit

### Model Details

**Random Forest Classifier:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',  # Handle 1.5% fraud rate
    random_state=42
)
```

**Top 5 Most Important Features:**
1. 24-hour spending velocity
2. Transaction amount
3. 24-hour transaction count
4. Available credit
5. 7-day spending velocity

---

## Repository Structure

```
fraud-detection-portfolio/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── generate_synthetic_data.py         # Data generation
├── fraud_detection_models.py       # ML + Rules system
├── fraud_cost_analysis.py             # Cost calculations
├── fraud_pattern_visualization.py     # Pattern analysis
├── results/
│   ├── cardholders.csv               # Generated profiles
│   ├── transactions.csv              # Generated transactions
│   ├── hybrid_evaluation_test.csv    # Test set predictions
│   ├── cost_comparison_results.csv   # Cost analysis
│   ├── system_comparison_test_set.png
│   ├── cost_comparison_analysis.png
│   └── threshold_optimization_analysis.png
└── docs/
    └── assumptions.md                 # Detailed assumptions
```

---

### Installation

**Standard installation (recommended):**
```bash
pip install -r requirements.txt
```

**Exact reproduction (matches paper results):**
```bash
pip install -r requirements-lock.txt
```

**Note:** requirements-lock.txt contains exact versions used to generate
results in this analysis. Use this if you encounter version-related issues.

---

## Quick Start

### Prerequisites
```bash
Python 3.8+
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Run Full Analysis
```bash
# 1. Generate synthetic data (500 cardholders, 6 months, 1.5% fraud rate)
python generate_synthetic_data.py

# 2. Train models and evaluate systems
python fraud_detection_models.py

# 3. Calculate costs and ROI
python fraud_cost_analysis.py

# 4. Visualize fraud patterns (optional)
python fraud_pattern_visualization.py
```

---

## Key Assumptions

### Model Scope & Context
- **Geography**: U.S. credit card market
- **Issuer Type**: Credit card issuer (not debit, not merchant, not processor)
- **Card Type**: Consumer credit cards (not commercial/business cards)
- **Time Period**: 2024 (reflects current EMV adoption, 3DS rates)
- **Analysis Window**: 6 months of transaction history

### Card Verification & Compliance
- **EMV chip technology**: Works as intended (no chip failures modeled)
- **3D Secure**: Authentication always succeeds when attempted
- **Merchant EMV compliance**: 85% of merchants (U.S. 2024 estimate)
- **Merchant 3DS adoption**: 15% for CNP transactions (U.S. average)
- **AVS/CVV**: Available for verification but doesn't shift liability

### Liability & Regulatory
- **EMV liability shift**: October 2015 U.S. implementation applies
- **3DS liability shift**: Applies when merchant uses it
- **Regulation E**: Consumer protections apply ($0-$50 consumer liability)
- **Chargeback window**: 30-90 days after transaction
- **Card network rules**: Visa/Mastercard U.S. rules as of 2024

### Fraud Detection System
- **Decision timing**: Authorization-time only (real-time)
- **Detection scope**: Transaction-level (not account-level monitoring)
- **Data sharing**: No cross-institution fraud sharing modeled
- **Advanced features**: No device fingerprinting or behavioral biometrics

### Data Generation
- **Synthetic data**: Matches industry fraud patterns and distributions
- **Fraud rate**: 1.5% overall (industry average)
- **Fraud distribution**: Based on 2023-2024 industry reports
- **Seasonality**: None (constant fraud rate over 6 months)
- **Fraud rings**: No coordinated attacks modeled

### Cost Model Parameters
- **Interchange revenue**: 2% of transaction (average across card types)
- **Chargeback fee**: $25 (Visa/Mastercard average)
- **Investigation cost**: $50 per case (analyst hourly rate)
- **False positive friction**: 10% of transaction amount
- **Customer churn rate**: 2% after false positive
- **Customer lifetime value**: $2,000 (5-year average)
- **Manual review cost**: $15 per transaction

### Exclusions / Out of Scope
- Bust-out fraud (requires 12+ months of history; our data covers 6 months)
- Merchant fraud / transaction laundering
- Cross-border transactions
- Cryptocurrency purchases
- ATM fraud
- Account opening fraud (application screening)

---

## References

**Note:** This analysis prioritizes freely accessible sources for verification and reproducibility. Where industry reports are cited, key statistics are validated against government data and public sources.

### Fraud Statistics & Trends

**Government & Public Sources (Free):**
- Federal Reserve (2023). "The Federal Reserve Payments Study: 2023 Triennial Initial Data Release." Available at: https://www.federalreserve.gov/paymentsystems/
- Federal Trade Commission (2024). "Consumer Sentinel Network Data Book." Available at: https://www.ftc.gov/reports/consumer-sentinel-network  
- Bureau of Justice Statistics (2023). "Victims of Identity Theft, 2021." Available at: https://bjs.ojp.gov/identity-theft

**Academic Research (Open Access):**
- Randhawa, K., et al. (2018). "Credit Card Fraud Detection using AdaBoost and Majority Voting." *IEEE Access*, 6, 14277-14284. DOI: 10.1109/ACCESS.2018.2806420
- Dal Pozzolo, A., et al. (2015). "Calibrating Probability with Undersampling for Unbalanced Classification." *IEEE Symposium Series on Computational Intelligence*.

### Payment Network Rules & Standards

**Industry Standards (Free with Registration):**
- Visa (2024). "Visa Core Rules and Visa Product and Service Rules." Available at: https://usa.visa.com/support/merchant/library/visa-rules.html
- Mastercard (2024). "Rules & Standards." Available at: https://www.mastercard.us/en-us/merchants/safety-security/security-recommendations/rules-and-standards.html  
- EMVCo (2024). "EMV Payment Tokenization Specification – Technical Framework." Available at: https://www.emvco.com/specifications/

**Regulatory (Public Domain):**
- Federal Reserve. "Regulation E: Electronic Fund Transfers" (12 CFR Part 1005). Available at: https://www.ecfr.gov/
- Consumer Financial Protection Bureau. "Regulation E (Electronic Fund Transfers)." Available at: https://www.consumerfinance.gov/rules-policy/regulations/1005/

### Machine Learning & Detection Methods

**Academic (Open Access):**
- Carcillo, F., et al. (2018). "SCARFF: A Scalable Framework for Streaming Credit Card Fraud Detection with Spark." *Information Fusion*, 41, 182-194. (Available via ResearchGate)
- Khatri, S., et al. (2020). "Credit Card Fraud Detection in the Era of Disruptive Technologies: A Systematic Review." *Journal of King Saud University - Computer and Information Sciences*. (Open Access)

**Industry White Papers (Free Download):**
- FICO (2023). "Falcon Fraud Manager: Advanced Analytics for Card Fraud Detection." Available at: https://www.fico.com/falcon-platform
- LexisNexis Risk Solutions (2024). "True Cost of Fraud Study: Financial Services & Lending Edition." Available at: https://risk.lexisnexis.com/ (Free with email registration)
- Stripe (2024). "The State of Online Fraud." Available at: https://stripe.com/resources (Free download)

### Cost & Economic Estimates

**Methodology Note:** Cost parameters in this analysis (chargeback fees, investigation costs, interchange rates) are based on payment network rules (Visa, Mastercard) and industry standards. Liability percentages (EMV shift, 3DS adoption) are estimated from publicly available merchant surveys, government fraud statistics, and industry reports.

**Key Cost Assumptions:**
- Chargeback fee: $25 per chargeback (Visa/Mastercard network standard)
- Investigation cost: $50 per fraud case (industry average from public sources)
- Interchange revenue: 2% of transaction amount (varies by card type and merchant category)
- 3D Secure adoption: ~15% of U.S. online merchants (estimated from industry surveys)
- EMV liability shift: Effective October 2015 (merchants liable for card-present fraud without EMV)

For detailed cost modeling assumptions, see `docs/assumptions.md`.

### Technical Implementation

**Open Source & Documentation:**
- Python pandas documentation: https://pandas.pydata.org/docs/
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Python-pptx documentation: https://python-pptx.readthedocs.io/

---

## Analysis Notes

### Observations from Model Training

**Feature Importance:**
- Velocity metrics (24h/7d transaction counts and spend) are the most predictive features
- This aligns with industry literature indicating that behavioral changes are strong fraud signals
- Amount alone is moderately predictive; context (velocity, timing) matters more

**Detection vs. False Positive Trade-off:**
- Rules-only achieves very high precision (98%+) but lower recall (67%)
- ML-only achieves higher recall (92%+) but more false positives
- Hybrid approach balances both, achieving 94% precision and 94% recall

**Threshold Optimization Impact:**
- Optimizing thresholds on validation data (vs. using arbitrary values like 0.95, 0.7, 0.5) reduces total cost by ~$40K
- Demonstrates importance of data-driven threshold selection

### Cost Analysis Observations

**Issuer Liability Concentration:**
- CNP fraud drives 85% of issuer fraud costs despite being preventable with 3DS
- Low 3DS adoption (15% of merchants) leaves issuers exposed
- Card-present fraud largely shifted to merchants post-EMV (2015)

**False Positive Economics:**
- Blocking legitimate transactions costs more than just customer friction
- Lost interchange revenue (2% per transaction) is significant
- Customer churn risk (2% × $2,000 CLV) dominates large transaction declines

**Dispute Probability Impact:**
- Small fraud often goes unnoticed/undisputed, reducing actual issuer costs
- This benefits rules-only systems (which miss small fraud) more than ML systems
- Highlights importance of modeling actual cardholder behavior, not just fraud occurrence

---

## Future Enhancements

**Data & Features:**
- Device fingerprinting and IP geolocation data
- Cross-account pattern detection (fraud rings)
- Merchant risk scoring based on historical fraud rates
- Customer service interaction history
- Behavioral biometrics (typing patterns, session behavior)

**Modeling:**
- Deep learning models (LSTM for sequential patterns)
- Graph neural networks (detect fraud rings)
- Online learning (model updates daily with new fraud patterns)
- Ensemble methods (combine multiple ML algorithms)
- Explainable AI (SHAP values for individual predictions)

**Detection Scope:**
- Post-authorization monitoring (not just real-time)
- Account-level fraud detection (bust-out, synthetic ID)
- Cross-channel analysis (online + in-store + ATM)
- Merchant fraud detection (transaction laundering)

**Production Readiness:**
- Real-time streaming pipeline (Kafka, Flink)
- Model monitoring and drift detection
- A/B testing framework for threshold variations
- Automated retraining pipeline
- API deployment (Docker, Kubernetes)

---

## Learning Objectives Demonstrated

This project showcases:

1. **End-to-End ML Pipeline**: Data generation → Feature engineering → Model training → Threshold optimization → Evaluation → Cost analysis

2. **Proper ML Methodology**: Train/validation/test split (60/20/20), threshold optimization on validation (not test), unbiased evaluation, cost-based optimization (not just accuracy)

3. **Domain Knowledge**: Understanding of fraud typologies, payment network liability rules, dispute processes, issuer economics including interchange revenue

4. **Business Impact Analysis**: Financial cost modeling, ROI calculation, trade-off analysis (fraud prevention vs. customer friction)

5. **Software Engineering**: Modular code, clear documentation, reproducible analysis, version control best practices

---

## Methodology Notes

**Why Train/Validation/Test?**

Standard practice is train/test (70/30), but this project uses train/validation/test (60/20/20) because:
- Decision thresholds must be optimized on held-out data
- Optimizing on test set would leak information and bias results
- Validation set allows fair threshold selection, test set provides unbiased final evaluation
- This approach follows ML best practices for hyperparameter tuning

**Why Cost-Based Optimization?**

Optimizing for accuracy or F1-score ignores business reality:
- False positives have different costs than false negatives
- A declined $5,000 transaction has different impact than a declined $10 transaction
- Customer churn from false positives can exceed fraud losses
- Interchange revenue makes blocking transactions costly (lost revenue)

Total cost to issuer is the appropriate optimization metric.

**Why Issuer Perspective?**

Card issuers have:
- The most comprehensive transaction data access
- Direct financial stake in fraud losses
- Ability to make real-time authorization decisions
- Regulatory responsibility for consumer protection

This makes the issuer perspective most relevant for fraud detection system evaluation.

---

## License & Usage

This project is for portfolio demonstration and educational purposes. The synthetic data generation methodology and detection framework are available for learning and reference.

**Note:** This analysis uses synthetic data. Real-world fraud patterns may be more complex. Always validate models on production data before deployment.

---


---

## Repository Structure

```
fraud-detection-portfolio/
├── README.md                          # Main documentation
├── QUICKSTART.md                      # Step-by-step execution guide
├── requirements.txt                   # Flexible dependencies (recommended)
├── requirements-lock.txt              # Exact versions for reproduction
├── generate_synthetic_data.py         # Synthetic data generation
├── fraud_detection_models.py          # Model training and evaluation
├── fraud_cost_analysis.py             # Cost modeling and comparison
├── fraud_pattern_visualization.py     # Fraud pattern analysis
├── docs/                              # Documentation
│   ├── EXECUTIVE_SUMMARY.md          # Business case (2-page)
│   ├── assumptions.md                 # Detailed assumptions
│   └── DESIGN_DECISIONS.md           # Technical parameter justifications
├── notebooks/                         # Jupyter notebooks (exploratory)
│   ├── synthetic-fraud-data-generator.ipynb
│   └── hybrid-detection-system-w-typologies.ipynb
└── results/                           # Generated outputs
    ├── cardholders.csv                # Synthetic cardholder data
    ├── transactions.csv               # Synthetic transaction data
    ├── *.png                          # Visualizations
    └── *.csv                          # Evaluation results
```


### Purpose of Notebooks vs. Scripts

**Python Scripts (.py files):**
- Complete production-ready implementation
- Run from command line for full analysis
- Include all optimizations and 8 fraud typologies
- Generate fresh results (~20 minutes runtime)
- Best for: Reproducibility, command-line workflows, production deployment

**Jupyter Notebook (.ipynb file):**
- Interactive walkthrough with explanations
- Step-by-step demonstration of methodology
- Visualizations and intermediate results
- Educational narrative with markdown commentary
- Best for: Learning the approach, presentations, portfolio demonstration

**Recommendation:** Review the notebook first to understand the methodology, then run the Python scripts for complete analysis with your own parameters.


## Contact

**Project Type:** Portfolio Demonstration  
**Domain:** Financial Services - Fraud Detection  
**Skills:** Machine Learning, Feature Engineering, Cost Optimization, Python, Scikit-learn

---

**Last Updated:** January 2026