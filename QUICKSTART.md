# Quick Start Guide: Credit Card Fraud Detection System

## Overview

This guide provides step-by-step instructions to generate synthetic data, train the fraud detection models, and evaluate the hybrid system.

**Time to Complete:** 30-45 minutes  
**Prerequisites:** Python 3.8+, basic command line familiarity

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Generate Synthetic Data](#generate-synthetic-data)
3. [Run Model Training & Evaluation](#run-model-training--evaluation)
4. [Calculate Costs & ROI](#calculate-costs--roi)
5. [Generate Visualizations (Optional)](#generate-visualizations-optional)
6. [Review Results](#review-results)
7. [Jupyter Notebook Analysis (Optional)](#jupyter-notebook-analysis-optional)
8. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Step 1: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

**Required Packages:**
- pandas (≥1.3.0) - Data manipulation
- numpy (≥1.20.0) - Numerical computing
- scikit-learn (≥0.24.0) - Machine learning
- matplotlib (≥3.3.0) - Plotting
- seaborn (≥0.11.0) - Statistical visualizations
- scipy (≥1.6.0) - Statistical functions

### Step 2: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, scipy; print('All packages installed successfully!')"
```

---

## Generate Synthetic Data

### Step 1: Run Data Generator

```bash
python generate_synthetic_data.py
```

**What This Does:**
- Generates 500 cardholder profiles with diverse spending personas
- Creates 6 months of transaction history (~45,000-50,000 transactions)
- Injects fraud across 8 typologies at 1.5% rate (industry average)
- Adds velocity metrics and verification features

**Expected Runtime:** 2-5 minutes

**Output Files Created:**
- `cardholders.csv` - Cardholder profiles (500 rows)
- `transactions.csv` - Transaction dataset with fraud labels (~47,000 rows)

### Step 2: Verify Data Generation

Check that files were created:

```bash
# On Windows:
dir cardholders.csv transactions.csv

# On macOS/Linux:
ls -lh cardholders.csv transactions.csv
```

**Expected Output:**
```
cardholders.csv      ~50 KB
transactions.csv     ~8-10 MB
```

Quick data check:

```bash
# View first few rows
python -c "import pandas as pd; df = pd.read_csv('transactions.csv'); print(f'Loaded {len(df):,} transactions'); print(f'Fraud rate: {df[\"is_fraud\"].mean():.2%}')"
```

**Expected:** ~47,000 transactions with ~1.5% fraud rate

---

## Run Model Training & Evaluation

### Step 1: Train Models and Evaluate Systems

```bash
python fraud_detection_models.py
```

**What This Does:**
1. Loads transaction data
2. Splits into train (60%), validation (20%), test (20%)
3. Engineers 18 features including velocity metrics
4. Trains Random Forest classifier on training set
5. Optimizes thresholds on validation set
6. Evaluates three systems on test set:
   - Rules-Only
   - ML-Only
   - Hybrid (Rules + ML)

**Expected Runtime:** 5-10 minutes

**Progress Indicators:**
```
FRAUD DETECTION MODELS
==========================================================
✓ Loaded 47,108 transactions
✓ Train/Validation/Test split complete
✓ Feature engineering complete (18 features)
✓ ML model trained (Random Forest, 100 trees)
✓ Validation set optimization complete
✓ Test set evaluation complete
```

### Step 2: Review Console Output

The script prints comprehensive results including:
- Dataset statistics
- Fraud type distribution
- Model performance metrics (precision, recall, F1)
- System comparison on test set
- Cost breakdown by system

**Key Metrics to Look For:**
- **Hybrid System**: 94%+ precision and recall
- **False Positive Rate**: <0.5%
- **Cost Reduction**: ~50% vs Rules-Only

### Step 3: Check Output Files

```bash
# On Windows:
dir *_test.csv

# On macOS/Linux:
ls -lh *_test.csv
```

**Output Files Created:**
- `ml_evaluation_test.csv` - ML model predictions on test set
- `hybrid_evaluation_test.csv` - Hybrid system decisions on test set
- `system_comparison_test_set.png` - Visual comparison chart

---

## Calculate Costs & ROI

### Step 1: Run Cost Analysis

```bash
python fraud_cost_analysis.py
```

**What This Does:**
1. Loads test set evaluation results
2. Calculates costs for each system:
   - Fraud losses (by type and liability)
   - Chargeback fees
   - Investigation costs
   - False positive costs
   - Manual review costs
   - Interchange revenue
3. Compares total costs across systems
4. Generates cost breakdown visualizations

**Expected Runtime:** 1-2 minutes

### Step 2: Review Cost Results

**Console Output Includes:**
- Total cost by system (Rules-Only, ML-Only, Hybrid)
- Cost breakdown by component
- Savings calculations
- ROI analysis

**Expected Results:**
```
COST COMPARISON:
  Rules-Only: $314,000
  ML-Only:    $198,000
  Hybrid:     $162,000

SAVINGS:
  Hybrid vs Rules-Only: $152,000 (48% reduction)
  Hybrid vs ML-Only:     $36,000 (18% reduction)
```

### Step 3: Check Output Files

**Output Files Created:**
- `cost_comparison_results.csv` - Detailed cost breakdown
- `cost_comparison_analysis.png` - Cost visualization
- `threshold_optimization_analysis.png` - Threshold impact chart

---

## Generate Visualizations (Optional)

### Step 1: Run Visualization Script

```bash
python fraud_pattern_visualization.py
```

**What This Does:**
- Creates detailed visualizations for each of 8 fraud types
- Generates summary dashboard
- Saves high-resolution PNG files (300 DPI)

**Expected Runtime:** 3-5 minutes

**Output Files Created (9 total):**
1. `card_testing_patterns.png`
2. `cnp_fraud_patterns.png` (stolen_card_fraud)
3. `ato_fraud_patterns.png` (account_takeover)
4. `friendly_fraud_patterns.png`
5. `refund_fraud_patterns.png`
6. `lost_stolen_card_patterns.png`
7. `application_fraud_patterns.png`
8. `synthetic_id_patterns.png` (if present in data)
9. `fraud_summary_dashboard.png`

**Note:** Visualizations are optional but helpful for understanding fraud patterns.

---

## Review Results

### File Overview

After running all scripts, you should have:

```
fraud-detection-portfolio/
├── cardholders.csv                       # Generated cardholder profiles
├── transactions.csv                      # Generated transaction data
├── ml_evaluation_test.csv                # ML predictions on test set
├── hybrid_evaluation_test.csv            # Hybrid system decisions
├── cost_comparison_results.csv           # Cost analysis
├── system_comparison_test_set.png        # Performance comparison chart
├── cost_comparison_analysis.png          # Cost breakdown chart
├── threshold_optimization_analysis.png   # Threshold impact
└── [fraud_type]_patterns.png            # Pattern visualizations (9 files)
```

### Key Results to Review

#### 1. Model Performance (from fraud_detection_models.py)

Open `system_comparison_test_set.png` to see:
- Precision and recall by system
- False positive rates
- Overall performance comparison

**What to Look For:**
- Hybrid should have highest F1-score (~94%)
- False positive rate should be <0.5%
- Rules-Only should have lower recall (~67%)

#### 2. Cost Analysis (from fraud_cost_analysis.py)

Open `cost_comparison_analysis.png` to see:
- Total cost by system
- Cost breakdown by component
- Savings calculations

**What to Look For:**
- Hybrid should have lowest total cost
- Significant savings vs Rules-Only (~$150K on test set)
- Moderate savings vs ML-Only (~$36K)

#### 3. Fraud Patterns (from fraud_pattern_visualization.py)

Open `fraud_summary_dashboard.png` for overview, then individual pattern files for details.

**What to Look For:**
- Distinctive patterns for each fraud type
- Card testing: Small amounts, high decline rates
- CNP fraud: Larger amounts, poor AVS/CVV matches
- Account takeover: High velocity, unusual merchants

---

## Jupyter Notebook Analysis (Optional)

### Option A: Data Generation Notebook

```bash
jupyter notebook synthetic-fraud-data-generator.ipynb
```

**Purpose:** Interactive exploration of data generation process
- Step-by-step generation with explanations
- Data quality validation
- Fraud type distribution analysis

### Option B: Hybrid Detection System Notebook

```bash
jupyter notebook hybrid-detection-system-w-typologies.ipynb
```

**Purpose:** Comprehensive analysis walkthrough
- Business context and methodology
- Feature engineering details
- Model training and evaluation
- Cost analysis and ROI

**Note:** Notebooks provide educational value but scripts are sufficient for results.

---

## Troubleshooting

### Common Issues

#### Issue 1: "ModuleNotFoundError: No module named 'pandas'"

**Solution:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

#### Issue 2: "FileNotFoundError: transactions.csv"

**Solution:** Run data generator first:
```bash
python generate_synthetic_data.py
```

#### Issue 3: "ValueError: time data doesn't match format"

**Cause:** Mixed timestamp formats in CSV
**Solution:** Already fixed in updated scripts (uses `format='mixed'`)

If still encountering issues, manually fix:
```python
# In fraud_detection_models.py, line ~653
transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], format='mixed')
```

#### Issue 4: "AttributeError: 'numpy.ndarray' object has no attribute 'apply'"

**Cause:** Using pandas `.apply()` on numpy array
**Solution:** See detailed fix in UPDATE_SUMMARY.md or replace line with:
```python
df['decision'] = ['BLOCK' if score >= threshold else 'APPROVE' for score in ml_scores]
```

#### Issue 5: Low Memory Warning

**Solution:** Increase available memory or reduce dataset size:
```python
# In generate_synthetic_data.py, line 27
generator = FraudDataGenerator(
    n_cardholders=250,  # Reduced from 500
    months=6,
    fraud_rate=0.015
)
```

#### Issue 6: Visualizations Don't Display

**Cause:** Running in headless environment or missing display
**Solution:** Visualizations are saved to PNG files automatically. View files directly:
```bash
# On Windows:
start card_testing_patterns.png

# On macOS:
open card_testing_patterns.png

# On Linux:
xdg-open card_testing_patterns.png
```

### Getting Help

If issues persist:
1. Check `README.md` for detailed documentation
2. Review `assumptions.md` for technical specifications
3. Examine `UPDATE_SUMMARY.md` for recent fixes
4. Verify Python version: `python --version` (requires 3.8+)

---

## Next Steps

### For Portfolio Presentation:

1. **Keep These Files:**
   - `system_comparison_test_set.png` - Key performance chart
   - `cost_comparison_analysis.png` - Business impact chart
   - `fraud_summary_dashboard.png` - Pattern overview
   - `README.md` - Project documentation

2. **Key Talking Points:**
   - 52% cost reduction vs rules-only system
   - 94%+ fraud detection rate with <0.5% false positives
   - Proper ML methodology (train/validation/test split)
   - Cost-based threshold optimization

3. **Demonstrate Understanding:**
   - "Why hybrid beats ML-only" (explainability, lower FP cost)
   - "Why threshold optimization matters" (saves $40K+)
   - "Why proper splits matter" (prevents overfitting)

### For Production Deployment:

Review recommendations in notebook Section 8 and README.md:
- Real-time inference requirements (100-500ms)
- Model monitoring and drift detection
- Retraining pipeline (quarterly)
- A/B testing framework
- Regulatory compliance (explainability)

---

## Quick Command Reference

```bash
# Full analysis pipeline (run in order)
python generate_synthetic_data.py           # 2-5 minutes
python fraud_detection_models.py            # 5-10 minutes
python fraud_cost_analysis.py               # 1-2 minutes
python fraud_pattern_visualization.py       # 3-5 minutes (optional)

# Review results
ls -lh *.csv *.png                          # List output files
head -5 transactions.csv                    # Preview data
python -c "import pandas as pd; print(pd.read_csv('cost_comparison_results.csv'))"

# Jupyter notebooks (optional)
jupyter notebook synthetic-fraud-data-generator.ipynb
jupyter notebook hybrid-detection-system-w-typologies.ipynb
```

---

**Total Runtime:** 30-45 minutes (including optional visualizations)

**Success Criteria:**
✅ All output files generated without errors  
✅ Hybrid system shows ~94% precision and recall  
✅ Cost reduction of ~50% vs rules-only  
✅ Fraud rate in data is ~1.5%  

---

**Last Updated:** January 2026  
**Project:** Credit Card Fraud Detection - Hybrid ML + Rules System
