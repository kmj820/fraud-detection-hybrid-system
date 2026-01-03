"""
Fraud Detection System: Rules + Machine Learning with Proper Methodology

This script implements a hybrid fraud detection system with:
- Rule-based detection (hard and soft rules)
- Machine learning (Random Forest)
- Proper train/validation/test split (60/20/20)
- Threshold optimization on validation set
- Fair comparison across all systems on test set

Fraud types covered:
- Card testing, Stolen card fraud, Account takeover, Friendly fraud
- Refund fraud, Synthetic ID, Application fraud, Lost/stolen card
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, precision_recall_curve,
                            average_precision_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionRules:
    """Rule-based fraud detection system"""
    
    def __init__(self):
        self.rules_triggered = []
        
    def check_impossible_travel(self, row):
        """Flag impossible travel: >500 miles in <1 hour"""
        if pd.notna(row['distance_from_last_txn']) and pd.notna(row['time_since_last_txn_hours']):
            if row['time_since_last_txn_hours'] > 0:
                speed = row['distance_from_last_txn'] / row['time_since_last_txn_hours']
                if speed > 500:
                    return True, 'impossible_travel'
        return False, None
    
    def check_card_testing(self, row, window_txns):
        """Flag card testing: multiple small transactions in short time"""
        if len(window_txns) >= 8:
            small_amount_count = (window_txns['amount'] <= 5).sum()
            declined_count = (~window_txns['authorized']).sum()
            
            if small_amount_count >= 5 and declined_count >= 3:
                return True, 'card_testing'
        return False, None
    
    def check_high_velocity(self, row):
        """Flag unusually high transaction velocity"""
        if row['txn_count_24h'] > 15:
            return True, 'high_velocity'
        if row['spend_24h'] > 5000:
            return True, 'high_spend_velocity'
        return False, None
    
    def check_cnp_risk_factors(self, row):
        """Flag CNP transactions with multiple risk factors"""
        risk_score = 0
        
        if row['amount'] > 500:
            risk_score += 1
        if not row['card_present']:
            risk_score += 1
        if row['avs_match'] in ['No Match', 'Not Available']:
            risk_score += 1
        if row['cvv_match'] in ['No Match', 'Not Provided']:
            risk_score += 1
        if row['mcc'] in [5732, 5944, 5999]:
            risk_score += 1
        hour = pd.to_datetime(row['timestamp']).hour
        if 1 <= hour <= 5:
            risk_score += 1
        
        if risk_score >= 3:
            return True, 'cnp_high_risk'
        return False, None
    
    def check_new_pattern(self, row, history_txns):
        """Flag transactions in new merchant categories"""
        if len(history_txns) > 0:
            historical_mccs = set(history_txns['mcc'].unique())
            if row['mcc'] not in historical_mccs and row['amount'] > 200:
                return True, 'new_merchant_pattern'
        return False, None
    
    def apply_rules(self, transactions):
        """Apply all rules to transaction dataset"""
        results = []
        
        transactions = transactions.sort_values(['cardholder_id', 'timestamp'])
        
        for idx, row in transactions.iterrows():
            cardholder_id = row['cardholder_id']
            timestamp = row['timestamp']
            
            ch_txns = transactions[transactions['cardholder_id'] == cardholder_id]
            history_txns = ch_txns[ch_txns['timestamp'] < timestamp]
            
            window_start = timestamp - pd.Timedelta(hours=24)
            window_txns = history_txns[history_txns['timestamp'] >= window_start]
            
            triggered_rules = []
            
            flag, rule = self.check_impossible_travel(row)
            if flag:
                triggered_rules.append(rule)
            
            flag, rule = self.check_card_testing(row, window_txns)
            if flag:
                triggered_rules.append(rule)
            
            flag, rule = self.check_high_velocity(row)
            if flag:
                triggered_rules.append(rule)
            
            flag, rule = self.check_cnp_risk_factors(row)
            if flag:
                triggered_rules.append(rule)
            
            flag, rule = self.check_new_pattern(row, history_txns)
            if flag:
                triggered_rules.append(rule)
            
            results.append({
                'transaction_id': row['transaction_id'],
                'rules_triggered': triggered_rules,
                'has_hard_rule': ('impossible_travel' in triggered_rules or 
                                 'card_testing' in triggered_rules),
                'has_soft_rule': any(r in triggered_rules for r in 
                                    ['high_velocity', 'high_spend_velocity', 
                                     'cnp_high_risk', 'new_merchant_pattern'])
            })
        
        return pd.DataFrame(results)

class MLFraudDetector:
    """Machine Learning fraud detection system"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.feature_importance = None
        
    def prepare_features(self, transactions, fit=True):
        """Prepare features for ML model"""
        df = transactions.copy()
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_late_night'] = df['hour'].between(1, 5).astype(int)
        
        categorical_cols = ['avs_match', 'cvv_match']
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        feature_cols = [
            'amount', 'mcc', 'card_present', 'authorized',
            'avs_match_encoded', 'cvv_match_encoded',
            'distance_from_last_txn', 'time_since_last_txn_hours',
            'txn_count_24h', 'txn_count_7d', 'spend_24h', 'spend_7d',
            'hour', 'day_of_week', 'is_weekend', 'is_late_night',
            'current_balance', 'available_credit',
        ]
        
        X = df[feature_cols].fillna(0)
        X['card_present'] = X['card_present'].astype(int)
        X['authorized'] = X['authorized'].astype(int)
        
        if fit:
            self.feature_names = feature_cols
        
        return X
    
    def train(self, train_transactions, val_transactions):
        """Train ML model with separate validation set"""
        print("Preparing features...")
        X_train = self.prepare_features(train_transactions, fit=True)
        y_train = train_transactions['is_fraud'].astype(int)
        
        X_val = self.prepare_features(val_transactions, fit=False)
        y_val = val_transactions['is_fraud'].astype(int)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        print(f"Training set: {len(X_train)} transactions ({y_train.mean():.2%} fraud)")
        print(f"Validation set: {len(X_val)} transactions ({y_val.mean():.2%} fraud)")
        
        print("\nTraining Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("✓ Model training complete")
        
        y_val_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        return X_val_scaled, y_val, y_val_pred_proba
    
    def predict(self, transactions):
        """Predict fraud probability for new transactions"""
        X = self.prepare_features(transactions, fit=False)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        return predictions

class ThresholdOptimizer:
    """Optimize decision thresholds based on validation data"""
    
    def __init__(self):
        self.thresholds = {
            'block': None,
            'review_high': None,
            'review_medium': None
        }
        
        # Cost parameters (issuer perspective with interchange revenue)
        self.costs = {
            # Fraud loss rates (issuer liability)
            'cnp_fraud_loss_rate': 0.85,
            'cp_fraud_loss_rate': 0.15,
            'friendly_fraud_loss_rate': 0.5,
            
            # Fixed costs
            'chargeback_fee': 25,
            'investigation_cost': 50,
            'false_positive_loss': 0.10,
            'false_positive_churn_rate': 0.02,
            'customer_lifetime_value': 2000,
            'review_cost': 15,
            
            # Revenue
            'interchange_rate': 0.02,  # 2% of transaction
        }
    
    def calculate_cost(self, y_true, ml_scores, rules_df, block_thresh, review_high_thresh, review_med_thresh, amounts):
        """Calculate total cost for given thresholds"""
        total_cost = 0
        
        for i in range(len(y_true)):
            is_fraud = y_true.iloc[i]
            ml_score = ml_scores[i]
            has_hard_rule = rules_df.iloc[i]['has_hard_rule']
            has_soft_rule = rules_df.iloc[i]['has_soft_rule']
            amount = amounts.iloc[i]
            
            # Determine decision
            if has_hard_rule or ml_score >= block_thresh:
                decision = 'BLOCK'
            elif ml_score >= review_high_thresh or (ml_score >= review_med_thresh and has_soft_rule):
                decision = 'REVIEW'
            else:
                decision = 'APPROVE'
            
            # Calculate cost
            if is_fraud:
                if decision == 'APPROVE':
                    # False negative - but earned interchange
                    cost = amount * 0.85 + self.costs['chargeback_fee'] + self.costs['investigation_cost']
                    cost -= amount * self.costs['interchange_rate']  # Subtract revenue earned
                    total_cost += cost
                elif decision == 'REVIEW':
                    # Review cost, 10% slip through
                    total_cost += self.costs['review_cost']
                    if np.random.random() < 0.10:
                        cost = amount * 0.85 + self.costs['chargeback_fee']
                        cost -= amount * self.costs['interchange_rate']
                        total_cost += cost
                # BLOCK: fraud prevented, lost interchange revenue
                else:
                    total_cost += amount * self.costs['interchange_rate']  # Lost revenue
            else:  # Legitimate
                if decision == 'APPROVE':
                    # Revenue earned (negative cost)
                    total_cost -= amount * self.costs['interchange_rate']
                elif decision == 'BLOCK':
                    # Lost interchange + customer friction + potential churn
                    total_cost += amount * self.costs['false_positive_loss']
                    if np.random.random() < self.costs['false_positive_churn_rate']:
                        total_cost += self.costs['customer_lifetime_value']
                    total_cost += amount * self.costs['interchange_rate']  # Lost revenue
                else:  # REVIEW
                    total_cost += self.costs['review_cost']
                    total_cost += amount * self.costs['false_positive_loss'] * 0.5
                    # Still earn interchange after review approves
                    total_cost -= amount * self.costs['interchange_rate'] * 0.9  # 90% approved after review
        
        return total_cost
    
    def optimize_thresholds(self, y_val, ml_scores_val, rules_val, amounts_val, 
                           max_review_rate=0.02):
        """Find optimal thresholds on validation data"""
        print("\n" + "="*60)
        print("OPTIMIZING THRESHOLDS ON VALIDATION DATA")
        print("="*60)
        print(f"Maximum review rate constraint: {max_review_rate*100:.1f}%")
        
        best_cost = float('inf')
        best_thresholds = None
        
        block_candidates = np.linspace(0.85, 0.99, 8)
        review_high_candidates = np.linspace(0.60, 0.85, 8)
        review_med_candidates = np.linspace(0.40, 0.65, 8)
        
        print(f"\nSearching {len(block_candidates) * len(review_high_candidates) * len(review_med_candidates)} combinations...")
        
        results = []
        
        for block_t in block_candidates:
            for review_high_t in review_high_candidates:
                for review_med_t in review_med_candidates:
                    if review_high_t >= block_t or review_med_t >= review_high_t:
                        continue
                    
                    # Calculate review rate
                    n_reviews = 0
                    for i in range(len(y_val)):
                        ml_score = ml_scores_val[i]
                        has_hard_rule = rules_val.iloc[i]['has_hard_rule']
                        has_soft_rule = rules_val.iloc[i]['has_soft_rule']
                        
                        if has_hard_rule or ml_score >= block_t:
                            pass
                        elif ml_score >= review_high_t or (ml_score >= review_med_t and has_soft_rule):
                            n_reviews += 1
                    
                    review_rate = n_reviews / len(y_val)
                    
                    if review_rate > max_review_rate:
                        continue
                    
                    cost = self.calculate_cost(y_val, ml_scores_val, rules_val, 
                                              block_t, review_high_t, review_med_t, amounts_val)
                    
                    results.append({
                        'block_thresh': block_t,
                        'review_high_thresh': review_high_t,
                        'review_med_thresh': review_med_t,
                        'cost': cost,
                        'review_rate': review_rate
                    })
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_thresholds = {
                            'block': block_t,
                            'review_high': review_high_t,
                            'review_medium': review_med_t
                        }
        
        self.thresholds = best_thresholds
        
        print(f"\n✓ Optimization complete!")
        print(f"Optimal thresholds:")
        print(f"  Block threshold: {self.thresholds['block']:.3f}")
        print(f"  Review (high) threshold: {self.thresholds['review_high']:.3f}")
        print(f"  Review (medium) threshold: {self.thresholds['review_medium']:.3f}")
        print(f"  Estimated cost on validation: ${best_cost:,.2f}")
        
        return pd.DataFrame(results)

class HybridFraudDetectionSystem:
    """Combined rule-based and ML detection system with optimized thresholds"""
    
    def __init__(self):
        self.rules_engine = FraudDetectionRules()
        self.ml_detector = MLFraudDetector()
        self.optimizer = ThresholdOptimizer()
        self.thresholds = None
        
    def train_and_optimize(self, train_transactions, val_transactions):
        """Train ML model and optimize thresholds"""
        print("="*60)
        print("TRAINING HYBRID FRAUD DETECTION SYSTEM")
        print("="*60)
        
        X_val, y_val, y_val_pred_proba = self.ml_detector.train(train_transactions, val_transactions)
        
        print("\nApplying rules to validation set...")
        rules_val = self.rules_engine.apply_rules(val_transactions)
        
        amounts_val = val_transactions.reset_index(drop=True)['amount']
        y_val_reset = val_transactions.reset_index(drop=True)['is_fraud'].astype(int)
        
        optimization_results = self.optimizer.optimize_thresholds(
            y_val_reset, y_val_pred_proba, rules_val, amounts_val
        )
        
        self.thresholds = self.optimizer.thresholds
        
        return optimization_results
    
    def make_decision(self, ml_score, has_hard_rule, has_soft_rule):
        """Make decision using optimized thresholds"""
        if has_hard_rule or ml_score >= self.thresholds['block']:
            return 'BLOCK'
        elif ml_score >= self.thresholds['review_high']:
            return 'REVIEW'
        elif ml_score >= self.thresholds['review_medium'] and has_soft_rule:
            return 'REVIEW'
        else:
            return 'APPROVE'
    
    def evaluate(self, test_transactions):
        """Evaluate system on test set"""
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET")
        print("="*60)
        
        ml_scores = self.ml_detector.predict(test_transactions)
        rules_results = self.rules_engine.apply_rules(test_transactions)
        
        decisions = []
        for i in range(len(test_transactions)):
            decision = self.make_decision(
                ml_scores[i],
                rules_results.iloc[i]['has_hard_rule'],
                rules_results.iloc[i]['has_soft_rule']
            )
            decisions.append(decision)
        
        evaluation = test_transactions.copy().reset_index(drop=True)
        evaluation['ml_score'] = ml_scores
        evaluation['has_hard_rule'] = rules_results['has_hard_rule'].values
        evaluation['has_soft_rule'] = rules_results['has_soft_rule'].values
        evaluation['rules_triggered'] = rules_results['rules_triggered'].values
        evaluation['decision'] = decisions
        
        return evaluation

def create_rules_only_system(test_transactions):
    """Evaluate rules-only system on test set"""
    df = test_transactions.copy().reset_index(drop=True)
    df['decision'] = 'APPROVE'
    
    rules_engine = FraudDetectionRules()
    rules_results = rules_engine.apply_rules(test_transactions)
    
    df['decision'] = rules_results['has_hard_rule'].apply(
        lambda x: 'BLOCK' if x else 'APPROVE'
    )
    
    return df

def create_ml_only_system(test_transactions, ml_scores, threshold):
    """Evaluate ML-only system on test set"""
    df = test_transactions.copy().reset_index(drop=True)
    df['ml_score'] = ml_scores
    # Fix: ml_scores is numpy array, not pandas Series - use list comprehension
    df['decision'] = ['BLOCK' if score >= threshold else 'APPROVE' for score in ml_scores]
    return df

def optimize_ml_threshold(y_val, ml_scores_val, amounts_val, costs):
    """Find optimal ML threshold on validation set"""
    best_threshold = 0.5
    best_cost = float('inf')
    
    thresholds = np.linspace(0.3, 0.9, 30)
    
    for thresh in thresholds:
        total_cost = 0
        for i in range(len(y_val)):
            is_fraud = y_val.iloc[i]
            ml_score = ml_scores_val[i]
            amount = amounts_val.iloc[i]
            
            decision = 'BLOCK' if ml_score >= thresh else 'APPROVE'
            
            if is_fraud:
                if decision == 'APPROVE':
                    cost = amount * 0.85 + costs['chargeback_fee'] + costs['investigation_cost']
                    cost -= amount * costs['interchange_rate']
                    total_cost += cost
                else:  # BLOCK
                    total_cost += amount * costs['interchange_rate']  # Lost revenue
            else:
                if decision == 'APPROVE':
                    total_cost -= amount * costs['interchange_rate']  # Revenue
                else:  # BLOCK
                    total_cost += amount * costs['false_positive_loss']
                    if np.random.random() < costs['false_positive_churn_rate']:
                        total_cost += costs['customer_lifetime_value']
                    total_cost += amount * costs['interchange_rate']  # Lost revenue
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = thresh
    
    return best_threshold

def calculate_metrics(evaluation_df):
    """Calculate performance metrics"""
    y_true = evaluation_df['is_fraud']
    blocked = (evaluation_df['decision'] == 'BLOCK')
    reviewed = (evaluation_df['decision'] == 'REVIEW')
    
    tp = (y_true & (blocked | reviewed)).sum()
    fp = (~y_true & (blocked | reviewed)).sum()
    tn = (~y_true & ~(blocked | reviewed)).sum()
    fn = (y_true & ~(blocked | reviewed)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr
    }

def visualize_comparison(rules_metrics, ml_metrics, hybrid_metrics):
    """Visualize system comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detection System Comparison (Test Set)', fontsize=16, fontweight='bold')
    
    systems = ['Rules-Only', 'ML-Only', 'Hybrid']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    
    # 1. Precision, Recall, F1
    ax1 = axes[0, 0]
    metrics_to_plot = ['precision', 'recall', 'f1']
    x = np.arange(len(systems))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        values = [rules_metrics[metric], ml_metrics[metric], hybrid_metrics[metric]]
        ax1.bar(x + i*width - width, values, width, label=metric.capitalize(), alpha=0.7)
    
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Precision, Recall, F1-Score', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. False Positive Rate
    ax2 = axes[0, 1]
    fpr_values = [rules_metrics['fpr']*100, ml_metrics['fpr']*100, hybrid_metrics['fpr']*100]
    bars = ax2.bar(systems, fpr_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('False Positive Rate (%)', fontsize=11)
    ax2.set_title('False Positive Rate (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, fpr_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Confusion matrices
    ax3 = axes[1, 0]
    all_metrics = [rules_metrics, ml_metrics, hybrid_metrics]
    cm_combined = np.column_stack([
        np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]]) for m in all_metrics
    ])
    
    im = ax3.imshow(cm_combined, cmap='Blues', aspect='auto')
    ax3.set_xticks([0.5, 1, 2.5, 3, 4.5, 5])
    ax3.set_xticklabels(['TN', 'FP', 'TN', 'FP', 'TN', 'FP'], fontsize=9)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Legit', 'Fraud'], fontsize=10)
    
    for i in range(2):
        for j in range(6):
            text = ax3.text(j, i, cm_combined[i, j],
                          ha="center", va="center", color="black", fontsize=9)
    
    ax3.text(0.5, -0.5, 'Rules', ha='center', fontsize=10, fontweight='bold')
    ax3.text(2.5, -0.5, 'ML', ha='center', fontsize=10, fontweight='bold')
    ax3.text(4.5, -0.5, 'Hybrid', ha='center', fontsize=10, fontweight='bold')
    ax3.set_title('Confusion Matrices', fontsize=12, fontweight='bold')
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    TEST SET PERFORMANCE SUMMARY
    {'='*40}
    
    {"Metric":<20} {"Rules":<8} {"ML":<8} {"Hybrid":<8}
    {'-'*40}
    Precision:
      {rules_metrics['precision']:>6.3f}   {ml_metrics['precision']:>6.3f}   {hybrid_metrics['precision']:>6.3f}
    
    Recall:
      {rules_metrics['recall']:>6.3f}   {ml_metrics['recall']:>6.3f}   {hybrid_metrics['recall']:>6.3f}
    
    F1-Score:
      {rules_metrics['f1']:>6.3f}   {ml_metrics['f1']:>6.3f}   {hybrid_metrics['f1']:>6.3f}
    
    False Positive Rate:
      {rules_metrics['fpr']*100:>5.2f}%   {ml_metrics['fpr']*100:>5.2f}%   {hybrid_metrics['fpr']*100:>5.2f}%
    
    True Positives:
      {rules_metrics['tp']:>6}   {ml_metrics['tp']:>6}   {hybrid_metrics['tp']:>6}
    
    False Positives:
      {rules_metrics['fp']:>6}   {ml_metrics['fp']:>6}   {hybrid_metrics['fp']:>6}
    
    False Negatives:
      {rules_metrics['fn']:>6}   {ml_metrics['fn']:>6}   {hybrid_metrics['fn']:>6}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('system_comparison_test_set.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved: system_comparison_test_set.png")
    plt.show()

def main():
    """Main execution function"""
    print("="*60)
    print("FRAUD DETECTION: PROPER TRAIN/VAL/TEST METHODOLOGY")
    print("="*60)
    
    # Load data
    print("\nLoading transaction data...")
    try:
        transactions = pd.read_csv('transactions.csv')
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], format='mixed')
        print(f"✓ Loaded {len(transactions):,} transactions")
        print(f"  Fraud rate: {transactions['is_fraud'].mean():.2%}")
    except FileNotFoundError:
        print("Error: transactions.csv not found. Please run the data generator first.")
        return
    
    # Split into train/val/test (60/20/20)
    print("\n" + "="*60)
    print("SPLITTING DATA: TRAIN/VAL/TEST")
    print("="*60)
        
    train_val, test = train_test_split(
        transactions, test_size=0.20, random_state=42, 
        stratify=transactions['is_fraud']
    )
    
    train, val = train_test_split(
        train_val, test_size=0.25, random_state=42,
        stratify=train_val['is_fraud']
    )
    
    print(f"Train set: {len(train):,} ({train['is_fraud'].mean():.2%} fraud)")
    print(f"Validation set: {len(val):,} ({val['is_fraud'].mean():.2%} fraud)")
    print(f"Test set: {len(test):,} ({test['is_fraud'].mean():.2%} fraud)")
    
    # Initialize and train hybrid system
    hybrid_system = HybridFraudDetectionSystem()
    optimization_results = hybrid_system.train_and_optimize(train, val)
    
    # Print feature importance
    print("\nTop 10 Most Important Features:")
    print(hybrid_system.ml_detector.feature_importance.head(10).to_string(index=False))
    
    # Optimize ML-only threshold on validation set
    print("\n" + "="*60)
    print("OPTIMIZING ML-ONLY THRESHOLD")
    print("="*60)
    
    val_ml_scores = hybrid_system.ml_detector.predict(val)
    ml_optimal_threshold = optimize_ml_threshold(
        val.reset_index(drop=True)['is_fraud'].astype(int),
        val_ml_scores,
        val.reset_index(drop=True)['amount'],
        hybrid_system.optimizer.costs
    )
    print(f"✓ Optimal ML threshold: {ml_optimal_threshold:.3f}")
    
    # EVALUATE ALL SYSTEMS ON TEST SET ONLY
    print("\n" + "="*60)
    print("EVALUATING ALL SYSTEMS ON TEST SET")
    print("="*60)
    
    # 1. Hybrid system
    print("\n1. Hybrid System (Rules + ML with optimized thresholds)...")
    hybrid_eval = hybrid_system.evaluate(test)
    hybrid_metrics = calculate_metrics(hybrid_eval)
    print(f"   Precision: {hybrid_metrics['precision']:.3f}")
    print(f"   Recall: {hybrid_metrics['recall']:.3f}")
    print(f"   F1-Score: {hybrid_metrics['f1']:.3f}")
    print(f"   FPR: {hybrid_metrics['fpr']*100:.2f}%")
    
    # 2. Rules-only system
    print("\n2. Rules-Only System...")
    rules_eval = create_rules_only_system(test)
    rules_metrics = calculate_metrics(rules_eval)
    print(f"   Precision: {rules_metrics['precision']:.3f}")
    print(f"   Recall: {rules_metrics['recall']:.3f}")
    print(f"   F1-Score: {rules_metrics['f1']:.3f}")
    print(f"   FPR: {rules_metrics['fpr']*100:.2f}%")
    
    # 3. ML-only system (with optimized threshold)
    print("\n3. ML-Only System (optimized threshold)...")
    test_ml_scores = hybrid_system.ml_detector.predict(test)
    ml_eval = create_ml_only_system(test, test_ml_scores, ml_optimal_threshold)
    ml_metrics = calculate_metrics(ml_eval)
    print(f"   Precision: {ml_metrics['precision']:.3f}")
    print(f"   Recall: {ml_metrics['recall']:.3f}")
    print(f"   F1-Score: {ml_metrics['f1']:.3f}")
    print(f"   FPR: {ml_metrics['fpr']*100:.2f}%")
    
    # Save results
    hybrid_eval.to_csv('hybrid_evaluation_test.csv', index=False)
    rules_eval.to_csv('rules_evaluation_test.csv', index=False)
    ml_eval.to_csv('ml_evaluation_test.csv', index=False)
    
    print("\n✓ Evaluation results saved:")
    print("  • hybrid_evaluation_test.csv")
    print("  • rules_evaluation_test.csv")
    print("  • ml_evaluation_test.csv")
    
    # Visualize comparison
    print("\nGenerating comparison visualization...")
    visualize_comparison(rules_metrics, ml_metrics, hybrid_metrics)
    
    # Print optimal thresholds
    print("\n" + "="*60)
    print("OPTIMIZED THRESHOLDS (from validation set)")
    print("="*60)
    print(f"Hybrid System:")
    print(f"  Block threshold: {hybrid_system.thresholds['block']:.3f}")
    print(f"  Review (high) threshold: {hybrid_system.thresholds['review_high']:.3f}")
    print(f"  Review (medium) threshold: {hybrid_system.thresholds['review_medium']:.3f}")
    print(f"\nML-Only:")
    print(f"  Block threshold: {ml_optimal_threshold:.3f}")
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nAnalysis Summary:")
    print("✓ All thresholds optimized on validation set")
    print("✓ All systems evaluated on same test set")
    print("✓ Fair comparison with proper methodology")
    print("\nGenerated files:")
    print("  • system_comparison_test_set.png")
    print("  • hybrid_evaluation_test.csv")
    print("  • rules_evaluation_test.csv")
    print("  • ml_evaluation_test.csv")

if __name__ == "__main__":
    main()