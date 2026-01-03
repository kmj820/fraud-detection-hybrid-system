import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class FraudCostCalculator:
    """Calculate financial costs of fraud detection from card issuer perspective"""
    
    def __init__(self):
        # Cost parameters (industry averages from issuer perspective)
        self.costs = {
            # Fraud loss percentages (what issuer actually pays)
            'cnp_fraud_loss_rate': 0.85,     # Issuer pays 85% (only 15% of merchants use 3DS)
            'cp_fraud_loss_rate': 0.15,      # Issuer pays 15% (edge cases despite EMV shift)
            'friendly_fraud_loss_rate': 0.5,  # Split 50/50 with merchant
            
            # Fixed costs per fraud incident
            'chargeback_fee': 25,            # Card network fee
            'investigation_cost': 50,        # Internal fraud analyst time
            
            # False positive costs
            'false_positive_loss': 0.10,     # 10% of transaction (customer inconvenience)
            'false_positive_churn_rate': 0.02, # 2% of customers churn after false decline
            'customer_lifetime_value': 2000,  # Average CLV
            
            # Operational costs
            'review_cost': 15,               # Manual review by analyst
        }
    
    def calculate_issuer_fraud_cost(self, row):
        """
        Calculate fraud cost to issuer based on fraud type and merchant EMV compliance
        
        Returns: Cost to card issuer only (not total ecosystem cost)
        """
        amount = row['amount']
        fraud_type = row.get('fraud_type', 'unknown')
        card_present = row.get('card_present', False)
        merchant_emv_compliant = row.get('merchant_emv_compliant', True)
        
        cost = 0
        
        # Fraud loss (varies by type and liability rules)
        if fraud_type == 'cnp_fraud' or fraud_type == 'card_testing':
            # CNP: Issuer always pays
            cost += amount * self.costs['cnp_fraud_loss_rate']
            cost += self.costs['chargeback_fee']
            cost += self.costs['investigation_cost']
        
        elif fraud_type == 'account_takeover':
            # ATO can be mix of CNP and card-present
            if not card_present:
                # CNP portion
                cost += amount * self.costs['cnp_fraud_loss_rate']
            else:
                # Card-present portion
                if merchant_emv_compliant:
                    # Merchant used EMV, issuer liable
                    cost += amount * self.costs['cp_fraud_loss_rate']
                else:
                    # Merchant didn't use EMV, merchant liable (liability shift)
                    cost += 0  # Issuer doesn't pay
            
            cost += self.costs['chargeback_fee']
            cost += self.costs['investigation_cost']
        
        elif fraud_type == 'friendly_fraud':
            # Split between issuer and merchant based on dispute outcome
            cost += amount * self.costs['friendly_fraud_loss_rate']
            cost += self.costs['chargeback_fee']
            cost += self.costs['investigation_cost']
        
        elif fraud_type == 'refund_fraud':
            # Similar to friendly fraud
            cost += amount * self.costs['friendly_fraud_loss_rate']
            cost += self.costs['investigation_cost']
        
        elif fraud_type == 'synthetic_id' or fraud_type == 'bust_out':
            # Issuer eats full loss (no merchant involved)
            cost += amount * 1.0
            cost += self.costs['investigation_cost']
        
        else:
            # Default: assume issuer pays (conservative estimate)
            cost += amount * 0.5
            cost += self.costs['chargeback_fee']
            cost += self.costs['investigation_cost']
        
        return cost
    
    def calculate_transaction_costs(self, evaluation_df):
        """Calculate costs for each transaction based on decision and outcome"""
        df = evaluation_df.copy()
        df['cost'] = 0.0
        
        for idx, row in df.iterrows():
            is_fraud = row['is_fraud']
            decision = row['decision']
            
            cost = 0
            
            if is_fraud:
                if decision == 'APPROVE':
                    # False Negative: Fraud approved (missed)
                    cost += self.calculate_issuer_fraud_cost(row)
                
                elif decision == 'REVIEW':
                    # True Positive (caught in review)
                    cost += self.costs['review_cost']
                    # Assume 90% of fraud caught in review, 10% slip through
                    if np.random.random() < 0.10:
                        cost += self.calculate_issuer_fraud_cost(row)
                
                elif decision == 'BLOCK':
                    # True Positive (blocked) - fraud prevented!
                    cost += 0
            
            else:  # Legitimate transaction
                if decision == 'APPROVE':
                    # True Negative: Correctly approved
                    cost += 0
                
                elif decision == 'REVIEW':
                    # False Positive (review): Some customer friction
                    cost += self.costs['review_cost']
                    cost += row['amount'] * self.costs['false_positive_loss'] * 0.5
                
                elif decision == 'BLOCK':
                    # False Positive (blocked): Worst customer experience
                    cost += row['amount'] * self.costs['false_positive_loss']
                    # Some customers churn after being declined
                    if np.random.random() < self.costs['false_positive_churn_rate']:
                        cost += self.costs['customer_lifetime_value']
            
            df.at[idx, 'cost'] = cost
        
        return df
    
    def calculate_system_costs(self, evaluation_df):
        """Calculate aggregate costs for the detection system"""
        df = self.calculate_transaction_costs(evaluation_df)
        
        # Confusion matrix
        tp = ((df['is_fraud'] == True) & (df['decision'].isin(['BLOCK', 'REVIEW']))).sum()
        fp = ((df['is_fraud'] == False) & (df['decision'].isin(['BLOCK', 'REVIEW']))).sum()
        tn = ((df['is_fraud'] == False) & (df['decision'] == 'APPROVE')).sum()
        fn = ((df['is_fraud'] == True) & (df['decision'] == 'APPROVE')).sum()
        
        # Cost breakdown
        fraud_caught_cost = df[df['is_fraud'] & df['decision'].isin(['BLOCK', 'REVIEW'])]['cost'].sum()
        fraud_missed_cost = df[df['is_fraud'] & (df['decision'] == 'APPROVE')]['cost'].sum()
        false_positive_cost = df[~df['is_fraud'] & df['decision'].isin(['BLOCK', 'REVIEW'])]['cost'].sum()
        
        total_cost = df['cost'].sum()
        
        # Fraud amounts
        total_fraud_amount = df[df['is_fraud']]['amount'].sum()
        fraud_prevented = df[df['is_fraud'] & df['decision'].isin(['BLOCK', 'REVIEW'])]['amount'].sum()
        fraud_losses = df[df['is_fraud'] & (df['decision'] == 'APPROVE')]['amount'].sum()
        
        # Cost by fraud type (for missed fraud)
        fraud_cost_by_type = {}
        for fraud_type in df[df['is_fraud']]['fraud_type'].dropna().unique():
            fraud_type_missed = df[(df['is_fraud']) & 
                                  (df['fraud_type'] == fraud_type) & 
                                  (df['decision'] == 'APPROVE')]
            fraud_cost_by_type[fraud_type] = fraud_type_missed['cost'].sum()
        
        results = {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'fraud_caught_cost': fraud_caught_cost,
            'fraud_missed_cost': fraud_missed_cost,
            'false_positive_cost': false_positive_cost,
            'total_cost': total_cost,
            'total_fraud_amount': total_fraud_amount,
            'fraud_prevented': fraud_prevented,
            'fraud_losses': fraud_losses,
            'fraud_prevention_rate': fraud_prevented / total_fraud_amount if total_fraud_amount > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'fraud_cost_by_type': fraud_cost_by_type,
        }
        
        return results

def create_rules_only_system(transactions):
    """Simulate a rules-only detection system"""
    df = transactions.copy()
    df['decision'] = 'APPROVE'
    df['ml_score'] = 0.0
    
    for idx, row in df.iterrows():
        block = False
        
        # Rule 1: Impossible travel
        if pd.notna(row['distance_from_last_txn']) and pd.notna(row['time_since_last_txn_hours']):
            if row['time_since_last_txn_hours'] > 0:
                speed = row['distance_from_last_txn'] / row['time_since_last_txn_hours']
                if speed > 500:
                    block = True
        
        # Rule 2: High velocity
        if row['txn_count_24h'] > 15 or row['spend_24h'] > 5000:
            block = True
        
        # Rule 3: CNP risk factors
        risk_score = 0
        if row['amount'] > 500: risk_score += 1
        if not row['card_present']: risk_score += 1
        if row['avs_match'] in ['No Match', 'Not Available']: risk_score += 1
        if row['cvv_match'] in ['No Match', 'Not Provided']: risk_score += 1
        if row['mcc'] in [5732, 5944, 5999]: risk_score += 1
        hour = pd.to_datetime(row['timestamp']).hour
        if 1 <= hour <= 5: risk_score += 1
        if risk_score >= 4:
            block = True
        
        if block:
            df.at[idx, 'decision'] = 'BLOCK'
    
    return df

def create_ml_only_system(transactions, ml_scores, threshold=0.5):
    """Simulate an ML-only detection system"""
    df = transactions.copy()
    df['ml_score'] = ml_scores
    df['decision'] = df['ml_score'].apply(lambda x: 'BLOCK' if x >= threshold else 'APPROVE')
    return df

def find_optimal_ml_threshold(transactions, ml_scores, cost_calculator):
    """Find optimal ML threshold that minimizes total cost"""
    thresholds = np.linspace(0.1, 0.99, 50)
    costs = []
    detection_rates = []
    fp_rates = []
    
    for threshold in thresholds:
        eval_df = create_ml_only_system(transactions, ml_scores, threshold)
        results = cost_calculator.calculate_system_costs(eval_df)
        
        costs.append(results['total_cost'])
        detection_rates.append(results['detection_rate'])
        fp_rates.append(results['false_positive_rate'])
    
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = costs[optimal_idx]
    
    return optimal_threshold, optimal_cost, thresholds, costs, detection_rates, fp_rates

def visualize_cost_comparison(rules_results, ml_results, hybrid_results, system_names):
    """Create comprehensive cost comparison visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fraud Detection Cost Analysis: System Comparison (Issuer Perspective)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Total Cost Comparison
    ax1 = axes[0, 0]
    systems = system_names
    total_costs = [rules_results['total_cost'], ml_results['total_cost'], hybrid_results['total_cost']]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    
    bars = ax1.bar(systems, total_costs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Total Cost to Issuer ($)', fontsize=11)
    ax1.set_title('Total System Cost Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, cost in zip(bars, total_costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Cost Breakdown
    ax2 = axes[0, 1]
    cost_categories = ['Fraud\nMissed', 'Fraud\nCaught', 'False\nPositives']
    
    rules_breakdown = [rules_results['fraud_missed_cost'], 
                      rules_results['fraud_caught_cost'],
                      rules_results['false_positive_cost']]
    ml_breakdown = [ml_results['fraud_missed_cost'], 
                   ml_results['fraud_caught_cost'],
                   ml_results['false_positive_cost']]
    hybrid_breakdown = [hybrid_results['fraud_missed_cost'], 
                       hybrid_results['fraud_caught_cost'],
                       hybrid_results['false_positive_cost']]
    
    x = np.arange(len(cost_categories))
    width = 0.25
    
    ax2.bar(x - width, rules_breakdown, width, label='Rules-Only', color=colors[0], alpha=0.7)
    ax2.bar(x, ml_breakdown, width, label='ML-Only', color=colors[1], alpha=0.7)
    ax2.bar(x + width, hybrid_breakdown, width, label='Hybrid', color=colors[2], alpha=0.7)
    
    ax2.set_ylabel('Cost to Issuer ($)', fontsize=11)
    ax2.set_title('Cost Breakdown by Category', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cost_categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Fraud Prevention
    ax3 = axes[0, 2]
    prevention_rates = [rules_results['fraud_prevention_rate'] * 100,
                       ml_results['fraud_prevention_rate'] * 100,
                       hybrid_results['fraud_prevention_rate'] * 100]
    
    bars = ax3.bar(systems, prevention_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Fraud Prevention Rate (%)', fontsize=11)
    ax3.set_title('Fraud Prevention Rate', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, prevention_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Detection vs False Positive Trade-off
    ax4 = axes[1, 0]
    detection_rates = [rules_results['detection_rate'] * 100,
                      ml_results['detection_rate'] * 100,
                      hybrid_results['detection_rate'] * 100]
    fp_rates = [rules_results['false_positive_rate'] * 100,
               ml_results['false_positive_rate'] * 100,
               hybrid_results['false_positive_rate'] * 100]
    
    ax4.scatter(fp_rates, detection_rates, s=500, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
    
    for i, system in enumerate(systems):
        ax4.annotate(system, (fp_rates[i], detection_rates[i]), 
                    fontsize=10, fontweight='bold', ha='center', va='center')
    
    ax4.set_xlabel('False Positive Rate (%)', fontsize=11)
    ax4.set_ylabel('Fraud Detection Rate (%)', fontsize=11)
    ax4.set_title('Detection vs False Positive Trade-off', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.1, max(fp_rates) * 1.2)
    ax4.set_ylim(min(detection_rates) * 0.9, 105)
    
    ax4.plot([0], [100], 'g*', markersize=20, label='Ideal (100% detection, 0% FP)')
    ax4.legend(loc='lower right')
    
    # 5. Confusion Matrix Comparison
    ax5 = axes[1, 1]
    
    confusion_data = []
    for results, name in zip([rules_results, ml_results, hybrid_results], systems):
        cm = np.array([[results['tn'], results['fp']], 
                      [results['fn'], results['tp']]])
        confusion_data.append(cm)
    
    combined_cm = np.column_stack(confusion_data)
    
    im = ax5.imshow(combined_cm, cmap='Blues', aspect='auto')
    
    ax5.set_xticks([0.5, 1, 2.5, 3, 4.5, 5])
    ax5.set_xticklabels(['TN', 'FP', 'TN', 'FP', 'TN', 'FP'], fontsize=9)
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Legit', 'Fraud'], fontsize=10)
    
    for i in range(2):
        for j in range(6):
            text = ax5.text(j, i, combined_cm[i, j],
                          ha="center", va="center", color="black", fontsize=9)
    
    ax5.text(0.5, -0.5, 'Rules-Only', ha='center', fontsize=10, fontweight='bold')
    ax5.text(2.5, -0.5, 'ML-Only', ha='center', fontsize=10, fontweight='bold')
    ax5.text(4.5, -0.5, 'Hybrid', ha='center', fontsize=10, fontweight='bold')
    ax5.set_title('Confusion Matrices Comparison', fontsize=12, fontweight='bold')
    
    # 6. Detailed Metrics Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    metrics_text = f"""
    ISSUER COST METRICS
    {'='*45}
    
    {"Metric":<25} {"Rules":<8} {"ML":<8} {"Hybrid":<8}
    {'-'*45}
    Total Cost to Issuer:
      ${rules_results['total_cost']:>6,.0f}   ${ml_results['total_cost']:>6,.0f}   ${hybrid_results['total_cost']:>6,.0f}
    
    Fraud Missed Cost:
      ${rules_results['fraud_missed_cost']:>6,.0f}   ${ml_results['fraud_missed_cost']:>6,.0f}   ${hybrid_results['fraud_missed_cost']:>6,.0f}
    
    False Positive Cost:
      ${rules_results['false_positive_cost']:>6,.0f}   ${ml_results['false_positive_cost']:>6,.0f}   ${hybrid_results['false_positive_cost']:>6,.0f}
    
    Detection Rate:
      {rules_results['detection_rate']*100:>5.1f}%    {ml_results['detection_rate']*100:>5.1f}%    {hybrid_results['detection_rate']*100:>5.1f}%
    
    False Positive Rate:
      {rules_results['false_positive_rate']*100:>5.2f}%    {ml_results['false_positive_rate']*100:>5.2f}%    {hybrid_results['false_positive_rate']*100:>5.2f}%
    
    Fraud Prevented (Amt):
      ${rules_results['fraud_prevented']:>6,.0f}   ${ml_results['fraud_prevented']:>6,.0f}   ${hybrid_results['fraud_prevented']:>6,.0f}
    
    Fraud Losses (Amt):
      ${rules_results['fraud_losses']:>6,.0f}   ${ml_results['fraud_losses']:>6,.0f}   ${hybrid_results['fraud_losses']:>6,.0f}
    
    WINNER: {"Hybrid System" if hybrid_results['total_cost'] == min(total_costs) else 
             "ML-Only" if ml_results['total_cost'] == min(total_costs) else "Rules-Only"}
    Cost Savings vs Best Alternative:
      ${max(total_costs) - min(total_costs):,.0f}
    
    Note: Costs from issuer perspective only.
    CNP fraud: 100% issuer liability
    Card-present: ~30% issuer (~70% merchant)
    Friendly fraud: ~50% issuer (~50% merchant)
    """
    
    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('cost_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Cost comparison visualization saved: cost_comparison_analysis.png")
    plt.show()

def visualize_optimal_threshold_analysis(thresholds, costs, detection_rates, fp_rates, optimal_threshold):
    """Visualize optimal threshold selection"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('ML Threshold Optimization Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cost vs Threshold
    ax1 = axes[0]
    ax1.plot(thresholds, costs, linewidth=3, color='darkblue', label='Total Cost')
    ax1.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Optimal Threshold: {optimal_threshold:.3f}')
    ax1.scatter([optimal_threshold], [costs[np.argmin(costs)]], 
               s=200, color='red', zorder=5, edgecolor='black', linewidth=2)
    
    ax1.set_xlabel('ML Score Threshold', fontsize=12)
    ax1.set_ylabel('Total Cost to Issuer ($)', fontsize=12)
    ax1.set_title('Total Cost vs Detection Threshold', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Detection Rate vs False Positive Rate
    ax2 = axes[1]
    
    costs_normalized = (costs - min(costs)) / (max(costs) - min(costs))
    scatter = ax2.scatter(fp_rates, detection_rates, c=costs_normalized, 
                         cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    optimal_idx = np.argmin(costs)
    ax2.scatter(fp_rates[optimal_idx], detection_rates[optimal_idx], 
               s=300, color='red', marker='*', zorder=5, 
               edgecolor='black', linewidth=2, label='Optimal Point')
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Normalized Cost', rotation=270, labelpad=20, fontsize=11)
    
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('Detection Rate', fontsize=12)
    ax2.set_title('Detection vs False Positives (Color = Cost)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    key_thresholds = [0.3, 0.5, 0.7, 0.9]
    for thresh in key_thresholds:
        idx = np.argmin(np.abs(thresholds - thresh))
        if abs(thresholds[idx] - optimal_threshold) > 0.05:
            ax2.annotate(f'{thresh:.1f}', 
                        (fp_rates[idx], detection_rates[idx]),
                        fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('threshold_optimization_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Threshold optimization visualization saved: threshold_optimization_analysis.png")
    plt.show()

def main():
    """Main execution function"""
    print("="*60)
    print("FRAUD DETECTION COST ANALYSIS (ISSUER PERSPECTIVE)")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    try:
        transactions = pd.read_csv('transactions.csv')
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], format='mixed')
        evaluation = pd.read_csv('hybrid_evaluation_test.csv')
        
        # IMPORTANT: evaluation is only the TEST SET (20% of data)
        # We need to use the same test set for all comparisons
        print(f"✓ Loaded {len(transactions):,} total transactions")
        print(f"✓ Loaded {len(evaluation):,} test set transactions")
        
        # Get test set transaction IDs and merge to ensure exact alignment
        test_txn_ids = evaluation['transaction_id'].values
        test_transactions = transactions[transactions['transaction_id'].isin(test_txn_ids)].copy()
        
        # Sort both by transaction_id to ensure alignment
        test_transactions = test_transactions.sort_values('transaction_id').reset_index(drop=True)
        evaluation_sorted = evaluation.sort_values('transaction_id').reset_index(drop=True)
        
        # Verify they match
        if len(test_transactions) != len(evaluation_sorted):
            print(f"WARNING: Mismatch - test_transactions has {len(test_transactions)} rows, evaluation has {len(evaluation_sorted)} rows")
            # Use only matching transaction IDs
            common_ids = set(test_transactions['transaction_id']) & set(evaluation_sorted['transaction_id'])
            test_transactions = test_transactions[test_transactions['transaction_id'].isin(common_ids)].sort_values('transaction_id').reset_index(drop=True)
            evaluation_sorted = evaluation_sorted[evaluation_sorted['transaction_id'].isin(common_ids)].sort_values('transaction_id').reset_index(drop=True)
        
        print(f"✓ Extracted {len(test_transactions):,} test transactions for comparison")
        
        # Use evaluation_sorted instead of evaluation from here on
        evaluation = evaluation_sorted

    except FileNotFoundError as e:
        print(f"Error: Required files not found. Please run the detection models script first.")
        print(f"Missing: {e.filename}")
        return
    
    # Initialize cost calculator
    cost_calc = FraudCostCalculator()
    
    print("\n" + "="*60)
    print("COST MODEL ASSUMPTIONS (ISSUER PERSPECTIVE)")
    print("="*60)
    print(f"CNP Fraud Loss Rate: {cost_calc.costs['cnp_fraud_loss_rate']*100:.0f}% (only 15% of merchants use 3DS)")
    print(f"Card-Present Fraud Loss Rate: {cost_calc.costs['cp_fraud_loss_rate']*100:.0f}% (edge cases; merchants liable post-EMV shift)")
    print(f"Friendly Fraud Loss Rate: {cost_calc.costs['friendly_fraud_loss_rate']*100:.0f}% (split based on dispute outcomes)")    
    print(f"Chargeback Fee: ${cost_calc.costs['chargeback_fee']}")
    print(f"Investigation Cost: ${cost_calc.costs['investigation_cost']}")
    print(f"False Positive Loss: {cost_calc.costs['false_positive_loss']*100:.0f}% of transaction amount")
    print(f"Customer Churn Rate (after FP): {cost_calc.costs['false_positive_churn_rate']*100:.1f}%")
    print(f"Customer Lifetime Value: ${cost_calc.costs['customer_lifetime_value']:,}")
    print(f"Manual Review Cost: ${cost_calc.costs['review_cost']}")
    
    # 1. Evaluate Hybrid System
    print("\n" + "="*60)
    print("EVALUATING HYBRID SYSTEM")
    print("="*60)
    hybrid_results = cost_calc.calculate_system_costs(evaluation)
    print(f"Total Cost to Issuer: ${hybrid_results['total_cost']:,.2f}")
    print(f"Detection Rate: {hybrid_results['detection_rate']*100:.1f}%")
    print(f"False Positive Rate: {hybrid_results['false_positive_rate']*100:.2f}%")
    
    # 2. Evaluate Rules-Only System
    print("\n" + "="*60)
    print("EVALUATING RULES-ONLY SYSTEM")
    print("="*60)
    rules_eval = create_rules_only_system(test_transactions)
    rules_results = cost_calc.calculate_system_costs(rules_eval)
    print(f"Total Cost to Issuer: ${rules_results['total_cost']:,.2f}")
    print(f"Detection Rate: {rules_results['detection_rate']*100:.1f}%")
    print(f"False Positive Rate: {rules_results['false_positive_rate']*100:.2f}%")
    
    # 3. Find Optimal ML Threshold
    print("\n" + "="*60)
    print("OPTIMIZING ML-ONLY SYSTEM")
    print("="*60)
    print("Finding optimal threshold...")
    
    ml_scores = evaluation['ml_score'].values
    optimal_threshold, optimal_cost, thresholds, costs, detection_rates, fp_rates = \
        find_optimal_ml_threshold(evaluation, ml_scores, cost_calc)
    
    print(f"✓ Optimal threshold found: {optimal_threshold:.3f}")
    print(f"  Optimal cost to issuer: ${optimal_cost:,.2f}")
    
    # Evaluate ML at optimal threshold
    ml_eval = create_ml_only_system(evaluation, ml_scores, optimal_threshold)
    ml_results = cost_calc.calculate_system_costs(ml_eval)
    print(f"Total Cost to Issuer: ${ml_results['total_cost']:,.2f}")
    print(f"Detection Rate: {ml_results['detection_rate']*100:.1f}%")
    print(f"False Positive Rate: {ml_results['false_positive_rate']*100:.2f}%")
    
    # 4. Compare Systems
    print("\n" + "="*60)
    print("SYSTEM COMPARISON")
    print("="*60)
    
    system_names = ['Rules-Only', 'ML-Only', 'Hybrid']
    all_results = [rules_results, ml_results, hybrid_results]
    
    comparison_df = pd.DataFrame({
        'System': system_names,
        'Total Cost': [r['total_cost'] for r in all_results],
        'Fraud Missed Cost': [r['fraud_missed_cost'] for r in all_results],
        'False Positive Cost': [r['false_positive_cost'] for r in all_results],
        'Detection Rate': [r['detection_rate'] * 100 for r in all_results],
        'FP Rate': [r['false_positive_rate'] * 100 for r in all_results],
        'Fraud Prevented': [r['fraud_prevented'] for r in all_results],
    })
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Find winner
    best_system_idx = np.argmin([r['total_cost'] for r in all_results])
    best_system = system_names[best_system_idx]
    cost_savings = max([r['total_cost'] for r in all_results]) - min([r['total_cost'] for r in all_results])
    
    print(f"\n{'='*60}")
    print(f"WINNER: {best_system}")
    print(f"Cost Savings vs Worst System: ${cost_savings:,.2f}")
    print(f"Savings as % of Worst System: {cost_savings/max([r['total_cost'] for r in all_results])*100:.1f}%")
    print(f"{'='*60}")
    
    # 5. Cost Breakdown by Fraud Type
    print("\n" + "="*60)
    print("FRAUD COST BREAKDOWN BY TYPE (Missed Fraud Only)")
    print("="*60)
    
    for system_name, results in zip(system_names, all_results):
        print(f"\n{system_name}:")
        if results['fraud_cost_by_type']:
            for fraud_type, cost in sorted(results['fraud_cost_by_type'].items(), 
                                          key=lambda x: x[1], reverse=True):
                print(f"  {fraud_type:20s}: ${cost:>8,.2f}")
        else:
            print("  No fraud missed!")
    
    # 6. Calculate ROI
    print("\n" + "="*60)
    print("RETURN ON INVESTMENT (ROI)")
    print("="*60)
    
    # Baseline: no fraud detection (all fraud goes through) - ON TEST SET
    total_fraud_txns = test_transactions[test_transactions['is_fraud']].copy()
    no_detection_cost = sum([cost_calc.calculate_issuer_fraud_cost(row) 
                             for _, row in total_fraud_txns.iterrows()])
    
    for system, results in zip(system_names, all_results):
        savings = no_detection_cost - results['total_cost']
        roi = (savings / results['total_cost']) * 100 if results['total_cost'] > 0 else 0
        print(f"\n{system}:")
        print(f"  No Detection Cost: ${no_detection_cost:,.2f}")
        print(f"  With Detection Cost: ${results['total_cost']:,.2f}")
        print(f"  Savings: ${savings:,.2f}")
        print(f"  ROI: {roi:.1f}%")
    
    # 7. Generate Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    visualize_cost_comparison(rules_results, ml_results, hybrid_results, system_names)
    visualize_optimal_threshold_analysis(thresholds, costs, detection_rates, fp_rates, optimal_threshold)
    
    # Save detailed results
    comparison_df.to_csv('cost_comparison_results.csv', index=False)
    print("\n✓ Cost comparison results saved to: cost_comparison_results.csv")
    
    # 8. Document Model Assumptions
    print("\n" + "="*60)
    print("IMPORTANT: COST MODEL ASSUMPTIONS")
    print("="*60)
    print("""
This cost model reflects the CARD ISSUER PERSPECTIVE only.

Key Assumptions:
1. CNP Fraud: Issuer pays 85% (only ~15% of U.S. merchants use 3DS)
2. Card-Present Fraud: Issuer pays 15% (edge cases; post-EMV shift, merchants liable)
3. Friendly Fraud: Issuer pays 50% (split based on dispute outcomes)

Why This Matters:
- CNP fraud is where issuer bears most liability (85%)
- Card-present fraud mostly falls on merchants post-EMV shift (2015)
- 3DS adoption is low in U.S. (~15%) due to cart abandonment concerns
- This makes CNP fraud the issuer's biggest cost driver

Reality Check:
- Total ecosystem cost is higher than what we show
- Merchants also bear significant fraud costs
- Our model conservatively estimates issuer's direct costs
- This is appropriate for a card issuer evaluating fraud detection systems

For More Details:
See documentation in README.md about cost model assumptions and limitations.
    """)
    
    print("\n" + "="*60)
    print("COST ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  • cost_comparison_analysis.png")
    print("  • threshold_optimization_analysis.png")
    print("  • cost_comparison_results.csv")

if __name__ == "__main__":
    main()