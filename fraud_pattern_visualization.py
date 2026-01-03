import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

def load_data():
    """Load the generated transaction data"""
    try:
        transactions = pd.read_csv('transactions.csv')
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], format='mixed')
        return transactions
    except FileNotFoundError:
        print("Error: transactions.csv not found. Please run the data generator first.")
        return None

def visualize_card_testing(transactions):
    """Visualize card testing patterns"""
    card_testing = transactions[transactions['fraud_type'] == 'card_testing'].copy()
    
    if len(card_testing) == 0:
        print("No card testing fraud found in dataset")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Card Testing Fraud Patterns', fontsize=16, fontweight='bold')
    
    # 1. Amount distribution
    ax1 = axes[0, 0]
    ax1.hist(card_testing['amount'], bins=30, color='crimson', alpha=0.7, edgecolor='black')
    ax1.axvline(card_testing['amount'].median(), color='darkred', linestyle='--', 
                linewidth=2, label=f'Median: ${card_testing["amount"].median():.2f}')
    ax1.set_xlabel('Transaction Amount ($)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Amount Distribution: Concentrated at Small Values ($1-$5)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: ${card_testing["amount"].mean():.2f}\n'
    stats_text += f'<$5: {(card_testing["amount"] <= 5).sum()} ({(card_testing["amount"] <= 5).mean()*100:.1f}%)\n'
    stats_text += f'=$1: {(card_testing["amount"] == 1.0).sum()} ({(card_testing["amount"] == 1.0).mean()*100:.1f}%)'
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Authorization rate comparison
    ax2 = axes[0, 1]
    legitimate = transactions[~transactions['is_fraud']]
    
    auth_rates = pd.DataFrame({
        'Card Testing': [
            card_testing['authorized'].mean() * 100,
            (1 - card_testing['authorized'].mean()) * 100
        ],
        'Legitimate': [
            legitimate['authorized'].mean() * 100,
            (1 - legitimate['authorized'].mean()) * 100
        ]
    }, index=['Approved', 'Declined'])
    
    auth_rates.plot(kind='bar', ax=ax2, color=['crimson', 'steelblue'], alpha=0.7)
    ax2.set_title('Authorization Rates: Card Testing Has High Decline Rate', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_xlabel('')
    ax2.legend(title='Transaction Type')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f%%')
    
    # 3. Time series showing burst pattern
    ax3 = axes[1, 0]
    
    # Group by cardholder to show burst patterns
    sample_cardholder = card_testing.groupby('cardholder_id').size().idxmax()
    ch_testing = card_testing[card_testing['cardholder_id'] == sample_cardholder].sort_values('timestamp')
    
    if len(ch_testing) > 0:
        # Create relative time in minutes from first transaction
        ch_testing['minutes_from_start'] = (
            (ch_testing['timestamp'] - ch_testing['timestamp'].min()).dt.total_seconds() / 60
        )
        
        ax3.scatter(ch_testing['minutes_from_start'], ch_testing['amount'], 
                   s=100, color='crimson', alpha=0.6, edgecolor='black', linewidth=1)
        ax3.plot(ch_testing['minutes_from_start'], ch_testing['amount'], 
                color='crimson', alpha=0.3, linewidth=1)
        
        # Mark declined transactions
        declined = ch_testing[~ch_testing['authorized']]
        ax3.scatter(declined['minutes_from_start'], declined['amount'], 
                   s=150, marker='x', color='darkred', linewidth=3, 
                   label=f'Declined ({len(declined)} txns)', zorder=5)
        
        ax3.set_xlabel('Minutes from First Transaction', fontsize=12)
        ax3.set_ylabel('Transaction Amount ($)', fontsize=12)
        ax3.set_title(f'Burst Pattern: {len(ch_testing)} Transactions in {ch_testing["minutes_from_start"].max():.0f} Minutes', 
                     fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. MCC distribution (should be random/scattered)
    ax4 = axes[1, 1]
    
    mcc_counts = card_testing['mcc'].value_counts().head(10)
    legitimate_mcc = legitimate['mcc'].value_counts()
    legitimate_mcc_normalized = (legitimate_mcc / legitimate_mcc.sum() * 100)
    
    x = np.arange(len(mcc_counts))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, 
                    (mcc_counts / mcc_counts.sum() * 100).values, 
                    width, label='Card Testing', color='crimson', alpha=0.7)
    
    # Get legitimate distribution for same MCCs
    legit_values = [legitimate_mcc_normalized.get(mcc, 0) for mcc in mcc_counts.index]
    bars2 = ax4.bar(x + width/2, legit_values, width, 
                    label='Legitimate', color='steelblue', alpha=0.7)
    
    ax4.set_xlabel('Merchant Category Code', fontsize=12)
    ax4.set_ylabel('Percentage of Transactions (%)', fontsize=12)
    ax4.set_title('MCC Distribution: Card Testing Shows Random Pattern (No Concentration)', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(mcc_counts.index, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('card_testing_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ Card testing visualization saved: card_testing_patterns.png")
    plt.show()

def visualize_cnp_fraud(transactions):
    """Visualize CNP fraud patterns"""
    cnp_fraud = transactions[transactions['fraud_type'] == 'cnp_fraud'].copy()
    legitimate = transactions[~transactions['is_fraud']].copy()
    
    if len(cnp_fraud) == 0:
        print("No CNP fraud found in dataset")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Card-Not-Present (CNP) Fraud Patterns', fontsize=16, fontweight='bold')
    
    # 1. Amount comparison
    ax1 = axes[0, 0]
    
    data_to_plot = [
        legitimate['amount'],
        cnp_fraud['amount']
    ]
    
    bp = ax1.boxplot(data_to_plot, labels=['Legitimate', 'CNP Fraud'], 
                     patch_artist=True, showmeans=True)
    
    colors = ['steelblue', 'crimson']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Transaction Amount ($)', fontsize=12)
    ax1.set_title('Amount Distribution: CNP Fraud Has Larger Transactions', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f'CNP Fraud:\nMean: ${cnp_fraud["amount"].mean():.2f}\nMedian: ${cnp_fraud["amount"].median():.2f}\n\n'
    stats_text += f'Legitimate:\nMean: ${legitimate["amount"].mean():.2f}\nMedian: ${legitimate["amount"].median():.2f}'
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes, 
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. AVS/CVV match rates
    ax2 = axes[0, 1]
    
    # AVS match comparison
    cnp_avs = cnp_fraud['avs_match'].value_counts(normalize=True) * 100
    legit_avs = legitimate['avs_match'].value_counts(normalize=True) * 100
    
    avs_comparison = pd.DataFrame({
        'CNP Fraud': cnp_avs,
        'Legitimate': legit_avs
    }).fillna(0)
    
    avs_comparison.plot(kind='bar', ax=ax2, color=['crimson', 'steelblue'], alpha=0.7)
    ax2.set_title('AVS Match Results: CNP Fraud Has More Mismatches', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_xlabel('AVS Match Result', fontsize=12)
    ax2.legend(title='Transaction Type')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Time of day distribution
    ax3 = axes[1, 0]
    
    cnp_fraud['hour'] = cnp_fraud['timestamp'].dt.hour
    legitimate['hour'] = legitimate['timestamp'].dt.hour
    
    hours = range(24)
    cnp_hourly = cnp_fraud['hour'].value_counts().reindex(hours, fill_value=0)
    legit_hourly = legitimate['hour'].value_counts().reindex(hours, fill_value=0)
    
    # Normalize to percentages
    cnp_hourly_pct = (cnp_hourly / cnp_hourly.sum() * 100)
    legit_hourly_pct = (legit_hourly / legit_hourly.sum() * 100)
    
    ax3.plot(hours, legit_hourly_pct, marker='o', linewidth=2, 
            label='Legitimate', color='steelblue', alpha=0.7)
    ax3.plot(hours, cnp_hourly_pct, marker='o', linewidth=2, 
            label='CNP Fraud', color='crimson', alpha=0.7)
    
    # Highlight late night hours (1-5am)
    ax3.axvspan(1, 5, alpha=0.2, color='red', label='Late Night (1-5 AM)')
    
    ax3.set_xlabel('Hour of Day', fontsize=12)
    ax3.set_ylabel('Percentage of Transactions (%)', fontsize=12)
    ax3.set_title('Time Distribution: CNP Fraud Peaks at Unusual Hours', fontsize=12)
    ax3.set_xticks(range(0, 24, 2))
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. MCC distribution for high-risk categories
    ax4 = axes[1, 1]
    
    high_risk_mccs = [5732, 5944, 5999]  # Electronics, Jewelry, Misc
    
    cnp_mcc = cnp_fraud['mcc'].value_counts()
    legit_mcc = legitimate['mcc'].value_counts()
    
    # Calculate percentage of transactions in high-risk categories
    cnp_high_risk_pct = (cnp_fraud['mcc'].isin(high_risk_mccs).sum() / len(cnp_fraud) * 100)
    legit_high_risk_pct = (legitimate['mcc'].isin(high_risk_mccs).sum() / len(legitimate) * 100)
    
    categories = ['High-Risk\nMerchants\n(Electronics,\nJewelry, Misc)', 'Other\nMerchants']
    cnp_values = [cnp_high_risk_pct, 100 - cnp_high_risk_pct]
    legit_values = [legit_high_risk_pct, 100 - legit_high_risk_pct]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, cnp_values, width, 
                    label='CNP Fraud', color='crimson', alpha=0.7)
    bars2 = ax4.bar(x + width/2, legit_values, width, 
                    label='Legitimate', color='steelblue', alpha=0.7)
    
    ax4.set_ylabel('Percentage of Transactions (%)', fontsize=12)
    ax4.set_title('Merchant Categories: CNP Fraud Targets High-Risk MCCs', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bars in [bars1, bars2]:
        ax4.bar_label(bars, fmt='%.1f%%')
    
    plt.tight_layout()
    plt.savefig('cnp_fraud_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ CNP fraud visualization saved: cnp_fraud_patterns.png")
    plt.show()

def visualize_account_takeover(transactions):
    """Visualize account takeover patterns"""
    ato_fraud = transactions[transactions['fraud_type'] == 'account_takeover'].copy()
    
    if len(ato_fraud) == 0:
        print("No account takeover fraud found in dataset")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Account Takeover (ATO) Fraud Patterns', fontsize=16, fontweight='bold')
    
    # 1. Show spending velocity spike for a sample cardholder
    ax1 = axes[0, 0]
    
    # Pick a cardholder with ATO
    sample_ch = ato_fraud['cardholder_id'].iloc[0]
    ch_all_txns = transactions[transactions['cardholder_id'] == sample_ch].sort_values('timestamp')
    ch_all_txns['cumulative_spend'] = ch_all_txns['amount'].cumsum()
    ch_all_txns['days_from_start'] = (ch_all_txns['timestamp'] - ch_all_txns['timestamp'].min()).dt.total_seconds() / 86400
    
    # Split into pre-ATO and ATO
    ato_start = ato_fraud[ato_fraud['cardholder_id'] == sample_ch]['timestamp'].min()
    pre_ato = ch_all_txns[ch_all_txns['timestamp'] < ato_start]
    ato_txns = ch_all_txns[ch_all_txns['timestamp'] >= ato_start]
    
    ax1.plot(pre_ato['days_from_start'], pre_ato['cumulative_spend'], 
            marker='o', linewidth=2, color='steelblue', alpha=0.7, label='Normal Activity')
    ax1.plot(ato_txns['days_from_start'], ato_txns['cumulative_spend'], 
            marker='o', linewidth=3, color='crimson', alpha=0.7, label='ATO Activity')
    
    # Mark ATO start
    ato_day = (ato_start - ch_all_txns['timestamp'].min()).total_seconds() / 86400
    ax1.axvline(ato_day, color='red', linestyle='--', linewidth=2, 
               label='ATO Event', alpha=0.7)
    
    ax1.set_xlabel('Days from First Transaction', fontsize=12)
    ax1.set_ylabel('Cumulative Spend ($)', fontsize=12)
    ax1.set_title('Spending Pattern: Sudden Velocity Spike After Takeover', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time between transactions
    ax2 = axes[0, 1]
    
    legitimate = transactions[~transactions['is_fraud']].copy()
    
    # Calculate time between transactions for each type
    ato_time_diffs = []
    for ch_id in ato_fraud['cardholder_id'].unique():
        ch_ato = ato_fraud[ato_fraud['cardholder_id'] == ch_id].sort_values('timestamp')
        if len(ch_ato) > 1:
            diffs = ch_ato['timestamp'].diff().dt.total_seconds() / 3600  # hours
            ato_time_diffs.extend(diffs.dropna().values)
    
    legit_time_diffs = []
    for ch_id in legitimate['cardholder_id'].unique()[:100]:  # Sample for speed
        ch_legit = legitimate[legitimate['cardholder_id'] == ch_id].sort_values('timestamp')
        if len(ch_legit) > 1:
            diffs = ch_legit['timestamp'].diff().dt.total_seconds() / 3600
            legit_time_diffs.extend(diffs.dropna().values)
    
    # Plot histograms
    ax2.hist(legit_time_diffs, bins=50, alpha=0.5, label='Legitimate', 
            color='steelblue', density=True, range=(0, 48))
    ax2.hist(ato_time_diffs, bins=50, alpha=0.7, label='ATO', 
            color='crimson', density=True, range=(0, 48))
    
    ax2.set_xlabel('Hours Between Transactions', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Transaction Timing: ATO Shows Rapid-Fire Transactions', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 48)
    
    # 3. Amount distribution
    ax3 = axes[1, 0]
    
    data_to_plot = [
        legitimate['amount'],
        ato_fraud['amount']
    ]
    
    bp = ax3.boxplot(data_to_plot, labels=['Legitimate', 'ATO Fraud'], 
                     patch_artist=True, showmeans=True)
    
    colors = ['steelblue', 'crimson']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Transaction Amount ($)', fontsize=12)
    ax3.set_title('Transaction Amounts: ATO Involves Larger Purchases', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Geographic distance from previous transaction
    ax4 = axes[1, 1]
    
    # Filter out null values
    ato_distances = ato_fraud[ato_fraud['distance_from_last_txn'].notna()]['distance_from_last_txn']
    legit_distances = legitimate[legitimate['distance_from_last_txn'].notna()]['distance_from_last_txn']
    
    # Create bins
    bins = [0, 25, 100, 500, 2000]
    labels = ['0-25\n(Local)', '25-100\n(Regional)', '100-500\n(Cross-state)', '500+\n(Cross-country)']
    
    ato_dist_binned = pd.cut(ato_distances, bins=bins, labels=labels)
    legit_dist_binned = pd.cut(legit_distances, bins=bins, labels=labels)
    
    ato_pct = ato_dist_binned.value_counts(normalize=True).sort_index() * 100
    legit_pct = legit_dist_binned.value_counts(normalize=True).sort_index() * 100
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, ato_pct.values, width, 
                    label='ATO Fraud', color='crimson', alpha=0.7)
    bars2 = ax4.bar(x + width/2, legit_pct.values, width, 
                    label='Legitimate', color='steelblue', alpha=0.7)
    
    ax4.set_ylabel('Percentage of Transactions (%)', fontsize=12)
    ax4.set_xlabel('Distance from Previous Transaction (miles)', fontsize=12)
    ax4.set_title('Geographic Pattern: ATO Shows Unusual Locations', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bars in [bars1, bars2]:
        ax4.bar_label(bars, fmt='%.1f%%')
    
    plt.tight_layout()
    plt.savefig('ato_fraud_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ Account takeover visualization saved: ato_fraud_patterns.png")
    plt.show()

def visualize_friendly_fraud(transactions):
    """Visualize friendly fraud patterns"""
    friendly_fraud = transactions[transactions['fraud_type'] == 'friendly_fraud'].copy()
    legitimate = transactions[~transactions['is_fraud']].copy()
    
    if len(friendly_fraud) == 0:
        print("No friendly fraud found in dataset")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Friendly Fraud / Chargeback Abuse Patterns', fontsize=16, fontweight='bold')
    
    # 1. Amount distribution
    ax1 = axes[0, 0]
    
    bins = np.linspace(0, max(legitimate['amount'].max(), friendly_fraud['amount'].max()), 40)
    ax1.hist(legitimate['amount'], bins=bins, alpha=0.5, label='Legitimate', 
            color='steelblue', density=True)
    ax1.hist(friendly_fraud['amount'], bins=bins, alpha=0.7, label='Friendly Fraud', 
            color='crimson', density=True)
    
    ax1.set_xlabel('Transaction Amount ($)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Amount Distribution: Friendly Fraud Resembles Legitimate Transactions', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    stats_text = f'Friendly Fraud:\nMean: ${friendly_fraud["amount"].mean():.2f}\n\n'
    stats_text += f'Legitimate:\nMean: ${legitimate["amount"].mean():.2f}'
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Merchant category distribution
    ax2 = axes[0, 1]
    
    ff_mcc = friendly_fraud['mcc'].value_counts().head(8)
    legit_mcc = legitimate['mcc'].value_counts()
    
    # Normalize
    ff_mcc_pct = (ff_mcc / len(friendly_fraud) * 100)
    legit_mcc_pct = pd.Series([legit_mcc.get(mcc, 0) / len(legitimate) * 100 for mcc in ff_mcc.index], 
                              index=ff_mcc.index)
    
    x = np.arange(len(ff_mcc))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, ff_mcc_pct.values, width, 
                    label='Friendly Fraud', color='crimson', alpha=0.7)
    bars2 = ax2.bar(x + width/2, legit_mcc_pct.values, width, 
                    label='Legitimate', color='steelblue', alpha=0.7)
    
    ax2.set_ylabel('Percentage of Transactions (%)', fontsize=12)
    ax2.set_xlabel('Merchant Category Code', fontsize=12)
    ax2.set_title('MCC Distribution: Similar Patterns (Hard to Distinguish)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ff_mcc.index, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Authorization rate (should be similar)
    ax3 = axes[1, 0]
    
    auth_comparison = pd.DataFrame({
        'Friendly Fraud': [
            friendly_fraud['authorized'].mean() * 100,
            (1 - friendly_fraud['authorized'].mean()) * 100
        ],
        'Legitimate': [
            legitimate['authorized'].mean() * 100,
            (1 - legitimate['authorized'].mean()) * 100
        ]
    }, index=['Approved', 'Declined'])
    
    auth_comparison.plot(kind='bar', ax=ax3, color=['crimson', 'steelblue'], alpha=0.7)
    ax3.set_title('Authorization Rates: Friendly Fraud Looks Legitimate at Transaction Time', fontsize=12)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_xlabel('')
    ax3.legend(title='Transaction Type')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.1f%%')
    
    # 4. Key insight: Why it's hard to detect
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    insight_text = """
    WHY FRIENDLY FRAUD IS HARD TO DETECT:
    
    ✗ Transaction amounts are normal
    ✗ Merchant categories are typical
    ✗ Authorization rates match legitimate
    ✗ AVS/CVV verification passes
    ✗ Geographic location is normal
    ✗ Timing is not suspicious
    
    ✓ Only detectable through:
       • History of previous disputes/chargebacks
       • Pattern of returns with same merchant
       • Cross-referencing delivery confirmation
       • Customer behavior analysis over time
    
    Detection Window: 30-120 days after transaction
    (when customer files chargeback)
    
    Key Metric: Dispute history per cardholder
    """
    
    ax4.text(0.1, 0.9, insight_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('friendly_fraud_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ Friendly fraud visualization saved: friendly_fraud_patterns.png")
    plt.show()

def visualize_refund_fraud(transactions):
    """Visualize refund fraud patterns"""
    refund_fraud = transactions[transactions['fraud_type'] == 'refund_fraud'].copy()
    legitimate = transactions[~transactions['is_fraud']].copy()
    
    if len(refund_fraud) == 0:
        print("No refund fraud found in dataset")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Refund Fraud Patterns', fontsize=16, fontweight='bold')
    
    # 1. MCC concentration
    ax1 = axes[0, 0]
    
    returnable_mccs = [5311, 5651, 5732]  # Dept store, clothing, electronics
    mcc_names = {5311: 'Dept Store', 5651: 'Clothing', 5732: 'Electronics'}
    
    refund_by_mcc = refund_fraud['mcc'].value_counts()
    legit_by_mcc = legitimate['mcc'].value_counts()
    
    # Calculate percentage
    refund_returnable_pct = (refund_fraud['mcc'].isin(returnable_mccs).sum() / len(refund_fraud) * 100)
    legit_returnable_pct = (legitimate['mcc'].isin(returnable_mccs).sum() / len(legitimate) * 100)
    
    categories = ['Returnable Items\n(Dept Store, Clothing,\nElectronics)', 'Other Items']
    refund_values = [refund_returnable_pct, 100 - refund_returnable_pct]
    legit_values = [legit_returnable_pct, 100 - legit_returnable_pct]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, refund_values, width, 
                    label='Refund Fraud', color='crimson', alpha=0.7)
    bars2 = ax1.bar(x + width/2, legit_values, width, 
                    label='Legitimate', color='steelblue', alpha=0.7)
    
    ax1.set_ylabel('Percentage of Transactions (%)', fontsize=12)
    ax1.set_title('Category Distribution: Refund Fraud Concentrates on Returnable Items', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        ax1.bar_label(bars, fmt='%.1f%%')
    
    # 2. Amount distribution
    ax2 = axes[0, 1]
    
    refund_returnable = refund_fraud[refund_fraud['mcc'].isin(returnable_mccs)]
    legit_returnable = legitimate[legitimate['mcc'].isin(returnable_mccs)]
    
    data_to_plot = [
        legit_returnable['amount'],
        refund_returnable['amount']
    ]
    
    bp = ax2.boxplot(data_to_plot, labels=['Legitimate', 'Refund Fraud'], 
                     patch_artist=True, showmeans=True)
    
    colors = ['steelblue', 'crimson']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Transaction Amount ($)', fontsize=12)
    ax2.set_title('Amount Distribution for Returnable Items', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Simulated return rate by cardholder
    ax3 = axes[1, 0]
    
    # Calculate "return rate" - proportion of refund fraud vs all transactions per cardholder
    refund_ch_counts = refund_fraud['cardholder_id'].value_counts()
    all_ch_counts = transactions.groupby('cardholder_id').size()
    
    # Get cardholders with refund fraud
    refund_cardholders = refund_ch_counts.index
    normal_cardholders = list(set(all_ch_counts.index) - set(refund_cardholders))[:50]
    
    # Calculate return rates
    refund_ch_rates = (refund_ch_counts / all_ch_counts[refund_cardholders] * 100)
    normal_ch_rates = pd.Series([
        (transactions[(transactions['cardholder_id'] == ch) & (transactions['fraud_type'] == 'refund_fraud')].shape[0] / 
         all_ch_counts[ch] * 100)
        for ch in normal_cardholders
    ])
    
    # Plot histograms
    ax3.hist(normal_ch_rates, bins=20, alpha=0.5, label='Normal Cardholders', 
            color='steelblue', range=(0, 100))
    ax3.hist(refund_ch_rates, bins=20, alpha=0.7, label='Refund Fraudsters', 
            color='crimson', range=(0, 100))
    
    ax3.set_xlabel('Return Rate (%)', fontsize=12)
    ax3.set_ylabel('Number of Cardholders', fontsize=12)
    ax3.set_title('Return Rate Distribution: Fraudsters Have Much Higher Return Rates', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Fraudsters:\nMean Rate: {refund_ch_rates.mean():.1f}%\n\n'
    stats_text += f'Normal:\nMean Rate: {normal_ch_rates.mean():.1f}%'
    ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 4. Detection insights
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    insight_text = """
    REFUND FRAUD DETECTION SIGNALS:
    
    ✓ High concentration in returnable categories
       • Clothing (wardrobing)
       • Electronics (return after use)
       • Department stores
    
    ✓ Elevated return rate
       • Normal: 5-15% return rate
       • Fraud: 40-80% return rate
    
    ✓ Pattern analysis
       • Same merchant repeatedly
       • Short time-to-return
       • Alternating between accounts/cards
    
    ✓ Cross-referencing
       • Original item vs. returned item
       • Delivery confirmation
       • Product condition reports
    
    Detection Window: 7-30 days
    (between purchase and return request)
    """
    
    ax4.text(0.1, 0.9, insight_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('refund_fraud_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ Refund fraud visualization saved: refund_fraud_patterns.png")
    plt.show()

def create_summary_dashboard(transactions):
    """Create a summary dashboard showing all fraud types"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fraud Typology Summary Dashboard', fontsize=18, fontweight='bold')
    
    fraud_txns = transactions[transactions['is_fraud']].copy()
    legitimate = transactions[~transactions['is_fraud']].copy()
    
    # 1. Fraud type distribution
    ax1 = axes[0, 0]
    fraud_counts = fraud_txns['fraud_type'].value_counts()
    colors_map = {'card_testing': 'red', 'cnp_fraud': 'orange', 
                  'account_takeover': 'purple', 'friendly_fraud': 'pink',
                  'refund_fraud': 'brown', 'synthetic_id': 'darkred'}
    colors = [colors_map.get(ft, 'gray') for ft in fraud_counts.index]
    
    ax1.pie(fraud_counts.values, labels=fraud_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax1.set_title('Fraud Distribution by Type', fontsize=12, fontweight='bold')
    
    # 2. Amount comparison by fraud type
    ax2 = axes[0, 1]
    fraud_types = fraud_txns['fraud_type'].dropna().unique()
    amounts_by_type = [fraud_txns[fraud_txns['fraud_type'] == ft]['amount'].values 
                       for ft in fraud_types]
    
    bp = ax2.boxplot(amounts_by_type, labels=fraud_types, patch_artist=True)
    for patch, ft in zip(bp['boxes'], fraud_types):
        patch.set_facecolor(colors_map.get(ft, 'gray'))
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Amount ($)', fontsize=11)
    ax2.set_title('Transaction Amounts by Fraud Type', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(fraud_types, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Authorization rates
    ax3 = axes[0, 2]
    auth_by_type = fraud_txns.groupby('fraud_type')['authorized'].mean() * 100
    legit_auth = legitimate['authorized'].mean() * 100
    
    auth_comparison = pd.concat([auth_by_type, pd.Series({'Legitimate': legit_auth})])
    colors_list = [colors_map.get(ft, 'steelblue') for ft in auth_comparison.index]
    
    bars = ax3.bar(range(len(auth_comparison)), auth_comparison.values, 
                   color=colors_list, alpha=0.7)
    ax3.set_xticks(range(len(auth_comparison)))
    ax3.set_xticklabels(auth_comparison.index, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Authorization Rate (%)', fontsize=11)
    ax3.set_title('Authorization Rates by Type', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.bar_label(bars, fmt='%.0f%%', fontsize=8)
    
    # 4. Transaction count over time
    ax4 = axes[1, 0]
    fraud_txns['date'] = fraud_txns['timestamp'].dt.date
    legitimate['date'] = legitimate['timestamp'].dt.date
    
    fraud_daily = fraud_txns.groupby('date').size()
    legit_daily = legitimate.groupby('date').size()
    
    ax4.plot(legit_daily.index, legit_daily.values, label='Legitimate', 
            linewidth=2, color='steelblue', alpha=0.7)
    ax4.plot(fraud_daily.index, fraud_daily.values, label='Fraud', 
            linewidth=2, color='crimson', alpha=0.7)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_ylabel('Transaction Count', fontsize=11)
    ax4.set_title('Transaction Volume Over Time', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # 5. Card present vs CNP
    ax5 = axes[1, 1]
    fraud_cp = fraud_txns['card_present'].value_counts(normalize=True) * 100
    legit_cp = legitimate['card_present'].value_counts(normalize=True) * 100
    
    cp_comparison = pd.DataFrame({
        'Fraud': fraud_cp,
        'Legitimate': legit_cp
    }).fillna(0)
    
    cp_comparison.plot(kind='bar', ax=ax5, color=['crimson', 'steelblue'], alpha=0.7)
    ax5.set_title('Card Present vs CNP', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Percentage (%)', fontsize=11)
    ax5.set_xlabel('Transaction Type', fontsize=11)
    ax5.set_xticklabels(['Card Not Present', 'Card Present'], rotation=0, fontsize=9)
    ax5.legend(title='', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Key statistics table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    total_txns = len(transactions)
    fraud_count = len(fraud_txns)
    fraud_rate = fraud_count / total_txns * 100
    total_fraud_amount = fraud_txns['amount'].sum()
    total_legit_amount = legitimate['amount'].sum()
    
    stats_text = f"""
    DATASET STATISTICS
    {'='*40}
    
    Total Transactions:     {total_txns:,}
    Fraudulent:             {fraud_count:,}
    Fraud Rate:             {fraud_rate:.2f}%
    
    Total Fraud Amount:     ${total_fraud_amount:,.2f}
    Total Legit Amount:     ${total_legit_amount:,.2f}
    
    Avg Fraud Amount:       ${fraud_txns['amount'].mean():.2f}
    Avg Legit Amount:       ${legitimate['amount'].mean():.2f}
    
    FRAUD BREAKDOWN:
    {'-'*40}
    """
    
    for ft in fraud_txns['fraud_type'].dropna().unique():
        count = (fraud_txns['fraud_type'] == ft).sum()
        pct = count / fraud_count * 100
        stats_text += f"\n    {ft:20s} {count:4d} ({pct:4.1f}%)"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('fraud_summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Summary dashboard saved: fraud_summary_dashboard.png")
    plt.show()

def main():
    """Main function to run all visualizations"""
    print("="*60)
    print("FRAUD PATTERN VISUALIZATION TOOL")
    print("="*60)
    
    # Load data
    print("\nLoading transaction data...")
    transactions = load_data()
    
    if transactions is None:
        return
    
    print(f"Loaded {len(transactions):,} transactions")
    print(f"Fraud rate: {transactions['is_fraud'].mean():.2%}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    print("-" * 60)
    
    visualize_card_testing(transactions)
    visualize_cnp_fraud(transactions)
    visualize_account_takeover(transactions)
    visualize_friendly_fraud(transactions)
    visualize_refund_fraud(transactions)
    create_summary_dashboard(transactions)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  • card_testing_patterns.png")
    print("  • cnp_fraud_patterns.png")
    print("  • ato_fraud_patterns.png")
    print("  • friendly_fraud_patterns.png")
    print("  • refund_fraud_patterns.png")
    print("  • fraud_summary_dashboard.png")

if __name__ == "__main__":
    main()