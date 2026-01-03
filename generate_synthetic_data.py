"""
Synthetic Credit Card Fraud Transaction Generator

Generates realistic transaction data with multiple fraud typologies including:
- Card testing
- Stolen card fraud (CNP)
- Account takeover
- Friendly fraud
- Synthetic identity fraud
- Refund fraud
- Application fraud (first-party)
- Lost/stolen physical card

Includes merchant EMV compliance, 3DS adoption, and dispute probabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import random

np.random.seed(42)
random.seed(42)

class FraudDataGenerator:
    def __init__(self, n_cardholders=1000, months=6, fraud_rate=0.015):
        self.n_cardholders = n_cardholders
        self.months = months
        self.fraud_rate = fraud_rate
        self.start_date = datetime(2024, 1, 1)
        
        # Merchant Category Codes
        self.mcc_categories = {
            5411: {'name': 'Grocery', 'avg_amount': 75, 'std': 35},
            5541: {'name': 'Gas Station', 'avg_amount': 45, 'std': 15},
            5812: {'name': 'Restaurant', 'avg_amount': 35, 'std': 25},
            5912: {'name': 'Pharmacy', 'avg_amount': 30, 'std': 20},
            5311: {'name': 'Department Store', 'avg_amount': 120, 'std': 80},
            5732: {'name': 'Electronics', 'avg_amount': 250, 'std': 200},
            5651: {'name': 'Clothing', 'avg_amount': 85, 'std': 50},
            5999: {'name': 'Misc Retail', 'avg_amount': 60, 'std': 40},
            5944: {'name': 'Jewelry', 'avg_amount': 350, 'std': 300},
            4121: {'name': 'Taxi', 'avg_amount': 25, 'std': 15},
        }
        
        self.zip_codes = [
            10001, 10002, 10003,  # NYC
            90001, 90002, 90003,  # LA
            60601, 60602, 60603,  # Chicago
            77001, 77002, 77003,  # Houston
            33101, 33102, 33109,  # Miami
            94102, 94103, 94104,  # SF
        ]
        
        self.cardholders = None
        self.transactions = None
        
    def generate_cardholder_profiles(self):
        """Generate cardholder profiles with correlated attributes"""
        profiles = []
        
        personas = ['convenience_user', 'revolver', 'transactor', 'light_user']
        persona_weights = [0.35, 0.40, 0.20, 0.05]
        
        for i in range(self.n_cardholders):
            persona = random.choices(personas, weights=persona_weights)[0]
            
            if persona == 'light_user':
                credit_score = np.random.normal(680, 60)
                base_limit = 2000
            elif persona == 'convenience_user':
                credit_score = np.random.normal(740, 50)
                base_limit = 8000
            elif persona == 'revolver':
                credit_score = np.random.normal(680, 70)
                base_limit = 5000
            else:  # transactor
                credit_score = np.random.normal(760, 40)
                base_limit = 15000
            
            credit_score = np.clip(credit_score, 300, 850)
            credit_limit = base_limit * (credit_score / 700) * np.random.lognormal(0, 0.3)
            credit_limit = round(credit_limit / 100) * 100
            
            account_age_days = int(np.random.exponential(500))
            account_age_days = min(account_age_days, 3650)
            
            home_zip = random.choice(self.zip_codes)
            
            if persona == 'light_user':
                monthly_txns_mean = 4
            elif persona == 'convenience_user':
                monthly_txns_mean = 12
            elif persona == 'revolver':
                monthly_txns_mean = 15
            else:  # transactor
                monthly_txns_mean = 25
            
            mcc_prefs = self._generate_merchant_preferences(persona)
            
            profile = {
                'cardholder_id': f'CH{i:06d}',
                'persona': persona,
                'credit_score': int(credit_score),
                'credit_limit': int(credit_limit),
                'account_age_days': account_age_days,
                'home_zip': home_zip,
                'monthly_txns_mean': monthly_txns_mean,
                'mcc_preferences': mcc_prefs,
            }
            profiles.append(profile)
        
        self.cardholders = pd.DataFrame(profiles)
        return self.cardholders
    
    def _generate_merchant_preferences(self, persona):
        """Generate merchant category preferences for a persona"""
        base_prefs = {
            5411: 0.25, 5541: 0.15, 5812: 0.20, 5912: 0.10, 5311: 0.10,
            5732: 0.05, 5651: 0.08, 5999: 0.05, 5944: 0.01, 4121: 0.01,
        }
        
        if persona == 'light_user':
            base_prefs[5411] = 0.35
            base_prefs[5541] = 0.25
            base_prefs[5812] = 0.15
        elif persona == 'transactor':
            base_prefs[5812] = 0.25
            base_prefs[5732] = 0.10
            base_prefs[5944] = 0.03
        
        total = sum(base_prefs.values())
        return {k: v/total for k, v in base_prefs.items()}
    
    def generate_legitimate_transactions(self):
        """Generate legitimate transaction sequences for all cardholders"""
        all_transactions = []
        
        for _, cardholder in self.cardholders.iterrows():
            ch_id = cardholder['cardholder_id']
            current_date = self.start_date
            current_balance = 0
            
            for month in range(self.months):
                n_txns = int(np.random.poisson(cardholder['monthly_txns_mean']))
                month_start = current_date + timedelta(days=30*month)
                
                txn_days = []
                for _ in range(n_txns):
                    if random.random() < 0.3:
                        day = random.randint(1, 5)
                    elif random.random() < 0.5:
                        day = random.randint(15, 20)
                    else:
                        day = random.randint(1, 30)
                    txn_days.append(day)
                
                txn_days.sort()
                prev_txn_time = None
                
                for day in txn_days:
                    txn_date = month_start + timedelta(days=day-1)
                    
                    mccs = list(cardholder['mcc_preferences'].keys())
                    probs = list(cardholder['mcc_preferences'].values())
                    mcc = random.choices(mccs, weights=probs)[0]
                    
                    hour = self._sample_transaction_hour(mcc, txn_date.weekday())
                    minute = random.randint(0, 59)
                    txn_timestamp = txn_date.replace(hour=hour, minute=minute)
                    
                    mcc_info = self.mcc_categories[mcc]
                    amount = np.random.lognormal(
                        np.log(mcc_info['avg_amount']), 
                        mcc_info['std'] / mcc_info['avg_amount']
                    )
                    amount = round(max(1, amount), 2)
                    
                    # Card present: 90% for legitimate
                    if mcc in [5411, 5541, 5812, 5912]:
                        card_present = random.random() < 0.95
                    else:
                        card_present = random.random() < 0.85
                    
                    # Merchant attributes
                    merchant_emv_compliant = random.random() < 0.85
                    merchant_uses_3ds = random.random() < 0.15 if not card_present else False
                    
                    if prev_txn_time:
                        time_since_last = (txn_timestamp - prev_txn_time).total_seconds() / 3600
                        if random.random() < 0.80:
                            distance = np.random.exponential(5)
                        else:
                            distance = np.random.exponential(50)
                    else:
                        time_since_last = None
                        distance = 0
                    
                    current_balance += amount
                    
                    avs_match = random.choices(
                        ['Full Match', 'ZIP Match Only', 'No Match', 'Not Available'],
                        weights=[0.92, 0.04, 0.02, 0.02]
                    )[0]
                    cvv_match = random.choices(
                        ['Match', 'No Match', 'Not Provided'],
                        weights=[0.96, 0.02, 0.02]
                    )[0]
                    
                    authorized = random.random() < 0.99
                    
                    txn = {
                        'transaction_id': f'TXN{len(all_transactions):09d}',
                        'cardholder_id': ch_id,
                        'timestamp': txn_timestamp,
                        'amount': amount,
                        'mcc': mcc,
                        'merchant_name': mcc_info['name'],
                        'card_present': card_present,
                        'merchant_emv_compliant': merchant_emv_compliant,
                        'merchant_uses_3ds': merchant_uses_3ds,
                        'authorized': authorized,
                        'avs_match': avs_match,
                        'cvv_match': cvv_match,
                        'merchant_zip': cardholder['home_zip'],
                        'distance_from_last_txn': distance,
                        'time_since_last_txn_hours': time_since_last,
                        'current_balance': current_balance,
                        'available_credit': cardholder['credit_limit'] - current_balance,
                        'is_fraud': False,
                        'fraud_type': None,
                        'disputed_by_cardholder': False,
                    }
                    
                    all_transactions.append(txn)
                    prev_txn_time = txn_timestamp
                
                if cardholder['persona'] in ['convenience_user', 'transactor']:
                    current_balance = 0
                elif cardholder['persona'] == 'revolver':
                    current_balance *= 0.7
        
        self.transactions = pd.DataFrame(all_transactions)
        return self.transactions
    
    def _sample_transaction_hour(self, mcc, day_of_week):
        """Sample transaction hour based on MCC and day of week"""
        is_weekend = day_of_week >= 5
        
        if mcc == 5411:  # Grocery
            if is_weekend:
                return int(np.clip(np.random.normal(11, 2), 6, 21))
            else:
                if random.random() < 0.5:
                    return int(np.clip(np.random.normal(7, 1), 6, 9))
                else:
                    return int(np.clip(np.random.normal(18, 1.5), 16, 21))
        elif mcc == 5812:  # Restaurant
            if random.random() < 0.4:
                return int(np.clip(np.random.normal(12, 1), 11, 14))
            else:
                return int(np.clip(np.random.normal(19, 1.5), 17, 22))
        elif mcc == 5541:  # Gas
            return random.randint(6, 22)
        else:
            return random.randint(9, 20)
    
    def inject_fraud(self):
        """Inject various fraud typologies into the transaction data"""
        if self.transactions is None:
            raise ValueError("Must generate legitimate transactions first")
        
        total_txns = len(self.transactions)
        fraud_txn_target = int(total_txns * self.fraud_rate)
        
        fraud_type_distribution = {
            'card_testing': 0.12,
            'stolen_card_fraud': 0.30,
            'account_takeover': 0.18,
            'friendly_fraud': 0.15,
            'synthetic_id': 0.08,
            'refund_fraud': 0.05,
            'application_fraud': 0.07,
            'lost_stolen_card': 0.05,
        }
        
        fraud_counts = {
            fraud_type: int(fraud_txn_target * pct)
            for fraud_type, pct in fraud_type_distribution.items()
        }
        
        self._inject_card_testing(fraud_counts['card_testing'])
        self._inject_stolen_card_fraud(fraud_counts['stolen_card_fraud'])
        self._inject_account_takeover(fraud_counts['account_takeover'])
        self._inject_friendly_fraud(fraud_counts['friendly_fraud'])
        self._inject_synthetic_id(fraud_counts['synthetic_id'])
        self._inject_refund_fraud(fraud_counts['refund_fraud'])
        self._inject_application_fraud(fraud_counts['application_fraud'])
        self._inject_lost_stolen_card(fraud_counts['lost_stolen_card'])
        
        # Add dispute flags
        self._add_dispute_probabilities()
        
        return self.transactions
    
    def _inject_card_testing(self, n_fraud_cases):
        """Inject card testing fraud patterns"""
        new_txns = []
        cardholders_to_hit = random.sample(list(self.cardholders['cardholder_id']), 
                                           min(n_fraud_cases//10, len(self.cardholders)))
        
        for ch_id in cardholders_to_hit:
            ch_txns = self.transactions[self.transactions['cardholder_id'] == ch_id]
            if len(ch_txns) == 0:
                continue
            
            base_time = random.choice(ch_txns['timestamp'].values)
            base_time = pd.to_datetime(base_time)
            
            n_tests = random.randint(8, 15)
            for i in range(n_tests):
                test_time = base_time + timedelta(minutes=random.randint(1, 180))
                
                if random.random() < 0.5:
                    amount = 1.00
                elif random.random() < 0.3:
                    amount = round(random.uniform(2, 5), 2)
                else:
                    amount = round(random.uniform(10, 20), 2)
                
                mcc = random.choice(list(self.mcc_categories.keys()))
                authorized = random.random() < 0.4
                
                txn = {
                    'transaction_id': f'FRAUD{len(self.transactions) + len(new_txns):09d}',
                    'cardholder_id': ch_id,
                    'timestamp': test_time,
                    'amount': amount,
                    'mcc': mcc,
                    'merchant_name': self.mcc_categories[mcc]['name'],
                    'card_present': False,
                    'merchant_emv_compliant': True,
                    'merchant_uses_3ds': random.random() < 0.15,
                    'authorized': authorized,
                    'avs_match': random.choice(['No Match', 'Not Available']),
                    'cvv_match': random.choice(['No Match', 'Not Provided']),
                    'merchant_zip': random.choice(self.zip_codes),
                    'distance_from_last_txn': 0,
                    'time_since_last_txn_hours': 0.05,
                    'current_balance': 0,
                    'available_credit': 0,
                    'is_fraud': True,
                    'fraud_type': 'card_testing',
                    'disputed_by_cardholder': False,  # Will be set later
                }
                new_txns.append(txn)
        
        self.transactions = pd.concat([self.transactions, pd.DataFrame(new_txns)], 
                                      ignore_index=True)
    
    def _inject_stolen_card_fraud(self, n_fraud_cases):
        """Inject stolen card (CNP) fraud"""
        new_txns = []
        cardholders_to_hit = random.sample(list(self.cardholders['cardholder_id']), 
                                           min(n_fraud_cases, len(self.cardholders)))
        
        for ch_id in cardholders_to_hit:
            ch_txns = self.transactions[self.transactions['cardholder_id'] == ch_id]
            cardholder = self.cardholders[self.cardholders['cardholder_id'] == ch_id].iloc[0]
            
            if len(ch_txns) == 0:
                continue
            
            base_time = random.choice(ch_txns['timestamp'].values)
            base_time = pd.to_datetime(base_time)
            fraud_time = base_time.replace(hour=random.randint(1, 5))
            
            amount = cardholder['credit_limit'] * random.uniform(0.3, 0.8)
            amount = round(amount, 2)
            
            mcc = random.choice([5732, 5944, 5999])
            
            avs_match = random.choices(
                ['Full Match', 'ZIP Match Only', 'No Match'],
                weights=[0.3, 0.3, 0.4]
            )[0]
            cvv_match = random.choices(
                ['Match', 'No Match'],
                weights=[0.7, 0.3]
            )[0]
            
            merchant_uses_3ds = random.random() < 0.15
            
            txn = {
                'transaction_id': f'FRAUD{len(self.transactions) + len(new_txns):09d}',
                'cardholder_id': ch_id,
                'timestamp': fraud_time,
                'amount': amount,
                'mcc': mcc,
                'merchant_name': self.mcc_categories[mcc]['name'],
                'card_present': False,
                'merchant_emv_compliant': True,
                'merchant_uses_3ds': merchant_uses_3ds,
                'authorized': True,
                'avs_match': avs_match,
                'cvv_match': cvv_match,
                'merchant_zip': random.choice(self.zip_codes),
                'distance_from_last_txn': random.uniform(500, 3000),
                'time_since_last_txn_hours': random.uniform(0.5, 3),
                'current_balance': amount,
                'available_credit': cardholder['credit_limit'] - amount,
                'is_fraud': True,
                'fraud_type': 'stolen_card_fraud',
                'disputed_by_cardholder': False,
            }
            new_txns.append(txn)
        
        self.transactions = pd.concat([self.transactions, pd.DataFrame(new_txns)], 
                                      ignore_index=True)
    
    def _inject_account_takeover(self, n_fraud_cases):
        """Inject account takeover fraud"""
        new_txns = []
        cardholders_to_hit = random.sample(list(self.cardholders['cardholder_id']), 
                                           min(n_fraud_cases//3, len(self.cardholders)))
        
        for ch_id in cardholders_to_hit:
            cardholder = self.cardholders[self.cardholders['cardholder_id'] == ch_id].iloc[0]
            ch_txns = self.transactions[self.transactions['cardholder_id'] == ch_id]
            
            if len(ch_txns) == 0:
                continue
            
            base_time = ch_txns['timestamp'].max()
            base_time = pd.to_datetime(base_time)
            ato_start = base_time + timedelta(days=random.randint(1, 10))
            
            for i in range(random.randint(2, 4)):
                fraud_time = ato_start + timedelta(hours=i*random.uniform(0.5, 3))
                
                amount = cardholder['credit_limit'] * random.uniform(0.2, 0.5)
                amount = round(amount, 2)
                
                mcc = random.choice([5732, 5944, 5311])
                
                card_present = random.random() < 0.3
                merchant_emv_compliant = random.random() < 0.85
                merchant_uses_3ds = random.random() < 0.15 if not card_present else False
                
                txn = {
                    'transaction_id': f'FRAUD{len(self.transactions) + len(new_txns):09d}',
                    'cardholder_id': ch_id,
                    'timestamp': fraud_time,
                    'amount': amount,
                    'mcc': mcc,
                    'merchant_name': self.mcc_categories[mcc]['name'],
                    'card_present': card_present,
                    'merchant_emv_compliant': merchant_emv_compliant,
                    'merchant_uses_3ds': merchant_uses_3ds,
                    'authorized': True,
                    'avs_match': random.choice(['Full Match', 'ZIP Match Only']),
                    'cvv_match': 'Match',
                    'merchant_zip': random.choice(self.zip_codes),
                    'distance_from_last_txn': random.uniform(100, 1000),
                    'time_since_last_txn_hours': i*random.uniform(0.5, 3),
                    'current_balance': amount * (i+1),
                    'available_credit': cardholder['credit_limit'] - amount * (i+1),
                    'is_fraud': True,
                    'fraud_type': 'account_takeover',
                    'disputed_by_cardholder': False,
                }
                new_txns.append(txn)
        
        self.transactions = pd.concat([self.transactions, pd.DataFrame(new_txns)], 
                                      ignore_index=True)
    
    def _inject_friendly_fraud(self, n_fraud_cases):
        """Mark legitimate transactions as friendly fraud"""
        legit_txns = self.transactions[~self.transactions['is_fraud']].copy()
        
        if len(legit_txns) < n_fraud_cases:
            n_fraud_cases = len(legit_txns)
        
        fraud_indices = random.sample(list(legit_txns.index), n_fraud_cases)
        
        self.transactions.loc[fraud_indices, 'is_fraud'] = True
        self.transactions.loc[fraud_indices, 'fraud_type'] = 'friendly_fraud'
    
    def _inject_synthetic_id(self, n_fraud_cases):
        """Create synthetic identity fraud accounts"""
        n_synthetic = max(1, n_fraud_cases // 20)
        
        for i in range(n_synthetic):
            ch_id = f'SYNTH{i:04d}'
            
            profile = {
                'cardholder_id': ch_id,
                'persona': 'synthetic_id',
                'credit_score': random.randint(580, 650),
                'credit_limit': random.randint(500, 2000),
                'account_age_days': random.randint(0, 180),
                'home_zip': random.choice(self.zip_codes),
                'monthly_txns_mean': 8,
                'mcc_preferences': self._generate_merchant_preferences('light_user'),
            }
            
            self.cardholders = pd.concat([
                self.cardholders, 
                pd.DataFrame([profile])
            ], ignore_index=True)
    
    def _inject_refund_fraud(self, n_fraud_cases):
        """Mark transactions as refund fraud"""
        returnable_mccs = [5311, 5651, 5732]
        returnable_txns = self.transactions[
            (self.transactions['mcc'].isin(returnable_mccs)) & 
            (~self.transactions['is_fraud'])
        ].copy()
        
        if len(returnable_txns) < n_fraud_cases:
            n_fraud_cases = len(returnable_txns)
        
        fraud_indices = random.sample(list(returnable_txns.index), n_fraud_cases)
        
        self.transactions.loc[fraud_indices, 'is_fraud'] = True
        self.transactions.loc[fraud_indices, 'fraud_type'] = 'refund_fraud'
    
    def _inject_application_fraud(self, n_fraud_cases):
        """Inject first-party application fraud"""
        new_txns = []
        n_fraudsters = max(1, n_fraud_cases // 15)
        
        for i in range(n_fraudsters):
            ch_id = f'APPFRAUD{i:04d}'
            
            # New account with low credit score
            profile = {
                'cardholder_id': ch_id,
                'persona': 'application_fraud',
                'credit_score': random.randint(600, 680),
                'credit_limit': random.randint(1000, 3000),
                'account_age_days': random.randint(0, 90),  # Very new
                'home_zip': random.choice(self.zip_codes),
                'monthly_txns_mean': 10,
                'mcc_preferences': self._generate_merchant_preferences('light_user'),
            }
            
            self.cardholders = pd.concat([
                self.cardholders, 
                pd.DataFrame([profile])
            ], ignore_index=True)
            
            # Generate immediate high utilization transactions
            credit_limit = profile['credit_limit']
            remaining_credit = credit_limit
            
            # 10-15 transactions maxing out the card immediately
            for j in range(random.randint(10, 15)):
                if remaining_credit < 50:
                    break
                
                txn_date = self.start_date + timedelta(days=random.randint(0, 30))
                hour = random.randint(9, 21)
                txn_timestamp = txn_date.replace(hour=hour, minute=random.randint(0, 59))
                
                # Large transactions
                amount = min(remaining_credit, random.uniform(100, 500))
                amount = round(amount, 2)
                remaining_credit -= amount
                
                mcc = random.choice(list(self.mcc_categories.keys()))
                card_present = random.random() < 0.7
                
                txn = {
                    'transaction_id': f'FRAUD{len(self.transactions) + len(new_txns):09d}',
                    'cardholder_id': ch_id,
                    'timestamp': txn_timestamp,
                    'amount': amount,
                    'mcc': mcc,
                    'merchant_name': self.mcc_categories[mcc]['name'],
                    'card_present': card_present,
                    'merchant_emv_compliant': random.random() < 0.85,
                    'merchant_uses_3ds': random.random() < 0.15 if not card_present else False,
                    'authorized': True,
                    'avs_match': random.choice(['Full Match', 'ZIP Match Only']),
                    'cvv_match': 'Match',
                    'merchant_zip': profile['home_zip'],
                    'distance_from_last_txn': random.uniform(0, 50),
                    'time_since_last_txn_hours': random.uniform(0.5, 12),
                    'current_balance': credit_limit - remaining_credit,
                    'available_credit': remaining_credit,
                    'is_fraud': True,
                    'fraud_type': 'application_fraud',
                    'disputed_by_cardholder': False,
                }
                new_txns.append(txn)
        
        self.transactions = pd.concat([self.transactions, pd.DataFrame(new_txns)], 
                                      ignore_index=True)
    
    def _inject_lost_stolen_card(self, n_fraud_cases):
        """Inject lost/stolen physical card fraud"""
        new_txns = []
        cardholders_to_hit = random.sample(list(self.cardholders['cardholder_id']), 
                                           min(n_fraud_cases//4, len(self.cardholders)))
        
        for ch_id in cardholders_to_hit:
            ch_txns = self.transactions[self.transactions['cardholder_id'] == ch_id]
            cardholder = self.cardholders[self.cardholders['cardholder_id'] == ch_id].iloc[0]
            
            if len(ch_txns) == 0:
                continue
            
            # Card stolen at some point
            steal_time = random.choice(ch_txns['timestamp'].values)
            steal_time = pd.to_datetime(steal_time)
            
            # 3-8 transactions in 24-48 hours before card is reported
            for i in range(random.randint(3, 8)):
                fraud_time = steal_time + timedelta(hours=random.uniform(0, 48))
                
                # Moderate amounts (fraudster wants to fly under radar)
                amount = random.uniform(50, 300)
                amount = round(amount, 2)
                
                # Unusual MCCs for the cardholder
                all_mccs = list(self.mcc_categories.keys())
                mcc = random.choice(all_mccs)
                
                # Card present (physical card stolen)
                card_present = True
                merchant_emv_compliant = random.random() < 0.85
                
                # Different location from cardholder's home
                distance = random.uniform(100, 500)
                
                txn = {
                    'transaction_id': f'FRAUD{len(self.transactions) + len(new_txns):09d}',
                    'cardholder_id': ch_id,
                    'timestamp': fraud_time,
                    'amount': amount,
                    'mcc': mcc,
                    'merchant_name': self.mcc_categories[mcc]['name'],
                    'card_present': card_present,
                    'merchant_emv_compliant': merchant_emv_compliant,
                    'merchant_uses_3ds': False,
                    'authorized': True,
                    'avs_match': 'Full Match',
                    'cvv_match': 'Match',
                    'merchant_zip': random.choice(self.zip_codes),
                    'distance_from_last_txn': distance,
                    'time_since_last_txn_hours': random.uniform(0.5, 6),
                    'current_balance': amount * (i+1),
                    'available_credit': cardholder['credit_limit'] - amount * (i+1),
                    'is_fraud': True,
                    'fraud_type': 'lost_stolen_card',
                    'disputed_by_cardholder': False,
                    }
                new_txns.append(txn)
            self.transactions = pd.concat([self.transactions, pd.DataFrame(new_txns)], 
                                      ignore_index=True)
    
    def _add_dispute_probabilities(self):
        """Add dispute flags based on fraud type and amount"""
        def get_dispute_probability(fraud_type, amount):
            """Calculate probability that cardholder disputes the transaction"""
            base_prob = {
                'card_testing': 0.05,
                'stolen_card_fraud': 0.90,
                'account_takeover': 0.95,
                'friendly_fraud': 1.00,
                'refund_fraud': 0.30,
                'synthetic_id': 0.00,
                'application_fraud': 0.00,
                'lost_stolen_card': 0.85,
            }
            
            prob = base_prob.get(fraud_type, 0.90)
            
            # Adjust by amount
            if amount < 10:
                prob *= 0.3
            elif amount < 50:
                prob *= 0.7
            elif amount > 500:
                prob = min(prob * 1.1, 1.0)
            
            return prob
        
        for idx, row in self.transactions[self.transactions['is_fraud']].iterrows():
            prob = get_dispute_probability(row['fraud_type'], row['amount'])
            disputed = random.random() < prob
            self.transactions.at[idx, 'disputed_by_cardholder'] = disputed
    
    def add_aggregated_features(self):
        """Add velocity and aggregated features"""
        self.transactions = self.transactions.sort_values(['cardholder_id', 'timestamp'])
        
        self.transactions['txn_count_24h'] = 0
        self.transactions['txn_count_7d'] = 0
        self.transactions['spend_24h'] = 0.0
        self.transactions['spend_7d'] = 0.0
        
        for ch_id in self.transactions['cardholder_id'].unique():
            ch_mask = self.transactions['cardholder_id'] == ch_id
            ch_txns = self.transactions[ch_mask].copy()
            
            for idx in ch_txns.index:
                txn_time = ch_txns.loc[idx, 'timestamp']
                
                time_24h_ago = txn_time - timedelta(hours=24)
                recent_24h = ch_txns[
                    (ch_txns['timestamp'] >= time_24h_ago) & 
                    (ch_txns['timestamp'] < txn_time)
                ]
                self.transactions.loc[idx, 'txn_count_24h'] = len(recent_24h)
                self.transactions.loc[idx, 'spend_24h'] = recent_24h['amount'].sum()
                
                time_7d_ago = txn_time - timedelta(days=7)
                recent_7d = ch_txns[
                    (ch_txns['timestamp'] >= time_7d_ago) & 
                    (ch_txns['timestamp'] < txn_time)
                ]
                self.transactions.loc[idx, 'txn_count_7d'] = len(recent_7d)
                self.transactions.loc[idx, 'spend_7d'] = recent_7d['amount'].sum()
        
        return self.transactions
    
    def generate_full_dataset(self):
        """Generate complete dataset with all features"""
        print("Generating cardholder profiles...")
        self.generate_cardholder_profiles()
        print(f"Generated {len(self.cardholders)} cardholders")
        
        print("Generating legitimate transactions...")
        self.generate_legitimate_transactions()
        print(f"Generated {len(self.transactions)} legitimate transactions")
        
        print("Injecting fraud...")
        self.inject_fraud()
        print(f"Total transactions after fraud injection: {len(self.transactions)}")
        print(f"Fraud rate: {self.transactions['is_fraud'].mean():.2%}")
        
        print("Adding aggregated features...")
        self.add_aggregated_features()
        
        print("\nFraud type distribution:")
        print(self.transactions[self.transactions['is_fraud']]['fraud_type'].value_counts())
        
        print("\nDispute rate by fraud type:")
        dispute_by_type = self.transactions[self.transactions['is_fraud']].groupby('fraud_type')['disputed_by_cardholder'].mean()
        for fraud_type, rate in dispute_by_type.items():
            print(f"  {fraud_type:20s}: {rate*100:.1f}%")
        
        return self.cardholders, self.transactions

if __name__ == "__main__":
    generator = FraudDataGenerator(
        n_cardholders=500,
        months=6,
        fraud_rate=0.015
    )
    
    cardholders, transactions = generator.generate_full_dataset()
    
    cardholders.to_csv('cardholders.csv', index=False)
    transactions.to_csv('transactions.csv', index=False)
    
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Total Cardholders: {len(cardholders)}")
    print(f"Total Transactions: {len(transactions)}")
    print(f"Fraud Transactions: {transactions['is_fraud'].sum()}")
    print(f"Fraud Rate: {transactions['is_fraud'].mean():.2%}")
    print(f"Disputed Fraud: {transactions[transactions['is_fraud']]['disputed_by_cardholder'].sum()}")
    print(f"Dispute Rate: {transactions[transactions['is_fraud']]['disputed_by_cardholder'].mean():.1%}")
    print("\nCardholder Personas:")
    print(cardholders['persona'].value_counts())
    print("\nCard Present Distribution:")
    print(f"Card Present: {transactions['card_present'].mean():.1%}")
    print(f"Card Not Present: {(~transactions['card_present']).mean():.1%}")
    print("\nMerchant Compliance:")
    print(f"EMV Compliant: {transactions['merchant_emv_compliant'].mean():.1%}")
    print(f"Uses 3DS (CNP only): {transactions[~transactions['card_present']]['merchant_uses_3ds'].mean():.1%}")

    print("\nSample of transactions:")
    print(transactions[['transaction_id', 'cardholder_id', 'timestamp', 
                        'amount', 'merchant_name', 'card_present', 'merchant_emv_compliant',
                        'is_fraud', 'fraud_type']].head(10))