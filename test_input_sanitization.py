"""
test_input_sanitization.py
==========================
Test Suite for MSME Application Input Sanitization and Formatting.
Simulates an external business applicant uploading a self-service CSV.
Tests how the ingestion pipeline handles formatting quirks, negative numbers, and overflow values.
"""

import sys
import numpy as np
import pandas as pd
import joblib

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import main

def run_sanitization_tests():
    print("=" * 80)
    print("  INPUT SANITIZATION & TEST SUITE — SELF-SERVICE SUBMISSION AUDIT")
    print("=" * 80)

    # Load assets
    try:
        main.load_assets()
    except Exception:
        main.xgb_model = joblib.load('models/xgb_model.joblib')
        main.imputer = joblib.load('models/imputer.joblib')
        main.FEATURES = joblib.load('models/feature_list.joblib')
        try:
            main.calibrator = joblib.load('models/calibrator.joblib')
        except FileNotFoundError:
            main.calibrator = None

    # Define test applications
    test_apps = [
        {
            "case_id": "CLEAN_BASELINE",
            "description": "Standard clean numerical inputs (Control)",
            "company_id": "MSME-CLEAN",
            "sector": "Retail",
            "outstanding_loan": 1000000.0,
            "revolving_utilization": 0.45,
            "debt_ratio": 0.50,
            "gst_compliance_score": 0.90,
            "cashflow_stress_ratio": 1.20,
            "working_capital_usage": 0.50,
            "revenue_trend_index": 1.10,
            "payment_history_score": 0.95,
            "supplier_payment_risk": 0.0,
            "officer_notes": "Clean application profile."
        },
        {
            "case_id": "FORMATTING_QUIRKS",
            "description": "Strings with currency symbols, percent signs, and commas",
            "company_id": "MSME-FORMAT-QUIRKS",
            "sector": "Retail",
            "outstanding_loan": "$1,500,000",          # Currency symbol and commas
            "revolving_utilization": "45%",             # Percent sign
            "debt_ratio": "0.55",                       # Standard
            "gst_compliance_score": "95%",              # Percent sign
            "cashflow_stress_ratio": "1,25",            # Comma separator or decimal error
            "working_capital_usage": "50%",             # Percent sign
            "revenue_trend_index": "1.10",
            "payment_history_score": "0.95",
            "supplier_payment_risk": "0.0",
            "officer_notes": "Formatting quirks present."
        },
        {
            "case_id": "NEGATIVE_VALUES",
            "description": "Negative values where only positive numbers belong",
            "company_id": "MSME-NEGATIVE",
            "sector": "Retail",
            "outstanding_loan": -500000.0,             # Negative loan amount
            "revolving_utilization": -0.25,             # Negative utilization
            "debt_ratio": -1.50,                        # Negative debt ratio
            "gst_compliance_score": -0.10,              # Negative score
            "cashflow_stress_ratio": -0.80,             # Negative stress
            "working_capital_usage": -0.50,
            "revenue_trend_index": -0.50,
            "payment_history_score": -0.95,
            "supplier_payment_risk": -1.0,
            "officer_notes": "Injected negative financials."
        },
        {
            "case_id": "INTEGER_OVERFLOW",
            "description": "Massive numeric strings exceeding standard limits",
            "company_id": "MSME-OVERFLOW",
            "sector": "Retail",
            "outstanding_loan": 999999999999999999999999999999, # Massive loan
            "revolving_utilization": 1e10,              # Massive utilization
            "debt_ratio": 1e20,                         # Massive debt ratio
            "gst_compliance_score": 1.0,
            "cashflow_stress_ratio": 1e5,
            "working_capital_usage": 1.0,
            "revenue_trend_index": 1e3,
            "payment_history_score": 1.0,
            "supplier_payment_risk": 0.0,
            "officer_notes": "Injected astronomical numbers."
        }
    ]

    # Convert to DataFrame simulating uploaded CSV file
    df_raw = pd.DataFrame(test_apps)
    
    print("\n[1/3] Ingesting CSV via main.map_upload_to_portfolio...")
    df_mapped = main.map_upload_to_portfolio(df_raw)

    print("\n[2/3] Comparing Raw inputs to Mapped numeric values:")
    for idx, row in df_raw.iterrows():
        case_id = row['case_id']
        mapped_row = df_mapped[df_mapped['company_id'] == row['company_id']].iloc[0]
        
        print(f"\n  Case: {case_id} — {row['description']}")
        
        # Check outstanding loan
        raw_loan = row['outstanding_loan']
        mapped_loan = mapped_row.get('outstanding_loan', 'N/A')
        print(f"    - outstanding_loan     : {raw_loan:20} => Mapped: {mapped_loan}")
        
        # Check utilization
        raw_util = row['revolving_utilization']
        mapped_util = mapped_row.get('revolving_utilization', 'N/A')
        print(f"    - revolving_utilization: {raw_util:20} => Mapped: {mapped_util}")

        # Check debt ratio
        raw_debt = row['debt_ratio']
        mapped_debt = mapped_row.get('debt_ratio', 'N/A')
        print(f"    - debt_ratio           : {raw_debt:20} => Mapped: {mapped_debt}")

    print("\n[3/3] Scoring ingested profiles through XGBoost Model:")
    try:
        probs = main.predict_portfolio(df_mapped)
        for idx, row in df_mapped.iterrows():
            pd_val = probs[idx]
            case_id = df_raw.iloc[idx]['case_id']
            print(f"    - {case_id:20} => Predicted PD: {pd_val*100:6.2f}% ({main.get_risk_band(pd_val)})")
    except Exception as e:
        print(f"  ✗ Prediction CRASHED during scoring: {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
    print("  AUDITOR ANALYSIS OF INGESTION PIPELINE VULNERABILITIES:")
    print("=" * 80)
    
    # Analyze Formatting Quirks Case
    fq_row = df_mapped[df_mapped['company_id'] == 'MSME-FORMAT-QUIRKS'].iloc[0]
    if pd.isna(fq_row['revolving_utilization']) or fq_row['revolving_utilization'] == 0.0:
        print("  [!] VULNERABILITY FOUND: Formatting Quirks Silent Discard")
        print("      Raw revolving_utilization of '45%' was coerced to NaN and set to 0.0 or imputed.")
        print("      The engine's mapping failed to strip '$', '%', or commas before parsing.")
        print("      This causes applicant-submitted data to be replaced silently with baseline defaults.")

    # Analyze Negative Values Case
    neg_row = df_mapped[df_mapped['company_id'] == 'MSME-NEGATIVE'].iloc[0]
    if neg_row['revolving_utilization'] < 0:
        print("  [!] VULNERABILITY FOUND: Negative Value Bypassed Validation")
        print(f"      Negative revolving_utilization (-0.25) bypassed sanitization and entered the model.")
        print("      XGBoost received negative values where only positive numbers belong.")
        print("      This causes unpredictable, non-monotonic probability shifts (PD scored as very low risk).")

    # Analyze Overflow Case
    of_row = df_mapped[df_mapped['company_id'] == 'MSME-OVERFLOW'].iloc[0]
    if of_row['outstanding_loan'] > 1e15:
        print("  [!] VULNERABILITY FOUND: Overflow Values Bypassed Validation")
        print(f"      Outstanding loan value of {of_row['outstanding_loan']} bypassed limits.")
        print("      This can cause memory errors, database column overflow, or division by zero in calculations.")
    print("=" * 80)

if __name__ == "__main__":
    run_sanitization_tests()
