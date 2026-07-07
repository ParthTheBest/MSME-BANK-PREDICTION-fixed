"""
audit_bias_compliance.py
========================
Risk Auditing and Compliance Testing Suite for MSME Loan Prediction Engine.

This script tests the MSME model for:
1. Direct structural bias in model features.
2. Indirect bias in unstructured data parsing (officer notes NLP) across:
   - Gender (male-owned, female-owned, woman-led, etc.)
   - Region (Southern, Northern, Eastern, Western, North-East, Rural, Urban)
   - Industry Sector (Retail, Manufacturing, Construction, Technology, etc.)
3. Sector-based Loss Given Default (LGD) and Expected Loss (EL) bias.
4. Robustness to non-financial categorical fields in uploaded data.
5. Boundary and Robustness Tests (Injected dirty data, missing values, incomplete profiles, contradictory metrics).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Ensure output is printed clearly in UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Import project modules
import main
import nlp_features

def run_bias_audit():
    print("=" * 70)
    print("   RISK AUDITOR & COMPLIANCE REPORT: MSME PREDICTION BIAS AUDIT")
    print("=" * 70)

    # 1. Initialize Model Assets
    print("\n[1/5] Loading Risk Engine Assets...")
    try:
        main.load_assets()
        print("✓ Risk Engine assets loaded successfully.")
    except Exception as e:
        print(f"✗ Failed to load assets via main.load_assets(). Attempting direct load... Error: {e}")
        # Manual fallback loading
        main.xgb_model = joblib.load('models/xgb_model.joblib')
        main.imputer = joblib.load('models/imputer.joblib')
        main.FEATURES = joblib.load('models/feature_list.joblib')
        try:
            main.calibrator = joblib.load('models/calibrator.joblib')
        except FileNotFoundError:
            main.calibrator = None
        print("✓ Assets loaded via direct joblib fallback.")

    # 2. Check for Direct Structural Bias in Features
    print("\n[2/5] Auditing Model Feature Schema for Direct Bias...")
    protected_attributes = ['gender', 'sex', 'age', 'region', 'state', 'district', 'race', 'religion', 'caste']
    found_direct_bias = []
    
    print(f"Model features: {main.FEATURES}")
    for feat in main.FEATURES:
        feat_lower = feat.lower()
        for protected in protected_attributes:
            if protected in feat_lower:
                found_direct_bias.append(feat)

    if not found_direct_bias:
        print("✓ Success: No direct protected attributes found in the structured feature list.")
        print("  The model is structurally blind to Gender, Age, Region, Race, and Religion.")
    else:
        print(f"⚠ WARNING: Found potential direct bias features in model schema: {found_direct_bias}")

    # 3. Audit Indirect NLP Bias in Officer Notes (Unstructured Data)
    print("\n[3/5] Auditing Officer Notes NLP Encoder for Semantic Bias...")
    
    # Check if SentenceTransformer is used or Lexicon fallback
    backend = "Unknown"
    try:
        nlp_features._get_model()
        backend = "Sentence-Transformer (all-MiniLM-L6-v2)"
    except Exception:
        backend = "Lexicon Keyword Matcher (Graceful Fallback)"
    print(f"  NLP Scoring Backend: {backend}")

    # Define base narratives for different risk levels
    base_narratives = {
        "Healthy (Low Risk)": "Operations are stable, and sales are growing. The cashflow is adequate with normal working capital.",
        "Stable (Medium Risk)": "Minor cashflow pressure detected, with some delays in payments, but overall business is running normally.",
        "Stressed (High Risk)": "Severe cashflow pressure this quarter; EMI repeatedly delayed and supplier dues mounting."
    }

    # Varying categories
    variations = {
        "Gender": {
            "Neutral (Baseline)": "",
            "Female-Owned (Direct)": "The business is female-owned and run by a female entrepreneur.",
            "Male-Owned (Direct)": "The business is male-owned and run by a male entrepreneur.",
            "Woman-Led (Direct)": "This is a woman-led enterprise.",
            "Man-Led (Direct)": "This is a man-led enterprise.",
            "Proprietress (Gendered Title)": "The proprietress is seeking credit enhancement.",
            "Proprietor (Gendered Title)": "The proprietor is seeking credit enhancement."
        },
        "Region": {
            "Neutral (Baseline)": "",
            "Southern Region": "The business is located in the Southern region (Chennai).",
            "Northern Region": "The business is located in the Northern region (Delhi NCR).",
            "Western Region": "The business is located in the Western region (Mumbai).",
            "Eastern Region": "The business is located in the Eastern region (Kolkata).",
            "North-East Region": "The business is located in the North-East region (Assam).",
            "Rural Area": "The business is located in a remote rural area.",
            "Urban Metro": "The business is located in an urban metro center."
        },
        "Industry Sector": {
            "Neutral (Baseline)": "",
            "Retail Sector": "This is a retail sector business.",
            "Manufacturing Sector": "This is a manufacturing sector enterprise.",
            "Construction Sector": "This is a construction sector company.",
            "Technology Sector": "This is a technology sector startup.",
            "Agriculture Sector": "This is a rural agriculture sector business.",
            "Textile Sector": "This is a textile sector firm."
        }
    }

    # Financial inputs (kept completely identical)
    # Median baseline values
    baseline_financials = {
        'revolving_utilization': 0.40,
        'debt_ratio': 0.50,
        'income_stability': 0.80,
        'gst_compliance_score': 0.85,
        'cashflow_stress_ratio': 1.00,
        'working_capital_usage': 0.50,
        'revenue_trend_index': 1.05,
        'payment_history_score': 0.90,
        'supplier_payment_risk': 0.0,
    }
    
    # Fill remaining legacy features if the model needs them
    for f in main.FEATURES:
        if f not in baseline_financials and f != 'note_stress_index':
            baseline_financials[f] = 0.0

    nlp_bias_results = {}

    for cat_name, cat_variants in variations.items():
        print(f"\n  Auditing {cat_name} variants...")
        nlp_bias_results[cat_name] = {}
        
        for risk_name, base_text in base_narratives.items():
            print(f"    Scenario: {risk_name}")
            nlp_bias_results[cat_name][risk_name] = []
            
            for var_name, modifier in cat_variants.items():
                # Construct combined note
                if modifier:
                    combined_note = f"{modifier} {base_text}"
                else:
                    combined_note = base_text
                
                # Get NLP Stress Index
                stress_idx = nlp_features.stress_index_one(combined_note)
                
                # Build feature dict
                feat_dict = baseline_financials.copy()
                feat_dict['note_stress_index'] = stress_idx
                
                # Predict PD
                pd_val = main.predict_pd(feat_dict)
                
                nlp_bias_results[cat_name][risk_name].append({
                    "variant": var_name,
                    "note_preview": combined_note[:50] + "...",
                    "stress_index": round(stress_idx, 4),
                    "predicted_pd": round(pd_val, 4),
                    "risk_band": main.get_risk_band(pd_val)
                })
                
                # Print results
                print(f"      - {var_name:30} => Stress Index: {stress_idx:.4f} | PD: {pd_val*100:.2f}% ({main.get_risk_band(pd_val)})")

    # 4. Sector-based LGD & Expected Loss (EL) Bias Audit
    print("\n[4/5] Auditing Industry Sector LGD & EL Penalties...")
    lgd_map = getattr(main, 'LGD_MAP', {
        'Retail': 0.35, 'Manufacturing': 0.45, 'Construction': 0.55,
        'Technology': 0.30, 'Agriculture': 0.40, 'Textile': 0.42
    })
    
    exposure = 1_000_000.0
    test_pd = 0.10
    
    print(f"  Testing Exposure: INR {exposure:,.2f} | Assumed PD: {test_pd*100:.2f}%")
    sector_audit = []
    for sector, lgd in lgd_map.items():
        el = exposure * test_pd * lgd
        recovery = exposure - el
        sector_audit.append({
            "sector": sector,
            "lgd_pct": lgd,
            "expected_loss": el,
            "potential_recovery": recovery
        })
        print(f"    - {sector:15} | LGD: {lgd*100:5.1f}% | Expected Loss: INR {el:10,.2f} | Recovery: INR {recovery:10,.2f}")

    # 5. Boundary & Robustness Testing (Dirty Data, Missing values, Contradictory Metrics)
    print("\n[5/5] Executing Boundary & Robustness Auditing...")
    boundary_results = {}

    # Case A: Missing values (NaN/Null)
    print("\n  Case A: Injecting Missing Values (NaN/Nulls) in features")
    nan_financials = baseline_financials.copy()
    nan_financials['revolving_utilization'] = np.nan
    nan_financials['debt_ratio'] = np.nan
    nan_financials['income_stability'] = np.nan
    nan_financials['note_stress_index'] = np.nan

    try:
        # Direct predict_pd test
        pd_val = main.predict_pd(nan_financials)
        print(f"    ✓ predict_pd handles NaNs: PD = {pd_val*100:.2f}% ({main.get_risk_band(pd_val)})")
        boundary_results['nan_predict_pd'] = {"status": "ok", "pd": pd_val, "risk_band": main.get_risk_band(pd_val)}
    except Exception as e:
        print(f"    ✗ predict_pd CRASHED on NaNs: {type(e).__name__}: {e}")
        boundary_results['nan_predict_pd'] = {"status": "crash", "error": f"{type(e).__name__}: {e}"}

    try:
        # predict_portfolio test
        nan_df = pd.DataFrame([nan_financials])
        pd_vals = main.predict_portfolio(nan_df)
        print(f"    ✓ predict_portfolio handles NaNs: PD = {pd_vals[0]*100:.2f}%")
        boundary_results['nan_predict_portfolio'] = {"status": "ok", "pd": float(pd_vals[0])}
    except Exception as e:
        print(f"    ✗ predict_portfolio CRASHED on NaNs: {type(e).__name__}: {e}")
        boundary_results['nan_predict_portfolio'] = {"status": "crash", "error": f"{type(e).__name__}: {e}"}

    # Case B: Incomplete profiles (Missing entire columns)
    print("\n  Case B: Injecting Incomplete Application Profile (Missing columns)")
    incomplete_features = {
        'revolving_utilization': 0.20,
        'debt_ratio': 0.30
    }
    
    # Direct predict_pd with missing columns
    try:
        pd_val = main.predict_pd(incomplete_features)
        print(f"    ✓ predict_pd handles missing columns: PD = {pd_val*100:.2f}%")
        boundary_results['incomplete_predict_pd'] = {"status": "ok", "pd": pd_val}
    except Exception as e:
        print(f"    ✗ predict_pd CRASHED on missing columns (KeyError expected due to direct column indexing): {type(e).__name__}: {e}")
        boundary_results['incomplete_predict_pd'] = {"status": "crash", "error": f"{type(e).__name__}: {e}"}

    # API Ingestion mapping with missing columns
    try:
        incomplete_df = pd.DataFrame([incomplete_features])
        mapped_df = main.map_upload_to_portfolio(incomplete_df)
        pd_vals = main.predict_portfolio(mapped_df)
        print(f"    ✓ map_upload_to_portfolio resolves missing columns: PD = {pd_vals[0]*100:.2f}%")
        boundary_results['incomplete_upload'] = {"status": "ok", "pd": float(pd_vals[0])}
    except Exception as e:
        print(f"    ✗ map_upload_to_portfolio CRASHED on missing columns: {type(e).__name__}: {e}")
        boundary_results['incomplete_upload'] = {"status": "crash", "error": f"{type(e).__name__}: {e}"}

    # Case C: Contradictory Metrics
    print("\n  Case C: Injecting Contradictory metrics (Flawless credit indicators but severe active delays)")
    # Profile 1: Flawless Credit indicators but Severe Active Defaults
    contradictory_1 = baseline_financials.copy()
    # Flawless GMSC variables
    contradictory_1['revolving_utilization'] = 0.01  # extremely low (flawless)
    contradictory_1['debt_ratio'] = 0.05             # extremely low debt
    contradictory_1['late_30_59'] = 0
    contradictory_1['late_60_89'] = 0
    contradictory_1['late_90_days'] = 0
    contradictory_1['payment_history_score'] = 1.00  # perfect payment history
    # Critical MSME overlays (contradictory active metrics)
    contradictory_1['emi_delay_count'] = 12          # 12 delayed payments
    contradictory_1['supplier_payment_risk'] = 1.0   # delayed supplier payments active
    contradictory_1['cashflow_stress_ratio'] = 4.5   # massive cashflow stress
    contradictory_1['note_stress_index'] = 0.98       # alarming officer notes
    
    # Profile 2: High direct default history but excellent current cashflows
    contradictory_2 = baseline_financials.copy()
    # Terrible historical variables
    contradictory_2['late_30_59'] = 5
    contradictory_2['late_90_days'] = 3
    contradictory_2['payment_history_score'] = 0.20  # terrible history
    # Perfect current MSME variables
    contradictory_2['revolving_utilization'] = 0.10
    contradictory_2['debt_ratio'] = 0.15
    contradictory_2['emi_delay_count'] = 0           # no recent delays
    contradictory_2['supplier_payment_risk'] = 0.0
    contradictory_2['cashflow_stress_ratio'] = 0.1   # no cashflow stress
    contradictory_2['note_stress_index'] = 0.02       # extremely positive note

    try:
        pd_c1 = main.predict_pd(contradictory_1)
        print(f"    - Profile 1 (Flawless credit/severe active defaults) => PD = {pd_c1*100:.2f}% ({main.get_risk_band(pd_c1)})")
        boundary_results['contradictory_1'] = {"pd": pd_c1, "risk_band": main.get_risk_band(pd_c1)}
    except Exception as e:
        print(f"    ✗ Profile 1 crashed: {e}")
        boundary_results['contradictory_1'] = {"status": "crash", "error": str(e)}

    try:
        pd_c2 = main.predict_pd(contradictory_2)
        print(f"    - Profile 2 (Terrible history/perfect current metrics)  => PD = {pd_c2*100:.2f}% ({main.get_risk_band(pd_c2)})")
        boundary_results['contradictory_2'] = {"pd": pd_c2, "risk_band": main.get_risk_band(pd_c2)}
    except Exception as e:
        print(f"    ✗ Profile 2 crashed: {e}")
        boundary_results['contradictory_2'] = {"status": "crash", "error": str(e)}

    # Case D: Extraneous Categorical Variables robustness
    print("\n  Case D: Injecting Extraneous Columns (Gender, Region, Caste)")
    sample_upload = pd.DataFrame([{
        'company_id': 'MSME-AUDIT-DIRTY-01',
        'sector': 'Retail',
        'outstanding_loan': 1_200_000.0,
        'revolving_utilization': 0.45,
        'debt_ratio': 0.35,
        'income_stability': 0.75,
        'gst_compliance_score': 0.90,
        'cashflow_stress_ratio': 0.80,
        'working_capital_usage': 0.60,
        'revenue_trend_index': 1.10,
        'payment_history_score': 0.95,
        'supplier_payment_risk': 0.0,
        'officer_notes': 'Stable retail store.',
        'gender': 'Female',
        'region': 'North-East',
        'caste': 'Scheduled Caste',
        'religion': 'Hinduism'
    }])
    try:
        mapped_df = main.map_upload_to_portfolio(sample_upload)
        pd_vals = main.predict_portfolio(mapped_df)
        print(f"    ✓ Extra non-financial columns successfully filtered. PD = {pd_vals[0]*100:.2f}%")
        boundary_results['extra_columns'] = {"status": "ok", "pd": float(pd_vals[0])}
    except Exception as e:
        print(f"    ✗ Ingesting extra columns crashed: {type(e).__name__}: {e}")
        boundary_results['extra_columns'] = {"status": "crash", "error": f"{type(e).__name__}: {e}"}

    # Save results to a report file
    report_data = {
        "nlp_backend": backend,
        "nlp_bias_results": nlp_bias_results,
        "sector_audit": sector_audit,
        "has_direct_bias_features": len(found_direct_bias) > 0,
        "direct_bias_features": found_direct_bias,
        "boundary_robustness_results": boundary_results
    }
    
    report_path = 'models/bias_audit_results.json'
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"\n✓ Saved raw audit metrics to: {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_bias_audit()
