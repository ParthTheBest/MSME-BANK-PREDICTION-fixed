"""
retrain_calibrated.py
======================
Retrains the MSME default model with:
  1. Only 10 LEADING INDICATOR features (removes late_60_89, late_90_days,
     emi_delay_count, open_credit_lines, real_estate_loans, num_dependents)
  2. XGBoost monotonic constraints — enforces economic logic
     (higher utilization / debt => higher risk; better GST => lower risk)
  3. Isotonic calibration via CalibratedClassifierCV so scores are
     genuinely distributed across 0-100% (not bimodal 0% / 100%)
  4. Saves: xgb_model.joblib, calibrator.joblib, shap_explainer.joblib,
            imputer.joblib, feature_list.joblib, performance_report.json
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score
import shap
import joblib
import os
import json

# ── Clean Leading-Indicator Feature Set (10 features, no data leakage) ──────
CLEAN_FEATURES = [
    'revolving_utilization',   # credit utilization trend → higher = worse
    'debt_ratio',              # leverage ratio → higher = worse
    'income_stability',        # revenue consistency → lower = worse
    'gst_compliance_score',    # GST filing regularity → lower = worse
    'cashflow_stress_ratio',   # cashflow pressure index → higher = worse
    'working_capital_usage',   # WC line draw → higher = worse
    'revenue_trend_index',     # MoM revenue trend → lower = worse
    'payment_history_score',   # overall payment track record → lower = worse
    'supplier_payment_risk',   # creditor delay flag → higher = worse
    'note_stress_index',       # NLP stress score from officer notes → higher = worse
]

# Monotonic constraints: +1 = feature increase raises PD, -1 = raises PD decreases
MONOTONE = {
    'revolving_utilization':   1,
    'debt_ratio':              1,
    'income_stability':       -1,
    'gst_compliance_score':   -1,
    'cashflow_stress_ratio':   1,
    'working_capital_usage':   1,
    'revenue_trend_index':    -1,
    'payment_history_score':  -1,
    'supplier_payment_risk':   1,
    'note_stress_index':       1,
}

SEED = 42


def make_msme_data(n: int = 50_000, seed: int = SEED) -> pd.DataFrame:
    """
    Generate synthetic MSME borrower dataset with ONLY leading indicators.
    Features are causally linked to defaults WITHOUT using lagging delinquency markers.
    Default rate: ~17%.
    """
    rng = np.random.default_rng(seed)
    d = (rng.random(n) < 0.17).astype(int)
    noise = lambda s: rng.normal(0, s, n)

    df = pd.DataFrame()
    df['default_flag'] = d

    # Defaults cluster around poor financial health
    df['revolving_utilization'] = np.clip(
        rng.beta(2, 5, n) + d * 0.38 + noise(0.05), 0.01, 0.99)

    df['debt_ratio'] = np.clip(
        rng.exponential(0.35, n) + d * 0.65 + noise(0.06), 0.01, 3.0)

    df['income_stability'] = np.clip(
        rng.beta(4, 2, n) * (1 - d * 0.38) + noise(0.04), 0.0, 1.0)

    df['gst_compliance_score'] = np.clip(
        rng.beta(5, 2, n) * (1 - d * 0.50) + noise(0.04), 0.0, 1.0)

    df['cashflow_stress_ratio'] = np.clip(
        rng.exponential(0.45, n) + d * 0.85 + noise(0.10), 0.0, 5.0)

    df['working_capital_usage'] = np.clip(
        rng.beta(2, 5, n) + d * 0.33 + noise(0.04), 0.0, 1.0)

    df['revenue_trend_index'] = np.clip(
        rng.normal(1.05, 0.18, n) - d * 0.38 + noise(0.05), 0.2, 2.0)

    df['payment_history_score'] = np.clip(
        rng.beta(6, 2, n) * (1 - d * 0.42) + noise(0.04), 0.0, 1.0)

    # Supplier payment risk (flag: 0 or 1)
    raw = rng.random(n)
    df['supplier_payment_risk'] = (raw < (0.08 + d * 0.42)).astype(float)

    # NLP stress index from officer notes proxy
    df['note_stress_index'] = np.clip(
        rng.beta(2, 6, n) + d * 0.55 + noise(0.06), 0.0, 1.0)

    return df


def train():
    print()
    print("=" * 65)
    print("  IDBI MSME — CALIBRATED RISK MODEL TRAINING PIPELINE")
    print("=" * 65)

    # ── 1. Data ────────────────────────────────────────────────────
    print("\n[1/6] Generating synthetic MSME dataset (50,000 borrowers)…")
    df = make_msme_data(n=50_000)
    X = df[CLEAN_FEATURES].values
    y = df['default_flag'].values
    print(f"      Default rate : {y.mean()*100:.1f}%")
    print(f"      Features     : {len(CLEAN_FEATURES)} leading indicators (no leakage)")

    # Stratified 70 / 15 / 15 split (train / calibration / test)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
    print(f"      Train {len(X_tr):,} | Calibration {len(X_cal):,} | Test {len(X_te):,}")

    # ── 2. Imputer ────────────────────────────────────────────────
    print("\n[2/6] Fitting imputer…")
    imputer = SimpleImputer(strategy='median')
    X_tr_imp  = imputer.fit_transform(X_tr)
    X_cal_imp = imputer.transform(X_cal)
    X_te_imp  = imputer.transform(X_te)

    # ── 3. XGBoost with monotonic constraints ─────────────────────
    print("\n[3/6] Training XGBoost with monotonic constraints…")
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    mc  = tuple(MONOTONE[f] for f in CLEAN_FEATURES)

    xgb_base = xgb.XGBClassifier(
        n_estimators       = 600,
        max_depth          = 4,
        learning_rate      = 0.035,
        subsample          = 0.80,
        colsample_bytree   = 0.80,
        reg_lambda         = 3.0,
        reg_alpha          = 1.0,
        scale_pos_weight   = spw,
        monotone_constraints = mc,
        eval_metric        = 'auc',
        random_state       = SEED,
        n_jobs             = -1,
    )
    xgb_base.fit(
        X_tr_imp, y_tr,
        eval_set=[(X_te_imp, y_te)],
        verbose=False,
    )
    raw_probs = xgb_base.predict_proba(X_te_imp)[:, 1]
    raw_auc   = roc_auc_score(y_te, raw_probs)
    raw_recall= recall_score(y_te, (raw_probs >= 0.50).astype(int))
    print(f"      Raw XGBoost  ->  AUC {raw_auc:.4f}  |  Recall(0.5) {raw_recall*100:.1f}%")
    print(f"      Score spread ->  min {raw_probs.min():.3f}  max {raw_probs.max():.3f}")

    # ── 4. Isotonic Calibration ────────────────────────────────────
    print("\n[4/6] Fitting isotonic calibration on held-out calibration set...")
    from sklearn.isotonic import IsotonicRegression

    raw_cal_probs = xgb_base.predict_proba(X_cal_imp)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_cal_probs, y_cal)

    # Evaluate: apply calibration to test set
    cal_probs  = calibrator.predict(xgb_base.predict_proba(X_te_imp)[:, 1])
    cal_auc    = roc_auc_score(y_te, cal_probs)
    cal_recall = recall_score(y_te, (cal_probs >= 0.50).astype(int))
    brier      = brier_score_loss(y_te, cal_probs)
    print(f"      Calibrated   ->  AUC {cal_auc:.4f}  |  Recall(0.5) {cal_recall*100:.1f}%")
    print(f"      Brier Score  ->  {brier:.4f}  (0 = perfect)")
    print(f"      Score spread ->  min {cal_probs.min():.3f}  max {cal_probs.max():.3f}  "
          f"mean {cal_probs.mean():.3f}  std {cal_probs.std():.3f}")

    # ── 5. SHAP Explainer ─────────────────────────────────────────
    print("\n[5/6] Building SHAP TreeExplainer (on raw XGBoost for compatibility)…")
    explainer = shap.TreeExplainer(xgb_base)

    # ── 6. Save ───────────────────────────────────────────────────
    print("\n[6/6] Saving models to models/ …")
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_base,        'models/xgb_model.joblib')
    joblib.dump(calibrator,      'models/calibrator.joblib')
    joblib.dump(explainer,       'models/shap_explainer.joblib')
    joblib.dump(imputer,         'models/imputer.joblib')
    joblib.dump(CLEAN_FEATURES,  'models/feature_list.joblib')

    perf = {
        'tuned': {
            'roc_auc':       round(cal_auc, 4),
            'raw_auc':       round(raw_auc, 4),
            'brier_score':   round(brier, 4),
            'recall_at_0.5': round(cal_recall, 4),
        },
        'features': CLEAN_FEATURES,
        'n_features':    len(CLEAN_FEATURES),
        'removed_leaky': ['late_60_89','late_90_days','emi_delay_count',
                          'open_credit_lines','real_estate_loans','num_dependents'],
        'monotone_constraints': MONOTONE,
        'calibration':   'isotonic (CalibratedClassifierCV, cv=prefit)',
        'n_train':       len(X_tr),
        'n_calibration': len(X_cal),
        'n_test':        len(X_te),
        'default_rate':  round(float(y.mean()), 4),
    }
    with open('models/performance_report.json', 'w') as f:
        json.dump(perf, f, indent=2)

    print(f"\n  [x] xgb_model.joblib      ({len(CLEAN_FEATURES)} features, monotonic constraints)")
    print(f"  [x] calibrator.joblib     (isotonic — real probability spread)")
    print(f"  [x] shap_explainer.joblib (SHAP values for XGBoost base model)")
    print(f"  [x] imputer.joblib        (median imputation)")
    print(f"  [x] feature_list.joblib   ({CLEAN_FEATURES})")
    print(f"  [x] performance_report.json")
    print()
    print("=" * 65)
    print("  Restart uvicorn to load the new calibrated model.")
    print("=" * 65)
    print()


if __name__ == '__main__':
    train()
