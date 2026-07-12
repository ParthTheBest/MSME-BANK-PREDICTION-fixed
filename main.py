"""
IDBI MSME Risk Intelligence — FastAPI Backend
===============================================
Real XGBoost model trained on Give Me Some Credit (150k borrowers)
with MSME behavioral feature overlays.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import pandas as pd
import numpy as np
import joblib
import os
import io
import json

try:
    from dotenv import load_dotenv
    load_dotenv()  # load ANTHROPIC_API_KEY (and any other vars) from a local .env file
except ImportError:
    pass

import nlp_features  # unstructured-data (officer notes) → note_stress_index


import sqlite3
import asyncio
import datetime
from fastapi import WebSocket, WebSocketDisconnect

import os
if os.environ.get("VERCEL"):
    DB_PATH = "/tmp/msme_risk.db"
else:
    DB_PATH = "msme_risk.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS borrowers (
        company_id TEXT PRIMARY KEY,
        sector TEXT,
        loan_type TEXT,
        outstanding_loan REAL,
        officer_notes TEXT,
        journey_events TEXT
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS online_features (
        company_id TEXT PRIMARY KEY,
        revolving_utilization REAL,
        debt_ratio REAL,
        income_stability REAL,
        gst_compliance_score REAL,
        cashflow_stress_ratio REAL,
        working_capital_usage REAL,
        revenue_trend_index REAL,
        payment_history_score REAL,
        supplier_payment_risk REAL,
        note_stress_index REAL,
        late_30_59 INTEGER,
        late_60_89 INTEGER,
        late_90_days INTEGER,
        open_credit_lines INTEGER,
        real_estate_loans INTEGER,
        num_dependents INTEGER,
        emi_delay_count INTEGER,
        risk_probability REAL,
        risk_band TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(company_id) REFERENCES borrowers(company_id) ON DELETE CASCADE
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id TEXT,
        event_type TEXT,
        payload TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()

def load_feature_df() -> pd.DataFrame:
    conn = get_db_connection()
    query = """
    SELECT b.company_id, b.sector, b.loan_type, b.outstanding_loan, b.officer_notes, b.journey_events,
           f.revolving_utilization, f.debt_ratio, f.income_stability, f.gst_compliance_score,
           f.cashflow_stress_ratio, f.working_capital_usage, f.revenue_trend_index, f.payment_history_score,
           f.supplier_payment_risk, f.note_stress_index, f.late_30_59, f.late_60_89, f.late_90_days,
           f.open_credit_lines, f.real_estate_loans, f.num_dependents, f.emi_delay_count,
           f.risk_probability, f.risk_band
    FROM borrowers b
    JOIN online_features f ON b.company_id = f.company_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()
event_stream_queue = asyncio.Queue()

async def streaming_consumer_worker():
    print("Starting background stream processor worker...")
    while True:
        try:
            event = await event_stream_queue.get()
            company_id = event.get("company_id")
            event_type = event.get("event_type")
            payload = event.get("payload", {})
            
            print(f"Stream Processor: Ingesting event {event_type} for borrower {company_id}")
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO events_log (company_id, event_type, payload)
            VALUES (?, ?, ?)
            """, (company_id, event_type, json.dumps(payload)))
            conn.commit()
            
            cursor.execute("SELECT * FROM borrowers WHERE company_id = ?", (company_id,))
            borrower = cursor.fetchone()
            if not borrower:
                print(f"Stream Processor Error: Borrower {company_id} not found in database.")
                conn.close()
                event_stream_queue.task_done()
                continue
                
            cursor.execute("SELECT * FROM online_features WHERE company_id = ?", (company_id,))
            features_row = cursor.fetchone()
            if not features_row:
                print(f"Stream Processor Error: Features for {company_id} not found.")
                conn.close()
                event_stream_queue.task_done()
                continue
            features = dict(features_row)
            
            updated_fields = {}
            journey_events = json.loads(borrower["journey_events"] or "[]")
            now_str = datetime.datetime.now().strftime("%Y-%m-%d")
            
            if event_type == "transaction":
                util = payload.get("revolving_utilization")
                cf = payload.get("cashflow_stress_ratio")
                if util is not None:
                    updated_fields["revolving_utilization"] = float(np.clip(util, 0.01, 0.99))
                if cf is not None:
                    updated_fields["cashflow_stress_ratio"] = float(np.clip(cf, 0.0, 5.0))
                if util is not None and util > 0.85:
                    journey_events.append({
                        "date": now_str,
                        "type": "warning",
                        "desc": f"Critical credit utilization spike: {util*100:.0f}%"
                    })
                    
            elif event_type == "gst_filing":
                score = payload.get("gst_compliance_score")
                if score is not None:
                    updated_fields["gst_compliance_score"] = float(np.clip(score, 0.0, 1.0))
                status = "on-time" if score > 0.8 else "delayed"
                journey_events.append({
                    "date": now_str,
                    "type": "ok" if score > 0.8 else "warning",
                    "desc": f"GST compliance update: {status} filing (Score: {score:.2f})"
                })
                
            elif event_type == "emi_payment":
                emi_delay = payload.get("emi_delay_count")
                pay_score = payload.get("payment_history_score")
                status = payload.get("status", "paid")
                if emi_delay is not None:
                    updated_fields["emi_delay_count"] = int(emi_delay)
                if pay_score is not None:
                    updated_fields["payment_history_score"] = float(np.clip(pay_score, 0.0, 1.0))
                if status == "delayed":
                    journey_events.append({
                        "date": now_str,
                        "type": "alert",
                        "desc": f"EMI payment delayed (Total Delays: {emi_delay})"
                    })
                else:
                    journey_events.append({
                        "date": now_str,
                        "type": "ok",
                        "desc": f"EMI installment paid successfully"
                    })
                    
            elif event_type == "officer_note":
                note = payload.get("officer_notes")
                if note:
                    cursor.execute("UPDATE borrowers SET officer_notes = ? WHERE company_id = ?", (note, company_id))
                    from nlp_features import stress_index
                    note_stress = float(stress_index([note])[0])
                    updated_fields["note_stress_index"] = note_stress
                    journey_events.append({
                        "date": now_str,
                        "type": "info",
                        "desc": f"New credit officer review note added (NLP stress score: {note_stress:.2f})"
                    })
                    
            if updated_fields:
                for k, v in updated_fields.items():
                    features[k] = v
                
                model_input = {f: features[f] for f in FEATURES}
                new_pd = predict_pd(model_input)
                new_band = get_risk_band(new_pd)
                
                old_band = features["risk_band"]
                
                set_clause = ", ".join([f"{k} = ?" for k in updated_fields.keys()])
                params = list(updated_fields.values())
                
                cursor.execute(f"""
                UPDATE online_features
                SET {set_clause}, risk_probability = ?, risk_band = ?, updated_at = CURRENT_TIMESTAMP
                WHERE company_id = ?
                """, params + [new_pd, new_band, company_id])
                
                cursor.execute("""
                UPDATE borrowers
                SET journey_events = ?
                WHERE company_id = ?
                """, (json.dumps(journey_events), company_id))
                
                conn.commit()
                
                broadcast_msg = {
                    "company_id": company_id,
                    "event_type": event_type,
                    "old_band": old_band,
                    "new_pd": new_pd,
                    "new_band": new_band,
                    "timestamp": now_str,
                    "alert": f"Borrower {company_id} risk updated to {new_band} (PD: {new_pd*100:.1f}%)"
                }
                await manager.broadcast(broadcast_msg)
                print(f"Broadcasted risk update for {company_id}: {new_band}")
                
            conn.close()
            event_stream_queue.task_done()
        except asyncio.CancelledError:
            print("Background stream processor worker cancelled.")
            break
        except Exception as e:
            print(f"Stream Processor Loop Error: {e}")
            await asyncio.sleep(1.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_assets()
    worker_task = asyncio.create_task(streaming_consumer_worker())
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="MSME Risk Intelligence API", lifespan=lifespan)

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response: Response = await call_next(request)
    if request.url.path.startswith("/dashboard") or request.url.path == "/dashboard":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

app.mount("/dashboard", StaticFiles(directory="static", html=True), name="static")
# Model-evaluation charts (ROC/PR curves, confusion matrices, dashboards) —
# the Model Performance module loads these from /reports.
app.mount("/reports", StaticFiles(directory="evaluation_report"), name="reports")

# ─── Global State ────────────────────────────────────────────
xgb_model   = None
explainer   = None
imputer     = None
calibrator  = None   # Isotonic calibration layer (loaded alongside xgb_model)
FEATURES    = None

IMPUTER_MEDIANS = {
    'revolving_utilization': 0.311059724762653,
    'debt_ratio': 0.3434650111542832,
    'income_stability': 0.6139865720483135,
    'gst_compliance_score': 0.6859868453969648,
    'cashflow_stress_ratio': 0.48113627037110945,
    'working_capital_usage': 0.3105415536187712,
    'revenue_trend_index': 1.0028420385171652,
    'payment_history_score': 0.7362094496836964,
    'supplier_payment_risk': 0.0,
    'note_stress_index': 0.2662945171272947
}

# ─── Sector LGD (Loss Given Default) ────────────────────────
LGD_MAP = {
    'Retail':        0.35,
    'Manufacturing': 0.45,
    'Construction':  0.55,
    'Technology':    0.30,
    'Agriculture':   0.40,
    'Textile':       0.42,
}

# ─── Action Escalation Ladder ────────────────────────────────
ACTION_LADDER = {
    'Low': [
        {"priority": "routine", "action": "Continue Standard Monitoring",
         "detail": "No immediate action required. Review at next scheduled cycle."}
    ],
    'Medium': [
        {"priority": "medium", "action": "Schedule Relationship Manager Call",
         "detail": "Discuss business health with borrower within 7 days."},
        {"priority": "medium", "action": "Request Updated Financial Statements",
         "detail": "Obtain last 3 months bank statements and GST returns."}
    ],
    'High': [
        {"priority": "high", "action": "Initiate Document Review",
         "detail": "Pull all collateral documents and verify current valuation."},
        {"priority": "high", "action": "Cashflow Assessment",
         "detail": "Conduct detailed analysis of inflows vs outflows for past 6 months."},
        {"priority": "high", "action": "Collateral Verification",
         "detail": "Dispatch field officer to verify primary collateral on record."}
    ],
    'Critical': [
        {"priority": "critical", "action": "Initiate Restructuring Discussion",
         "detail": "Escalate to Senior RM for restructuring evaluation immediately."},
        {"priority": "critical", "action": "Evaluate Tenure Extension",
         "detail": "Assess feasibility of extending loan tenure to reduce immediate burden."},
        {"priority": "critical", "action": "Assign to Recovery Monitoring",
         "detail": "Flag in Recovery system and assign dedicated recovery officer."},
        {"priority": "critical", "action": "Prepare Legal Documentation",
         "detail": "Engage legal team to prepare NPA declaration and recovery notices."}
    ]
}

# ─── SHAP → Plain English Narratives ─────────────────────────
FEATURE_NARRATIVE = {
    'revolving_utilization':  lambda v: f"Working capital line utilized at {v*100:.0f}% — {'dangerously high' if v > 0.7 else 'elevated' if v > 0.4 else 'healthy'}.",
    'debt_ratio':             lambda v: f"Debt-to-income ratio is {v:.2f} — {'critical' if v > 1.0 else 'high' if v > 0.5 else 'manageable'}.",
    'late_30_59':             lambda v: f"Borrower had {int(v)} payment(s) 30-59 days late in the observation window.",
    'late_60_89':             lambda v: f"Borrower had {int(v)} payment(s) 60-89 days overdue — significant stress indicator.",
    'late_90_days':           lambda v: f"Borrower had {int(v)} serious delinquency event(s) (90+ days late) — NPA risk.",
    'open_credit_lines':      lambda v: f"Borrower holds {int(v)} open credit lines — {'over-leveraged' if v > 15 else 'normal'}.",
    'real_estate_loans':      lambda v: f"Borrower has {int(v)} real estate loan(s) as collateral exposure.",
    'num_dependents':         lambda v: f"Borrower supports {int(v)} dependent(s) — impacts disposable income.",
    'income_stability':       lambda v: f"Income stability index: {v:.2f} — {'strong' if v > 0.6 else 'moderate' if v > 0.3 else 'weak'}.",
    'gst_compliance_score':   lambda v: f"GST compliance score: {v:.2f}/1.0 — {'regular filer' if v > 0.8 else 'irregular filings detected' if v > 0.5 else 'non-compliant, high risk'}.",
    'emi_delay_count':        lambda v: f"EMI delays recorded: {int(v)} — {'no delays' if v == 0 else 'payment stress visible'}.",
    'cashflow_stress_ratio':  lambda v: f"Cashflow stress index: {v:.2f} — {'severe' if v > 2.0 else 'moderate' if v > 1.0 else 'low'}.",
    'working_capital_usage':  lambda v: f"Working capital drawn: {v*100:.0f}% of sanctioned limit.",
    'revenue_trend_index':    lambda v: f"Revenue trend index: {v:.2f} — {'growing' if v > 1.1 else 'stable' if v > 0.9 else 'declining'}.",
    'payment_history_score':  lambda v: f"Payment history score: {v:.2f}/1.0 — {'excellent' if v > 0.9 else 'fair' if v > 0.7 else 'poor track record'}.",
    'supplier_payment_risk':  lambda v: f"Supplier payment risk flag: {'Active — delays to creditors detected' if v > 0 else 'Clear — no creditor delays'}.",
    'note_stress_index':      lambda v: f"Officer-notes stress index (NLP): {v:.2f}/1.0 — {'alarming language in notes' if v > 0.66 else 'some concern in notes' if v > 0.4 else 'notes read healthy'}.",
}

# ─── Human-readable feature labels + value formatting (for explainability) ───
FEATURE_LABEL = {
    'revolving_utilization': 'Credit-Line Utilisation',
    'debt_ratio':            'Debt-to-Income Ratio',
    'income_stability':      'Income Stability',
    'gst_compliance_score':  'GST Compliance',
    'cashflow_stress_ratio': 'Cashflow Stress',
    'working_capital_usage': 'Working-Capital Usage',
    'revenue_trend_index':   'Revenue Trend',
    'payment_history_score': 'Payment History',
    'supplier_payment_risk': 'Supplier-Payment Risk',
    'note_stress_index':     'Officer-Notes Stress (NLP)',
    'late_30_59':            'Payments 30-59d Late',
    'late_60_89':            'Payments 60-89d Late',
    'late_90_days':          'Serious Delinquencies (90d+)',
    'open_credit_lines':     'Open Credit Lines',
    'real_estate_loans':     'Real-Estate Loans',
    'num_dependents':        'Dependents',
    'emi_delay_count':       'EMI Delays',
}

# How to render each feature's raw value in plain language.
def _format_feature_value(feature: str, v: float) -> str:
    pct_feats = {'revolving_utilization', 'working_capital_usage'}
    score_feats = {'income_stability', 'gst_compliance_score',
                   'payment_history_score', 'note_stress_index'}
    if feature in pct_feats:
        return f"{v*100:.0f}%"
    if feature in score_feats:
        return f"{v:.2f}/1.0"
    if feature == 'supplier_payment_risk':
        return "Flagged" if v > 0 else "Clear"
    if feature in ('revenue_trend_index', 'cashflow_stress_ratio', 'debt_ratio'):
        return f"{v:.2f}"
    return f"{v:.2f}"


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def get_risk_band(pd_value: float) -> str:
    if pd_value < 0.20: return "Low"
    if pd_value < 0.50: return "Medium"
    if pd_value < 0.75: return "High"
    return "Critical"

SECTORS    = ['Retail', 'Manufacturing', 'Construction', 'Technology', 'Agriculture', 'Textile']
LOAN_TYPES = ['Term Loan', 'Working Capital', 'Trade Credit']
SECTOR_RISK_BIAS = {'Retail': 0.1, 'Manufacturing': 0.15, 'Construction': 0.25,
                    'Technology': 0.08, 'Agriculture': 0.18, 'Textile': 0.20}


_STRESSED_NOTES = [
    "Severe cashflow pressure this quarter; EMI repeatedly delayed and supplier dues mounting.",
    "Revenue declining for months; borrower struggling to meet obligations, recovery concerns.",
    "Account inflows weak and overdrafts frequent; serious liquidity stress observed.",
    "Multiple missed payments and falling GST turnover; business health deteriorating rapidly.",
]
_NEUTRAL_NOTES = [
    "Operations broadly normal this period with minor fluctuations in sales.",
    "Some seasonal dip in turnover but repayments largely on schedule.",
    "Cashflow adequate; one delayed payment noted but otherwise stable.",
]
_HEALTHY_NOTES = [
    "Strong sales growth and healthy margins; all EMIs paid on time, GST filed promptly.",
    "Comfortable liquidity and rising deposits; well-managed, low-risk borrower.",
    "Stable, profitable operations; no concerns, excellent payment discipline.",
]

def _pick_note(rng: np.random.RandomState, risk_proxy: float) -> str:
    """Choose an officer note whose tone reflects the borrower's risk (with noise)."""
    r = rng.random()
    if risk_proxy >= 0.66:
        pool = _STRESSED_NOTES if r < 0.7 else _NEUTRAL_NOTES
    elif risk_proxy >= 0.33:
        pool = _NEUTRAL_NOTES if r < 0.6 else (_STRESSED_NOTES if r < 0.8 else _HEALTHY_NOTES)
    else:
        pool = _HEALTHY_NOTES if r < 0.7 else _NEUTRAL_NOTES
    return pool[rng.randint(len(pool))]


def _make_borrower_row(company_id: str, rng: np.random.RandomState, year: int = 2024) -> dict:
    """Generate one realistic MSME borrower in the model's feature space using `rng`."""
    sector = rng.choice(SECTORS)
    loan_type = rng.choice(LOAN_TYPES)
    bias = SECTOR_RISK_BIAS[sector]

    # Inject distressed borrower logic to ensure the model sees high-risk cases
    # that match the distribution it was trained on.
    is_distressed = rng.random() < (0.15 + bias * 0.2)
    d = 1 if is_distressed else 0
    noise = lambda s: rng.normal(0, s)

    # 10 Leading Indicators (matches retrain_calibrated.py exactly)
    rev_util      = float(np.clip(rng.beta(2, 5) + d * 0.38 + noise(0.05), 0.01, 0.99))
    debt_ratio    = float(np.clip(rng.exponential(0.35) + d * 0.65 + noise(0.06), 0.01, 3.0))
    inc_stability = float(np.clip(rng.beta(4, 2) * (1 - d * 0.38) + noise(0.04), 0.0, 1.0))
    gst_score     = float(np.clip(rng.beta(5, 2) * (1 - d * 0.50) + noise(0.04), 0.0, 1.0))
    cf_stress     = float(np.clip(rng.exponential(0.45) + d * 0.85 + noise(0.10), 0.0, 5.0))
    wc_usage      = float(np.clip(rng.beta(2, 5) + d * 0.33 + noise(0.04), 0.0, 1.0))
    rev_trend     = float(np.clip(rng.normal(1.05, 0.18) - d * 0.38 + noise(0.05), 0.2, 2.0))
    pay_hist      = float(np.clip(rng.beta(6, 2) * (1 - d * 0.42) + noise(0.04), 0.0, 1.0))
    supp_risk     = float(1.0 if rng.random() < (0.08 + d * 0.42) else 0.0)

    # Legacy variables (kept for timeline/UI compatibility, but not used by model)
    late_3059     = int(rng.poisson(bias * 3 + d * 5))
    late_6089     = int(rng.poisson(bias * 1.5 + d * 3))
    late_90       = int(rng.poisson(bias * 0.8 + d * 2))
    open_lines    = int(rng.poisson(8) + 2)
    re_loans      = int(rng.poisson(1))
    num_dep       = int(rng.poisson(1.5))
    emi_delays    = min(late_3059 + late_6089, 12)
    outstanding   = float(rng.uniform(200000, 2500000))

    journey_events = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for m_idx, month in enumerate(months):
        gst_ok = rng.random() > (bias * 0.6 + m_idx * 0.02)
        emi_ok = rng.random() > (bias * 0.5 + m_idx * 0.03)
        if not gst_ok:
            journey_events.append({"date": f"{year}-{m_idx+1:02d}-07", "type": "warning",
                                    "desc": f"GST filing delayed — {month} {year}"})
        if not emi_ok:
            journey_events.append({"date": f"{year}-{m_idx+1:02d}-15", "type": "alert",
                                    "desc": f"EMI payment overdue — {month} {year}"})
        elif m_idx == 0:
            journey_events.append({"date": f"{year}-{m_idx+1:02d}-15", "type": "ok",
                                    "desc": f"EMI paid on time — {month} {year}"})

    # Risk proxy → pick an officer note (unstructured signal); note_stress_index is
    # scored in batch by the caller (build_portfolio / generate_new_loans).
    # We heavily weight the distressed flag 'd' to ensure the NLP model extracts a high stress index
    risk_proxy = float(np.clip(
        0.35 * rev_util + 0.30 * min(emi_delays / 4, 1) + 0.25 * min(late_90, 2) / 2
        + 0.10 * (1 - pay_hist) + d * 0.45, 0, 1))
    officer_note = _pick_note(rng, risk_proxy)

    return {
        'company_id': company_id, 'sector': sector, 'loan_type': loan_type,
        'outstanding_loan': outstanding, 'officer_notes': officer_note,
        'revolving_utilization': rev_util, 'debt_ratio': debt_ratio,
        'late_30_59': late_3059, 'late_60_89': late_6089, 'late_90_days': late_90,
        'open_credit_lines': open_lines, 'real_estate_loans': re_loans,
        'num_dependents': num_dep, 'income_stability': inc_stability,
        'gst_compliance_score': gst_score, 'emi_delay_count': emi_delays,
        'cashflow_stress_ratio': cf_stress, 'working_capital_usage': wc_usage,
        'revenue_trend_index': rev_trend, 'payment_history_score': pay_hist,
        'supplier_payment_risk': supp_risk,
        'journey_events': json.dumps(journey_events),
    }


def build_portfolio() -> pd.DataFrame:
    """Build the deterministic 200-borrower demo portfolio."""
    rng = np.random.RandomState(2024)
    rows = [_make_borrower_row(f"MSME-{i+1:04d}", rng) for i in range(200)]
    df = pd.DataFrame(rows)
    df['note_stress_index'] = nlp_features.stress_index(df['officer_notes'])
    return df


# Monotonic counter so freshly-disbursed loans always get unique IDs across refreshes.
_new_loan_seq = 0

def generate_new_loans(count: int) -> pd.DataFrame:
    """Simulate `count` companies taking new loans (non-deterministic each call)."""
    global _new_loan_seq
    rng = np.random.RandomState()  # fresh entropy → genuinely new borrowers each refresh
    rows = []
    for _ in range(count):
        _new_loan_seq += 1
        rows.append(_make_borrower_row(f"MSME-N{_new_loan_seq:04d}", rng, year=2025))
    df = pd.DataFrame(rows)
    df['note_stress_index'] = nlp_features.stress_index(df['officer_notes'])
    return df

# ─── Upload: map an arbitrary CSV into the model's feature space ──
# Accepts two shapes:
#   (a) "model schema"  — rows already carry the 16 model FEATURES
#   (b) "blueprint MSME schema" — monthly rows with columns like
#       company_id, month, sector, outstanding_loan, monthly_sales,
#       gst_turnover, monthly_inflow, monthly_outflow, emi_delay_count,
#       credit_utilization, working_capital_usage, account_balance, default
#   Monthly data (a `month` column) is aggregated to the latest snapshot
#   per company and derived into the behavioural feature space.
DEFAULT_SECTOR = 'Retail'

def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Clean string currency formats, commas, percentages, and parse safely as floats."""
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    
    # Convert to string and clean
    s_str = s.astype(str).str.strip()
    
    # Strip currency symbols, commas, spaces
    s_cleaned = s_str.str.replace(r'[\$\u20B9\u20A8\s,]', '', regex=True)
    
    # Detect percentage formatting
    pct_mask = s_cleaned.str.endswith('%', na=False)
    s_cleaned = s_cleaned.str.rstrip('%')
    
    # Safely convert to numeric float
    numeric_s = pd.to_numeric(s_cleaned, errors='coerce')
    
    # Standardize percent values (e.g. 45% -> 0.45)
    if pct_mask.any():
        numeric_s = numeric_s.mask(pct_mask, numeric_s / 100.0)
        
    return numeric_s

def map_upload_to_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Collapse monthly journeys → one row per company (latest month)
    if 'month' in df.columns and 'company_id' in df.columns:
        df = _aggregate_monthly(df)

    # Meta columns
    if 'company_id' not in df.columns:
        df['company_id'] = [f"MSME-U{i+1:04d}" for i in range(len(df))]
    if 'sector' not in df.columns:
        df['sector'] = DEFAULT_SECTOR
    if 'loan_type' not in df.columns:
        df['loan_type'] = 'Term Loan'
        
    # Sanitize outstanding loan
    if 'outstanding_loan' in df.columns:
        df['outstanding_loan'] = clean_numeric_series(df['outstanding_loan']).fillna(1_000_000.0).clip(0.0, 1e12)
    else:
        df['outstanding_loan'] = 1_000_000.0

    # Sanitize and bound features that are already present in df
    for f in FEATURES:
        if f in df.columns:
            df[f] = clean_numeric_series(df[f])
            # Apply domain-specific clips to prevent overflow or negative bypass:
            if f in ('revolving_utilization', 'working_capital_usage', 'income_stability', 'gst_compliance_score', 'payment_history_score'):
                df[f] = df[f].clip(0.0, 1.0)
            elif f == 'debt_ratio':
                df[f] = df[f].clip(0.0, 100.0)
            elif f == 'cashflow_stress_ratio':
                df[f] = df[f].clip(0.0, 10.0)
            elif f == 'revenue_trend_index':
                df[f] = df[f].clip(0.0, 10.0)
            elif f == 'supplier_payment_risk':
                df[f] = df[f].clip(0.0, 1.0)

    # If the file already carries the model features, keep them; otherwise derive.
    have = set(df.columns)
    missing = [f for f in FEATURES if (f not in have and f != 'note_stress_index')]
    if missing:
        df = _derive_features(df)

    # Unstructured data: score officer notes → note_stress_index (neutral if absent)
    if 'note_stress_index' not in df.columns:
        if 'officer_notes' in df.columns:
            df['note_stress_index'] = nlp_features.stress_index(df['officer_notes'])
        else:
            df['note_stress_index'] = 0.5

    # Final guarantee: every model feature exists, is numeric, and bounds are enforced
    for f in FEATURES:
        if f not in df.columns:
            df[f] = np.nan
        df[f] = pd.to_numeric(df[f], errors='coerce')
        if f in ('revolving_utilization', 'working_capital_usage', 'income_stability', 'gst_compliance_score', 'payment_history_score'):
            df[f] = df[f].clip(0.0, 1.0)
        elif f == 'debt_ratio':
            df[f] = df[f].clip(0.0, 100.0)
        elif f == 'cashflow_stress_ratio':
            df[f] = df[f].clip(0.0, 10.0)
        elif f == 'revenue_trend_index':
            df[f] = df[f].clip(0.0, 10.0)
        elif f == 'supplier_payment_risk':
            df[f] = df[f].clip(0.0, 1.0)

    if 'journey_events' not in df.columns:
        df['journey_events'] = '[]'
    if 'officer_notes' not in df.columns:
        df['officer_notes'] = ''

    keep = ['company_id', 'sector', 'loan_type', 'outstanding_loan',
            'journey_events', 'officer_notes'] + FEATURES
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def _aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce blueprint monthly rows to one enriched snapshot per company."""
    rows = []
    for cid, grp in df.groupby('company_id'):
        grp = grp.sort_values('month')
        first, last = grp.iloc[0], grp.iloc[-1]
        rec = last.to_dict()

        def safe(a, b):
            return float(a) / float(b) if b not in (0, None) and not pd.isna(b) else 0.0

        # Trends across the observed window
        if 'monthly_sales' in grp:
            rec['revenue_trend_index'] = np.clip(
                safe(last.get('monthly_sales', 1), first.get('monthly_sales', 1)), 0.2, 2.0)
        if {'monthly_inflow', 'monthly_outflow'}.issubset(grp.columns):
            net = (grp['monthly_inflow'] - grp['monthly_outflow'])
            rec['_cashflow_vol'] = float(net.std()) if len(net) > 1 else 0.0
            rec['cashflow_stress_ratio'] = np.clip(
                safe(last['monthly_outflow'], last['monthly_inflow']) * 2.0, 0, 5)
        rows.append(rec)
    return pd.DataFrame(rows)


def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map blueprint MSME columns → the 16 model features (mirrors training overlay)."""
    n = len(df)
    g = df.get
    out = df.copy()

    # Pre-calculate base indicators if we need them for missing values
    util = pd.to_numeric(g('credit_utilization', pd.Series([0.4] * n)), errors='coerce').fillna(0.4)
    wc   = pd.to_numeric(g('working_capital_usage', pd.Series([0.5] * n)), errors='coerce').fillna(0.5)
    emi  = pd.to_numeric(g('emi_delay_count', pd.Series([0] * n)), errors='coerce').fillna(0)

    if {'monthly_inflow', 'monthly_outflow'}.issubset(df.columns):
        inflow  = pd.to_numeric(df['monthly_inflow'], errors='coerce').replace(0, np.nan)
        outflow = pd.to_numeric(df['monthly_outflow'], errors='coerce')
        debt = np.clip((outflow / inflow).fillna(0.5), 0.01, 3.0)
    else:
        debt = pd.Series(np.clip(util * 1.2, 0.01, 3.0), index=df.index)

    # 1. revolving_utilization
    if 'revolving_utilization' not in out.columns or out['revolving_utilization'].isna().all():
        out['revolving_utilization'] = np.clip(util, 0.01, 0.99)

    # 2. debt_ratio
    if 'debt_ratio' not in out.columns or out['debt_ratio'].isna().all():
        out['debt_ratio'] = debt

    # 3. late_30_59
    if 'late_30_59' not in out.columns or out['late_30_59'].isna().all():
        out['late_30_59'] = np.clip(emi, 0, 12).astype(int)

    # 4. late_60_89
    if 'late_60_89' not in out.columns or out['late_60_89'].isna().all():
        out['late_60_89'] = np.clip(emi - 1, 0, 12).astype(int)

    # 5. late_90_days
    if 'late_90_days' not in out.columns or out['late_90_days'].isna().all():
        out['late_90_days'] = np.clip(emi - 2, 0, 12).astype(int)

    # 6. open_credit_lines
    if 'open_credit_lines' not in out.columns or out['open_credit_lines'].isna().all():
        out['open_credit_lines'] = pd.to_numeric(g('open_credit_lines', pd.Series([8] * n)), errors='coerce').fillna(8)

    # 7. real_estate_loans
    if 'real_estate_loans' not in out.columns or out['real_estate_loans'].isna().all():
        out['real_estate_loans'] = pd.to_numeric(g('real_estate_loans', pd.Series([1] * n)), errors='coerce').fillna(1)

    # 8. num_dependents
    if 'num_dependents' not in out.columns or out['num_dependents'].isna().all():
        out['num_dependents'] = pd.to_numeric(g('num_dependents', pd.Series([1] * n)), errors='coerce').fillna(1)

    # 9. income_stability
    if 'income_stability' not in out.columns or out['income_stability'].isna().all():
        if 'account_balance' in df.columns:
            bal = pd.to_numeric(df['account_balance'], errors='coerce').fillna(0)
            out['income_stability'] = np.clip(bal / 500000, 0.0, 1.0)
        else:
            out['income_stability'] = 0.5

    # 10. gst_compliance_score
    if 'gst_compliance_score' not in out.columns or out['gst_compliance_score'].isna().all():
        out['gst_compliance_score'] = np.clip(1 - emi * 0.12, 0.0, 1.0)

    # 11. emi_delay_count
    if 'emi_delay_count' not in out.columns or out['emi_delay_count'].isna().all():
        out['emi_delay_count'] = np.clip(emi, 0, 12).astype(int)

    # 12. cashflow_stress_ratio
    if 'cashflow_stress_ratio' not in out.columns or out['cashflow_stress_ratio'].isna().all():
        out['cashflow_stress_ratio'] = np.clip(debt * 1.0, 0, 5)

    # 13. working_capital_usage
    if 'working_capital_usage' not in out.columns or out['working_capital_usage'].isna().all():
        out['working_capital_usage'] = np.clip(wc, 0.0, 1.0)

    # 14. revenue_trend_index
    if 'revenue_trend_index' not in out.columns or out['revenue_trend_index'].isna().all():
        out['revenue_trend_index'] = np.clip(1.2 - debt * 0.4, 0.2, 2.0)

    # 15. payment_history_score
    if 'payment_history_score' not in out.columns or out['payment_history_score'].isna().all():
        out['payment_history_score'] = np.clip(1 - emi * 0.08, 0.0, 1.0)

    # 16. supplier_payment_risk
    if 'supplier_payment_risk' not in out.columns or out['supplier_payment_risk'].isna().all():
        out['supplier_payment_risk'] = (emi > 2).astype(float)

    return out


def seed_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM borrowers")
    count = cursor.fetchone()[0]
    if count == 0:
        print("Seeding database with default MSME borrowers...")
        df = build_portfolio()
        probs = predict_portfolio(df)
        df['risk_probability'] = probs
        df['risk_band'] = df['risk_probability'].apply(get_risk_band)
        
        for _, row in df.iterrows():
            cursor.execute("""
            INSERT OR REPLACE INTO borrowers (company_id, sector, loan_type, outstanding_loan, officer_notes, journey_events)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row['company_id'], row['sector'], row['loan_type'], float(row['outstanding_loan']),
                row['officer_notes'], row['journey_events']
            ))
            cursor.execute("""
            INSERT OR REPLACE INTO online_features (
                company_id, revolving_utilization, debt_ratio, income_stability, gst_compliance_score,
                cashflow_stress_ratio, working_capital_usage, revenue_trend_index, payment_history_score,
                supplier_payment_risk, note_stress_index, late_30_59, late_60_89, late_90_days,
                open_credit_lines, real_estate_loans, num_dependents, emi_delay_count, risk_probability, risk_band
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['company_id'], float(row['revolving_utilization']), float(row['debt_ratio']),
                float(row['income_stability']), float(row['gst_compliance_score']), float(row['cashflow_stress_ratio']),
                float(row['working_capital_usage']), float(row['revenue_trend_index']), float(row['payment_history_score']),
                float(row['supplier_payment_risk']), float(row['note_stress_index']),
                int(row['late_30_59']), int(row['late_60_89']), int(row['late_90_days']),
                int(row['open_credit_lines']), int(row['real_estate_loans']), int(row['num_dependents']),
                int(row['emi_delay_count']), float(row['risk_probability']), row['risk_band']
            ))
        conn.commit()
    conn.close()

def load_assets():
    global xgb_model, explainer, imputer, calibrator, FEATURES
    try:
        FEATURES  = joblib.load('models/feature_list.joblib')
        try:
            import xgboost as xgb
            xgb_model = xgb.Booster()
            xgb_model.load_model('models/xgb_model.json')
            print("Native XGBoost booster loaded successfully.")
        except Exception as e:
            try:
                xgb_model = joblib.load('models/xgb_model.joblib')
                print("XGBClassifier model loaded via joblib.")
            except Exception as ex:
                print(f"Failed to load XGBoost model: {ex}")
                raise
        try:
            with open('models/calibrator.json', 'r') as f:
                cal_data = json.load(f)
                calibrator = {
                    'xp': np.array(cal_data['xp']),
                    'fp': np.array(cal_data['fp'])
                }
            print("Native Isotonic calibrator loaded.")
        except Exception:
            try:
                calibrator = joblib.load('models/calibrator.joblib')
                print("Isotonic calibrator loaded via joblib.")
            except Exception:
                calibrator = None
                print("No calibrator found — using raw XGBoost probabilities.")
        try:
            imputer = joblib.load('models/imputer.joblib')
            print("Scikit-learn imputer loaded.")
        except Exception:
            imputer = None
            print("Using custom manual imputer fallback.")
        try:
            explainer = joblib.load('models/shap_explainer.joblib')
            print("SHAP explainer loaded.")
        except Exception:
            explainer = None
            print("Using built-in XGBoost C++ SHAP explainer fallback.")
        init_db()
        seed_db()
        print("Model and database loaded successfully.")
    except Exception as e:
        print(f"Failed to load models: {e}")
        raise

def predict_portfolio(df: pd.DataFrame) -> np.ndarray:
    """Score borrowers through XGBoost + isotonic calibration."""
    X = df[FEATURES].copy()
    if imputer is not None:
        X_imp = imputer.transform(X)
    else:
        X_imp_df = X.copy()
        for f in FEATURES:
            X_imp_df[f] = X_imp_df[f].fillna(IMPUTER_MEDIANS[f])
        X_imp = X_imp_df[FEATURES].values

    import xgboost as xgb
    if isinstance(xgb_model, xgb.Booster):
        dtrain = xgb.DMatrix(X_imp)
        raw_probs = xgb_model.predict(dtrain)
    else:
        raw_probs = xgb_model.predict_proba(X_imp)[:, 1]

    if calibrator is not None:
        if isinstance(calibrator, dict):
            return np.clip(np.interp(raw_probs, calibrator['xp'], calibrator['fp']), 0.0, 1.0)
        else:
            return np.clip(calibrator.predict(raw_probs), 0.0, 1.0)
    return raw_probs


def predict_pd(features: dict) -> float:
    """Calibrated PD for a single borrower (a {feature: value} dict).
    Routes through the SAME XGBoost + isotonic calibration path as
    predict_portfolio so every module reports one consistent PD."""
    df = pd.DataFrame([features])
    # Align columns to FEATURES list. Missing features will be NaN (gracefully handled by simple imputer)
    for f in FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    return float(predict_portfolio(df)[0])

# ─── API Routes ───────────────────────────────────────────────

@app.get("/")
def root():
    return RedirectResponse(url="/dashboard/index.html")

@app.get("/model-performance")
def get_model_performance():
    """Return training performance metrics for dashboard display."""
    try:
        with open('models/performance_report.json') as f:
            return json.load(f)
    except Exception:
        return {}

@app.get("/portfolio")
def get_portfolio_summary():
    if FEATURES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_df = load_feature_df()
    if feature_df.empty:
        return {
            "total_borrowers": 0, "avg_pd": 0.0, "total_exposure": 0.0, "total_expected_loss": 0.0,
            "risk_distribution": {"Low":0, "Medium":0, "High":0, "Critical":0}, "sector_analytics": []
        }

    probs = feature_df['risk_probability'].values
    df = feature_df.copy()
    df['pd']    = probs
    df['band']  = df['risk_band']
    df['lgd']   = df['sector'].map(lambda s: LGD_MAP.get(s, 0.4))
    df['el']    = df['outstanding_loan'] * df['pd'] * df['lgd']

    sector_stats = []
    for sector, grp in df.groupby('sector'):
        sector_stats.append({
            "sector":         sector,
            "exposure":       float(grp['outstanding_loan'].sum()),
            "avg_pd":         float(grp['pd'].mean()),
            "expected_loss":  float(grp['el'].sum()),
            "borrower_count": int(len(grp))
        })

    dist = df['band'].value_counts().to_dict()
    for band in ["Low", "Medium", "High", "Critical"]:
        dist.setdefault(band, 0)

    return {
        "total_borrowers":    int(len(df)),
        "avg_pd":             float(probs.mean()),
        "total_exposure":     float(df['outstanding_loan'].sum()),
        "total_expected_loss": float(df['el'].sum()),
        "risk_distribution":  dist,
        "sector_analytics":   sorted(sector_stats, key=lambda x: x['expected_loss'], reverse=True)
    }

@app.get("/borrowers")
def get_borrowers(limit: int = 50):
    if FEATURES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_df = load_feature_df()
    if feature_df.empty:
        return []

    df = feature_df.copy()
    df['pd'] = df['risk_probability']
    top = df.sort_values('pd', ascending=False).head(limit)

    results = []
    for _, row in top.iterrows():
        pd_val = float(row['pd'])
        risk_score = int(round(pd_val * 100))
        lgd = LGD_MAP.get(row['sector'], 0.4)
        el  = pd_val * float(row['outstanding_loan']) * lgd
        results.append({
            "company_id":       row['company_id'],
            "sector":           row['sector'],
            "loan_type":        row['loan_type'],
            "pd":               pd_val,
            "risk_score":       risk_score,
            "outstanding_loan": float(row['outstanding_loan']),
            "expected_loss":    el,
            "risk_band":        row['risk_band'],
        })
    return results

@app.get("/borrowers/{company_id}")
def get_borrower_details(company_id: str):
    if FEATURES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_df = load_feature_df()
    row = feature_df[feature_df['company_id'] == company_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Borrower not found")
    row = row.iloc[0]

    current_pd = float(row['risk_probability'])

    timeline = []
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    base_features = row[FEATURES].to_dict()
    bias = current_pd

    for m_idx, month in enumerate(months):
        scale = (m_idx / 11)
        monthly_feats = base_features.copy()
        monthly_feats['revolving_utilization'] = float(np.clip(
            base_features['revolving_utilization'] * (0.5 + 0.5 * scale), 0.01, 0.99))
        monthly_feats['gst_compliance_score'] = float(np.clip(
            base_features['gst_compliance_score'] * (1.1 - 0.2 * scale), 0, 1))
        monthly_feats['payment_history_score'] = float(np.clip(
            base_features['payment_history_score'] * (1.05 - 0.15 * scale), 0, 1))
        monthly_feats['cashflow_stress_ratio'] = float(
            base_features['cashflow_stress_ratio'] * (0.6 + 0.6 * scale))

        m_pd = predict_pd(monthly_feats)
        timeline.append({"month": month, "pd": m_pd})

    journey_events = []
    try:
        raw = json.loads(row['journey_events'])
        seen = set()
        for ev in raw:
            key = ev['date'] + ev['desc']
            if key not in seen:
                seen.add(key)
                journey_events.append(ev)
    except Exception:
        pass

    sector  = row['sector']
    lgd     = LGD_MAP.get(sector, 0.4)
    ead     = float(row['outstanding_loan'])
    el      = current_pd * ead * lgd

    return {
        "company_id":       company_id,
        "sector":           sector,
        "loan_type":        row.get('loan_type', 'Term Loan'),
        "outstanding_loan": ead,
        "lgd_pct":          lgd,
        "expected_loss":    el,
        "potential_recovery": ead - el,
        "current_pd":       current_pd,
        "risk_band":        row['risk_band'],
        "timeline":         timeline,
        "risk_migration": {
            "start_band": get_risk_band(timeline[0]['pd']),
            "end_band":   row['risk_band']
        },
        "journey_events":  journey_events,
        "officer_notes":   str(row.get('officer_notes', '') or ''),
        "note_stress_index": float(row['note_stress_index']) if 'note_stress_index' in row else None,
        "raw_features":    {k: float(row[k]) for k in FEATURES},
        "action_ladder":   ACTION_LADDER.get(row['risk_band'], [])
    }

def _portfolio_percentile(feature: str, value: float) -> float:
    try:
        feature_df = load_feature_df()
        col = pd.to_numeric(feature_df[feature], errors='coerce').dropna()
        if len(col) == 0:
            return 0.5
        return float((col < value).mean())
    except Exception:
        return 0.5

@app.get("/borrowers/{company_id}/explain")
def get_shap_explanation(company_id: str):
    if FEATURES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_df = load_feature_df()
    row = feature_df[feature_df['company_id'] == company_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Borrower not found")

    X = row[FEATURES].copy()
    if imputer is not None:
        X_imp = imputer.transform(X)
    else:
        X_imp_df = X.copy()
        for f in FEATURES:
            X_imp_df[f] = X_imp_df[f].fillna(IMPUTER_MEDIANS[f])
        X_imp = X_imp_df[FEATURES].values

    if explainer is not None:
        shap_vals = np.asarray(explainer.shap_values(X_imp))
        base_lo = float(np.ravel(explainer.expected_value)[-1])
    else:
        import xgboost as xgb
        d = xgb.DMatrix(X_imp)
        booster = xgb_model.get_booster() if hasattr(xgb_model, 'get_booster') else xgb_model
        contribs = booster.predict(d, pred_contribs=True)
        shap_vals = contribs[:, :-1]
        base_lo = float(contribs[0, -1])

    if shap_vals.ndim == 3:
        shap_vals = shap_vals[-1]
    shap_row = shap_vals[0]

    base_prob_raw = _sigmoid(base_lo)
    if calibrator is not None:
        base_prob = float(np.clip(calibrator.predict([base_prob_raw])[0], 0.0, 1.0))
    else:
        base_prob = base_prob_raw
    current_pd = float(row.iloc[0]['risk_probability'])
    band = row.iloc[0]['risk_band']

    feat_vals = X.iloc[0].to_dict()
    total_abs = float(sum(abs(float(s)) for s in shap_row)) or 1.0

    drivers = []
    for i, f in enumerate(FEATURES):
        val = float(feat_vals[f])
        impact = float(shap_row[i])
        pct = _portfolio_percentile(f, val)
        higher_is_worse = impact >= 0
        rank_txt = (f"higher than {pct*100:.0f}% of the portfolio" if val >= 0 else "")
        drivers.append({
            "feature":          f,
            "label":            FEATURE_LABEL.get(f, f.replace('_', ' ').title()),
            "value":            val,
            "value_display":    _format_feature_value(f, val),
            "impact":           impact,
            "abs_impact":       abs(impact),
            "direction":        "increases" if impact > 0 else "decreases" if impact < 0 else "neutral",
            "contribution_pct": round(abs(impact) / total_abs * 100, 1),
            "percentile":       round(pct * 100),
            "benchmark":        rank_txt,
            "narrative":        FEATURE_NARRATIVE.get(f, lambda x: f"Value: {x:.3f}")(val),
        })

    drivers.sort(key=lambda d: d["abs_impact"], reverse=True)
    increasing = [d for d in drivers if d["impact"] > 0]
    reducing   = [d for d in drivers if d["impact"] < 0]

    def _phrase(ds, n=3):
        return "; ".join(f"{d['label'].lower()} ({d['value_display']})" for d in ds[:n]) or "no dominant factor"

    direction_word = "well above" if current_pd > base_prob + 0.05 else \
                     "below" if current_pd < base_prob - 0.05 else "close to"
    summary = (
        f"{company_id} carries a 12-month default probability of {current_pd*100:.1f}% "
        f"({band} risk) — {direction_word} the model's portfolio baseline of {base_prob*100:.1f}%. "
        f"The risk is pushed UP mainly by {_phrase(increasing)}. "
        + (f"It is partially offset by stronger {_phrase(reducing)}. "
           if reducing else "Almost no factors pull the risk down. ")
        + f"In total, {len(increasing)} factor(s) raise risk and {len(reducing)} reduce it, "
        f"with {(increasing[0]['label'] if increasing else drivers[0]['label'])} the single largest driver "
        f"({(increasing[0] if increasing else drivers[0])['contribution_pct']}% of the decision)."
    )

    return {
        "company_id":   company_id,
        "current_pd":   current_pd,
        "risk_band":    band,
        "baseline": {
            "probability": base_prob,
            "log_odds":    base_lo,
            "label":       "Model baseline (average MSME borrower)",
        },
        "summary":         summary,
        "key_drivers":     drivers,
        "risk_increasing": increasing[:5],
        "risk_reducing":   reducing[:5],
        "note_stress_index": float(row.iloc[0]['note_stress_index']) if 'note_stress_index' in row.columns else None,
        "officer_notes":   str(row.iloc[0].get('officer_notes', '') or ''),
    }

@app.post("/upload")
async def upload_portfolio(file: UploadFile = File(...)):
    """Upload a borrower CSV. Replaces the active portfolio inside SQLite and re-scores it."""
    if FEATURES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    try:
        raw = await file.read()
        df_in = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    if df_in.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        mapped = map_upload_to_portfolio(df_in)
        probs = predict_portfolio(mapped)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not map file to model features: {e}")

    mapped['risk_probability'] = probs
    mapped['risk_band'] = mapped['risk_probability'].apply(get_risk_band)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM online_features")
    cursor.execute("DELETE FROM borrowers")
    
    for _, row in mapped.iterrows():
        loan_type = row.get('loan_type', 'Term Loan')
        officer_notes = row.get('officer_notes', '')
        journey_events = row.get('journey_events', '[]')
        
        cursor.execute("""
        INSERT OR REPLACE INTO borrowers (company_id, sector, loan_type, outstanding_loan, officer_notes, journey_events)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row['company_id'], row['sector'], loan_type, float(row['outstanding_loan']),
            officer_notes, journey_events
        ))
        
        cursor.execute("""
        INSERT OR REPLACE INTO online_features (
            company_id, revolving_utilization, debt_ratio, income_stability, gst_compliance_score,
            cashflow_stress_ratio, working_capital_usage, revenue_trend_index, payment_history_score,
            supplier_payment_risk, note_stress_index, late_30_59, late_60_89, late_90_days,
            open_credit_lines, real_estate_loans, num_dependents, emi_delay_count, risk_probability, risk_band
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['company_id'], float(row['revolving_utilization']), float(row['debt_ratio']),
            float(row['income_stability']), float(row['gst_compliance_score']), float(row['cashflow_stress_ratio']),
            float(row['working_capital_usage']), float(row['revenue_trend_index']), float(row['payment_history_score']),
            float(row['supplier_payment_risk']), float(row['note_stress_index']),
            int(row.get('late_30_59', 0)), int(row.get('late_60_89', 0)), int(row.get('late_90_days', 0)),
            int(row.get('open_credit_lines', 5)), int(row.get('real_estate_loans', 0)), int(row.get('num_dependents', 0)),
            int(row.get('emi_delay_count', 0)), float(row['risk_probability']), row['risk_band']
        ))
        
    conn.commit()
    conn.close()

    return {
        "status": "ok",
        "filename": file.filename,
        "rows_ingested": int(len(df_in)),
        "borrowers_scored": int(len(mapped)),
        "avg_pd": float(probs.mean()),
        "high_risk_count": int((probs >= 0.5).sum()),
    }

@app.post("/reset")
def reset_portfolio():
    """Restore the built-in demo portfolio inside SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM online_features")
    cursor.execute("DELETE FROM borrowers")
    conn.commit()
    conn.close()
    
    seed_db()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM borrowers")
    count = cursor.fetchone()[0]
    conn.close()
    
    return {"status": "ok", "borrowers": count}

@app.post("/refresh")
def refresh_portfolio(count: int = 10):
    """Simulate new companies taking loans and fold them into the SQLite database."""
    if FEATURES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    count = max(1, min(int(count), 100))

    new_df = generate_new_loans(count)
    probs = predict_portfolio(new_df)
    new_df['risk_probability'] = probs
    new_df['risk_band'] = new_df['risk_probability'].apply(get_risk_band)

    conn = get_db_connection()
    cursor = conn.cursor()
    
    new_loans = []
    for _, row in new_df.iterrows():
        cursor.execute("""
        INSERT OR REPLACE INTO borrowers (company_id, sector, loan_type, outstanding_loan, officer_notes, journey_events)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row['company_id'], row['sector'], row['loan_type'], float(row['outstanding_loan']),
            row['officer_notes'], row['journey_events']
        ))
        cursor.execute("""
        INSERT OR REPLACE INTO online_features (
            company_id, revolving_utilization, debt_ratio, income_stability, gst_compliance_score,
            cashflow_stress_ratio, working_capital_usage, revenue_trend_index, payment_history_score,
            supplier_payment_risk, note_stress_index, late_30_59, late_60_89, late_90_days,
            open_credit_lines, real_estate_loans, num_dependents, emi_delay_count, risk_probability, risk_band
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['company_id'], float(row['revolving_utilization']), float(row['debt_ratio']),
            float(row['income_stability']), float(row['gst_compliance_score']), float(row['cashflow_stress_ratio']),
            float(row['working_capital_usage']), float(row['revenue_trend_index']), float(row['payment_history_score']),
            float(row['supplier_payment_risk']), float(row['note_stress_index']),
            int(row['late_30_59']), int(row['late_60_89']), int(row['late_90_days']),
            int(row['open_credit_lines']), int(row['real_estate_loans']), int(row['num_dependents']),
            int(row['emi_delay_count']), float(row['risk_probability']), row['risk_band']
        ))
        
        new_loans.append({
            "company_id": row['company_id'],
            "sector": row['sector'],
            "pd": float(row['risk_probability']),
            "risk_band": row['risk_band'],
            "outstanding_loan": float(row['outstanding_loan'])
        })
        
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM borrowers")
    total_borrowers = cursor.fetchone()[0]
    conn.close()
    
    new_loans = sorted(new_loans, key=lambda x: x['pd'], reverse=True)
    return {
        "status": "ok",
        "added": count,
        "total_borrowers": total_borrowers,
        "new_high_risk": int((probs >= 0.5).sum()),
        "new_avg_pd": float(probs.mean()),
        "new_loans": new_loans,
    }

def _borrower_context(company_id: str) -> dict:
    feature_df = load_feature_df()
    row = feature_df[feature_df['company_id'] == company_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Borrower not found")
    X = row[FEATURES].copy()
    if imputer is not None:
        X_imp = imputer.transform(X)
    else:
        X_imp_df = X.copy()
        for f in FEATURES:
            X_imp_df[f] = X_imp_df[f].fillna(IMPUTER_MEDIANS[f])
        X_imp = X_imp_df[FEATURES].values
    pd_val = float(row.iloc[0]['risk_probability'])
    band = row.iloc[0]['risk_band']

    drivers = []
    if explainer is not None or xgb_model is not None:
        if explainer is not None:
            shap_vals = explainer.shap_values(X_imp)
        else:
            import xgboost as xgb
            d = xgb.DMatrix(X_imp)
            booster = xgb_model.get_booster() if hasattr(xgb_model, 'get_booster') else xgb_model
            contribs = booster.predict(d, pred_contribs=True)
            shap_vals = contribs[:, :-1]
        feat_vals = X.iloc[0].to_dict()
        drivers = sorted(
            [{"feature": f, "impact": float(shap_vals[0][i]),
              "narrative": FEATURE_NARRATIVE.get(f, lambda x: f"{f}: {x:.3f}")(float(feat_vals[f]))}
             for i, f in enumerate(FEATURES)],
            key=lambda d: abs(d['impact']), reverse=True)[:5]

    r = row.iloc[0]
    sector = r['sector']
    lgd = LGD_MAP.get(sector, 0.4)
    ead = float(r['outstanding_loan'])
    return {
        "company_id": company_id, "sector": sector, "pd": pd_val, "band": band,
        "ead": ead, "lgd": lgd, "expected_loss": pd_val * ead * lgd,
        "drivers": drivers,
        "officer_notes": str(r.get('officer_notes', '') or ''),
        "note_stress_index": float(r['note_stress_index']) if 'note_stress_index' in r else None,
        "actions": [a["action"] for a in ACTION_LADDER.get(band, [])],
    }

def _portfolio_attention() -> list:
    feature_df = load_feature_df()
    if feature_df.empty:
        return []
    df = feature_df.copy()
    df['pd'] = df['risk_probability']
    top = df.sort_values('pd', ascending=False).head(5)
    return [{"company_id": r['company_id'], "sector": r['sector'],
             "pd": float(r['pd']), "band": r['risk_band']}
            for _, r in top.iterrows()]

@app.post("/api/events")
async def ingest_event(payload: dict = Body(...)):
    company_id = payload.get("company_id")
    event_type = payload.get("event_type")
    event_payload = payload.get("payload")
    
    if not company_id or not event_type:
        raise HTTPException(status_code=400, detail="company_id and event_type are required")
        
    if event_type not in ["transaction", "gst_filing", "emi_payment", "officer_note"]:
        raise HTTPException(status_code=400, detail="invalid event_type")
        
    event_item = {
        "company_id": company_id,
        "event_type": event_type,
        "payload": event_payload
    }
    
    await event_stream_queue.put(event_item)
    return {"status": "queued", "company_id": company_id, "event_type": event_type}

@app.websocket("/ws/risk-stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print("WebSocket client connected.")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket client disconnected.")


def copilot_engine() -> str:
    """Pick the Copilot backend by which API key is present.
    Priority: Claude → Gemini → offline templates."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "template"


COPILOT_SYSTEM = (
    "You are a credit risk officer's copilot at a bank. Answer the manager's question "
    "using ONLY the structured borrower/portfolio context provided. Be concise, factual, "
    "and write in plain manager-friendly English. Do not invent numbers."
)


# ─── Privacy-First PII Masking Layer ─────────────────────────────────────────
# Replaces company identifiers and exact financial amounts with anonymised
# equivalents before the context payload reaches any external LLM API.
# This satisfies DPDPA / RBI data-localisation compliance requirements.
def _mask_for_llm(ctx: dict) -> dict:
    """Anonymise borrower context before sending to Claude / Gemini."""
    if ctx is None:
        return None
    m = ctx.copy()
    # Mask company identifier
    m['company_id'] = 'BORROWER-REDACTED'
    # Replace exact EAD with categorical band
    ead = m.get('ead', 0)
    if ead < 500_000:
        m['ead'] = 'low-exposure (<Rs5L)'
    elif ead < 1_500_000:
        m['ead'] = 'mid-exposure (Rs5L-Rs15L)'
    else:
        m['ead'] = 'high-exposure (>Rs15L)'
    # Replace exact expected loss with a rounded band
    el = m.get('expected_loss', 0)
    m['expected_loss'] = f'~Rs{round(el/100000)*100000:,.0f}'
    # Remove raw officer notes — keep only the NLP stress score
    m.pop('officer_notes', None)
    # Remove exact LGD (bank's internal model — confidential)
    m.pop('lgd', None)
    return m


def _copilot_context_json(ctx: dict | None, mask: bool = False) -> str:
    if ctx is None:
        # Portfolio-level: only send high-level band/sector, not exact amounts
        rows = _portfolio_attention()
        masked_rows = [{"sector": r['sector'], "band": r['band'],
                        "pd_band": f"{round(r['pd']*100/5)*5}%-bucket"}
                       for r in rows]
        return json.dumps({"top_risk_accounts": masked_rows}, indent=2)
    if mask:
        ctx = _mask_for_llm(ctx)
    return json.dumps(ctx, indent=2)


def _answer_template(question: str, ctx: dict | None) -> str:
    """Offline, rule-based Copilot. Produces richer, structured explanations from
    the SHAP-driven borrower context — used when no LLM API key is configured."""
    q = question.lower()
    if ctx is None:  # portfolio-level
        rows = _portfolio_attention()
        lines = [f"• {r['company_id']} ({r['sector']}) — PD {r['pd']*100:.1f}%, {r['band']}" for r in rows]
        return ("Borrowers requiring immediate attention (highest PD):\n" + "\n".join(lines)
                + "\n\nAsk me about any borrower by opening their dossier for a full driver breakdown.")

    drv = ctx.get('drivers', [])
    def _dir(d):  # ▲ raises / ▼ lowers, from SHAP sign
        return "raises" if d.get('impact', 0) > 0 else "lowers"
    bullets = "\n".join(f"  {i+1}. {d['narrative']} ({_dir(d)} risk)" for i, d in enumerate(drv[:4])) \
              or "  • no dominant signals"
    top_up = [d for d in drv if d.get('impact', 0) > 0][:3]
    up_line = "; ".join(d['narrative'] for d in top_up) or "no dominant upward signals"

    # ── intent routing ───────────────────────────────────────────────
    if any(k in q for k in ("action", "recommend", "do next", "should i", "intervention")):
        acts = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(ctx.get('actions', []))) or "  1. Continue standard monitoring"
        return (f"{ctx['company_id']} is in the {ctx['band']} band (PD {ctx['pd']*100:.1f}%). "
                f"Recommended interventions, in priority order:\n{acts}")

    if any(k in q for k in ("note", "nlp", "officer", "comment", "narrative")):
        nsi = ctx.get('note_stress_index')
        nsi_txt = f"{nsi:.2f}/1.0" if nsi is not None else "n/a"
        note = ctx.get('officer_notes') or "No officer note on file."
        return (f"Unstructured signal for {ctx['company_id']}: the NLP officer-notes stress index is {nsi_txt}. "
                f"Latest note on file: “{note}”. This feeds the model alongside the structured features.")

    if any(k in q for k in ("loss", "exposure", "ead", "lgd", "recover")):
        return (f"Financial exposure for {ctx['company_id']}: expected loss ₹{ctx['expected_loss']:,.0f} "
                f"= PD {ctx['pd']*100:.1f}% × EAD ₹{ctx['ead']:,.0f} × LGD {ctx['lgd']*100:.0f}%. "
                f"Potential recovery on default ≈ ₹{ctx['ead']-ctx['expected_loss']:,.0f}.")

    if any(k in q for k in ("changed", "trend", "month", "six", "trajectory", "worse")):
        return (f"Over the observation window the dominant deteriorating signals for {ctx['company_id']} are: "
                f"{up_line}. Together these drove the 12-month PD to {ctx['pd']*100:.1f}% ({ctx['band']} risk).")

    # default: full "why is this borrower risky" explanation
    return (f"{ctx['company_id']} ({ctx['sector']}) carries a {ctx['pd']*100:.1f}% 12-month probability of default "
            f"— {ctx['band']} band. The model's ranked drivers are:\n{bullets}\n"
            f"Projected expected loss is ₹{ctx['expected_loss']:,.0f} "
            f"(PD × EAD ₹{ctx['ead']:,.0f} × LGD {ctx['lgd']*100:.0f}%). "
            f"Recommended next step: {ctx['actions'][0] if ctx.get('actions') else 'continue standard monitoring'}.")


def _answer_claude(question: str, ctx: dict | None) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=400,
        system=COPILOT_SYSTEM,
        messages=[{"role": "user",
                   "content": f"Context:\n{_copilot_context_json(ctx, mask=True)}\n\nManager's question: {question}"}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def _answer_gemini(question: str, ctx: dict | None) -> str:
    import time
    from google import genai
    from google.genai import types
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    config = types.GenerateContentConfig(
        system_instruction=COPILOT_SYSTEM,
        max_output_tokens=600,
        # Disable "thinking" so the token budget goes to the actual answer (and it's faster).
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    contents = f"Context:\n{_copilot_context_json(ctx, mask=True)}\n\nManager's question: {question}"

    # Gemini's free endpoint intermittently throws transient 5xx / network errors;
    # retry those with backoff. Do NOT retry ClientError (e.g. 429 quota) — bubble up fast.
    TRANSIENT = ("ServerError", "ReadError", "ConnectError", "ReadTimeout",
                 "ConnectTimeout", "RemoteProtocolError", "APIError")
    last_err = None
    for attempt in range(4):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            text = (resp.text or "").strip()
            if text:
                return text
            last_err = RuntimeError("empty response")
        except genai.errors.ClientError:
            raise  # 4xx (quota/invalid) — retrying won't help quickly
        except Exception as e:
            if type(e).__name__ not in TRANSIENT:
                raise
            last_err = e
        time.sleep(1.2 * (attempt + 1))
    raise last_err


@app.post("/copilot")
def copilot(payload: dict = Body(...)):
    """Interactive Risk Copilot. Body: {question, company_id?}.
    Answers per-borrower, or portfolio-level if company_id is omitted."""
    if FEATURES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    company_id = payload.get("company_id")

    ctx = _borrower_context(company_id) if company_id else None
    engine = copilot_engine()
    try:
        if engine == "claude":
            answer = _answer_claude(question, ctx)
        elif engine == "gemini":
            answer = _answer_gemini(question, ctx)
        else:
            answer = _answer_template(question, ctx)
    except Exception as e:
        # Never fail the demo — fall back to the local template engine
        answer = _answer_template(question, ctx)
        engine = f"template (fallback: {type(e).__name__})"
    return {"engine": engine, "company_id": company_id, "question": question, "answer": answer}
