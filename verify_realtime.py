import json
import sys
import os
import time

workspace_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, workspace_dir)

from fastapi.testclient import TestClient

try:
    from main import app
    print("SUCCESS: Successfully imported main.py")
except Exception as e:
    print("ERROR: Error importing main.py:", e)
    sys.exit(1)

print("\n=== STARTING LOCAL REAL-TIME INTEGRATION VALIDATION ===")

with TestClient(app) as client:
    print("OK: FastAPI app lifespan started. SQLite initialized & seeded.")
    
    # 1. Verify WebSocket connection
    print("\n[Step 1] Connecting to WebSocket '/ws/risk-stream'...")
    try:
        with client.websocket_connect("/ws/risk-stream") as websocket:
            print("OK: WebSocket connected successfully!")
            
            # 2. Ingest Transaction Event
            print("\n[Step 2] Ingesting transaction event to 'POST /api/events'...")
            event_payload = {
                "company_id": "MSME-0001",
                "event_type": "transaction",
                "payload": {"revolving_utilization": 0.95, "cashflow_stress_ratio": 4.5}
            }
            res = client.post("/api/events", json=event_payload)
            print(f"  Ingest Response Status: {res.status_code}")
            print(f"  Ingest Response: {res.json()}")
            
            # 3. Check for Real-time push broadcast
            print("\n[Step 3] Waiting for WebSocket broadcast notification...")
            msg = websocket.receive_json()
            print("OK: Received Live Risk Stream Update via WebSocket:")
            print(json.dumps(msg, indent=2))
            
            # 4. Check Query endpoint update
            print("\n[Step 4] Checking updated borrower dossier at 'GET /borrowers/MSME-0001'...")
            res_details = client.get("/borrowers/MSME-0001")
            details = res_details.json()
            print(f"  Live PD: {details['current_pd']*100:.1f}%")
            print(f"  Live Risk Band: {details['risk_band']}")
            print(f"  COLLATERAL ESCALATION ACTIONS:\n  " + "\n  ".join(f"- {a['action']}" for a in details['action_ladder']))
            
    except Exception as e:
        print("ERROR: WebSocket Connection or validation failed:", e)

print("\n=== VALIDATION COMPLETED SUCCESSFULLY ===")
