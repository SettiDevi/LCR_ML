from fastapi import FastAPI, BackgroundTasks
import pandas as pd
import requests
import joblib
import os
import traceback
from datetime import datetime

app = FastAPI()

# ================= CONFIG =================
SERVICENOW_INSTANCE = os.getenv("SN_INSTANCE")
SN_USER = os.getenv("SN_USER")
SN_PASS = os.getenv("SN_PASS")

FEATURE_STORE_TABLE = "u_lcr_feature_store"
PREDICTIONS_TABLE = "u_lcr_predictions"
MODEL_NAME = "LCR_ML_v1"

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# ================= LOAD MODEL =================
model = joblib.load("reclaim_model.pkl")

# ================= HEALTH CHECK =================
@app.get("/")
def health():
    return {"status": "LCR ML service running"}

# ================= TRIGGER ENDPOINT =================
@app.post("/start_predictions")
def start_predictions(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_predictions_job)
    return {
        "status": "accepted",
        "message": "Prediction job started in background"
    }

# ================= BACKGROUND JOB =================
def run_predictions_job():
    try:
        print("🔹 LCR Prediction job started")

        # ---------- 1. FETCH ALL FEATURE STORE RECORDS ----------
        all_rows = []
        offset = 0
        limit = 1000

        while True:
            resp = requests.get(
                f"{SERVICENOW_INSTANCE}/api/now/table/{FEATURE_STORE_TABLE}",
                auth=(SN_USER, SN_PASS),
                headers=HEADERS,
                params={
                    "sysparm_limit": limit,
                    "sysparm_offset": offset,
                    "sysparm_query": "u_userISNOTEMPTY^u_license_skuISNOTEMPTY",
                    "sysparm_display_value": "false"  # ✅ CRITICAL FIX
                },
                timeout=90
            )

            data = resp.json().get("result", [])
            if not data:
                break

            all_rows.extend(data)
            offset += limit

        if not all_rows:
            print("⚠ No records found in feature store")
            return

        print(f"✅ Total feature rows fetched: {len(all_rows)}")

        df = pd.DataFrame(all_rows)

        # ---------- 2. FEATURE PREPARATION ----------
        feature_cols = [
            "u_days_since_last_use",
            "u_active_days_last_30_days",
            "u_active_days_last_90_days",
            "u_premium_feature_usage_last_30_days",
            "u_seasonal_user",
            "u_user_active"
        ]

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        for col in ["u_seasonal_user", "u_user_active"]:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .map({"true": 1, "false": 0})
                .fillna(0)
            )

        X = df[feature_cols].fillna(0).astype(float)

        # ---------- 3. ML PREDICTION ----------
        df["reclaim_probability"] = model.predict_proba(X)[:, 1]

        # ---------- 4. BUSINESS DECISION LOGIC ----------
        def decide(row):
            days = int(row.get("u_days_since_last_use", 0))
            premium = int(row.get("u_premium_feature_usage_last_30_days", 0))
            prob = float(row["reclaim_probability"])

            if prob >= 0.8:
                return "RECLAIM", "High reclaim risk predicted by ML"
            if days >= 60 and premium == 0:
                return "DOWNGRADE", "Premium features not used for 60+ days"
            if days >= 45:
                return "WARNING", "Inactive for 45+ days"
            return "KEEP", "License actively used"

        df[["u_predicted_action", "u_ai_reclaim_reason"]] = df.apply(
            decide, axis=1, result_type="expand"
        )

        # ---------- 5. UPSERT PREDICTIONS ----------
        processed = 0

        for _, row in df.iterrows():
            user_sys_id = row["u_user"]           # ✅ real sys_id
            license_sys_id = row["u_license_sku"] # ✅ real sys_id

            payload = {
                "u_user": user_sys_id,
                "u_license_sku": license_sys_id,
                "u_predicted_action": row["u_predicted_action"],
                "u_ai_reclaim_confidence": round(float(row["reclaim_probability"]), 2),
                "u_ai_reclaim_reason": row["u_ai_reclaim_reason"],
                "u_model_name": MODEL_NAME,
                "u_predicted_on": datetime.utcnow().isoformat()
            }

            check = requests.get(
                f"{SERVICENOW_INSTANCE}/api/now/table/{PREDICTIONS_TABLE}",
                auth=(SN_USER, SN_PASS),
                headers=HEADERS,
                params={
                    "sysparm_query": f"u_user={user_sys_id}^u_license_sku={license_sys_id}",
                    "sysparm_limit": 1
                }
            )

            existing = check.json().get("result", [])

            if existing:
                sys_id = existing[0]["sys_id"]
                requests.put(
                    f"{SERVICENOW_INSTANCE}/api/now/table/{PREDICTIONS_TABLE}/{sys_id}",
                    auth=(SN_USER, SN_PASS),
                    headers=HEADERS,
                    json=payload
                )
            else:
                requests.post(
                    f"{SERVICENOW_INSTANCE}/api/now/table/{PREDICTIONS_TABLE}",
                    auth=(SN_USER, SN_PASS),
                    headers=HEADERS,
                    json=payload
                )

            processed += 1

        print("✅ Prediction job completed successfully")
        print(f"✅ Total records processed: {processed}")

    except Exception:
        print("❌ Error during prediction job")
        traceback.print_exc()
