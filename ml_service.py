from fastapi import FastAPI
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

# ✅ Batch + pagination
BATCH_SIZE = 5000
OFFSET = int(os.getenv("OFFSET", 0))

# ================= LOAD MODEL =================
model = joblib.load("reclaim_model.pkl")

# ================= HELPERS =================
def get_user_sys_id(username):
    """Resolve username → sys_user.sys_id"""
    r = requests.get(
        f"{SERVICENOW_INSTANCE}/api/now/table/sys_user",
        auth=(SN_USER, SN_PASS),
        headers=HEADERS,
        params={
            "sysparm_query": f"user_name={username}",
            "sysparm_limit": "1"
        }
    )
    res = r.json().get("result", [])
    return res[0]["sys_id"] if res else None

def decide_action_and_reason(row):
    days = int(row.get("u_days_since_last_use", 0))
    premium_used = int(row.get("u_premium_feature_usage_last_30_days", 0))
    reclaim_prob = float(row.get("reclaim_probability", 0))

    if reclaim_prob >= 0.8:
        return "RECLAIM", "High reclaim risk predicted by ML"

    if days >= 60 and premium_used == 0:
        return "DOWNGRADE", "Premium features not used for 60+ days"

    if days >= 45:
        return "WARNING", "Inactive for 45+ days"

    return "KEEP", "License actively used"

# ================= API =================
@app.get("/")
def health_check():
    return {"status": "LCR ML service running"}

@app.post("/run_predictions")
def run_predictions():
    try:
        # 1️⃣ Fetch feature store WITH pagination
        fs = requests.get(
            f"{SERVICENOW_INSTANCE}/api/now/table/{FEATURE_STORE_TABLE}",
            auth=(SN_USER, SN_PASS),
            headers=HEADERS,
            params={
                "sysparm_limit": BATCH_SIZE,
                "sysparm_offset": OFFSET
            },
            timeout=90
        )

        if fs.status_code != 200:
            return {"status": "error", "details": fs.text}

        df = pd.DataFrame(fs.json().get("result", []))
        if df.empty:
            return {"status": "no data"}

        # 2️⃣ Feature preparation
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

        # Convert boolean strings → numbers
        for col in ["u_seasonal_user", "u_user_active"]:
            df[col] = (
                df[col].astype(str)
                .str.lower()
                .map({"true": 1, "false": 0})
                .fillna(0)
            )

        X = df[feature_cols].fillna(0).astype(float)
        df["reclaim_probability"] = model.predict_proba(X)[:, 1]

        df[["u_predicted_action", "u_ai_reclaim_reason"]] = df.apply(
            lambda r: decide_action_and_reason(r),
            axis=1,
            result_type="expand"
        )

        processed = 0

        # 3️⃣ UPSERT predictions
        for _, row in df.iterrows():

            if not row.get("u_user") or not row.get("u_license_sku"):
                continue

            user_sys_id = get_user_sys_id(row["u_user"])
            if not user_sys_id:
                continue

            payload = {
                "u_user": user_sys_id,
                "u_license_sku": row["u_license_sku"],
                "u_predicted_action": row["u_predicted_action"],
                "u_ai_reclaim_confidence": round(float(row["reclaim_probability"]), 2),
                "u_ai_reclaim_reason": row["u_ai_reclaim_reason"],
                "u_model_name": MODEL_NAME,
                "u_notification_stage": "NONE",
                "u_predicted_on": datetime.utcnow().isoformat()
            }

            check = requests.get(
                f"{SERVICENOW_INSTANCE}/api/now/table/{PREDICTIONS_TABLE}",
                auth=(SN_USER, SN_PASS),
                headers=HEADERS,
                params={
                    "sysparm_query": f"u_user={user_sys_id}^u_license_sku={row['u_license_sku']}",
                    "sysparm_limit": "1"
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

        return {
            "status": "success",
            "records_processed": processed,
            "offset_used": OFFSET
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "details": str(e)}
