from fastapi import FastAPI
import pandas as pd
import requests
import joblib
import os
import traceback

app = FastAPI()

# ================= CONFIG (ENV VARIABLES) =================
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

# ================= BUSINESS LOGIC =================
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
        print("RUN_PREDICTIONS CALLED")

        # 1️⃣ Fetch feature store
        response = requests.get(
            f"{SERVICENOW_INSTANCE}/api/now/table/{FEATURE_STORE_TABLE}",
            auth=(SN_USER, SN_PASS),
            headers=HEADERS,
            params={"sysparm_limit": "10000"},
            timeout=60
        )

        print("Feature store HTTP status:", response.status_code)

        if response.status_code != 200:
            print("Feature store fetch failed:", response.text)
            return {
                "status": "error",
                "message": "Failed to fetch feature store",
                "details": response.text
            }

        data = response.json().get("result", [])
        df = pd.DataFrame(data)

        print("Rows fetched:", len(df))
        print("Columns received:", df.columns.tolist())

        if df.empty:
            return {"status": "no data"}

        # 2️⃣ Select ML features
        feature_cols = [
            "u_days_since_last_use",
            "u_active_days_last_30_days",
            "u_active_days_last_90_days",
            "u_premium_feature_usage_last_30_days",
            "u_seasonal_user",
            "u_user_active"
        ]

        # Ensure all required columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Safe numeric conversion
        X = df[feature_cols].fillna(0).astype(float)

        # 3️⃣ ML prediction
        df["reclaim_probability"] = model.predict_proba(X)[:, 1]

        # 4️⃣ Decide action & reason
        df[["u_predicted_action", "u_ai_reclaim_reason"]] = df.apply(
            lambda r: decide_action_and_reason(r),
            axis=1,
            result_type="expand"
        )

        inserted = 0

        # 5️⃣ Insert predictions into ServiceNow
        for _, row in df.iterrows():
            payload = {
                "u_user": row.get("u_user"),
                "u_license_sku": row.get("u_license_sku"),
                "u_predicted_action": row.get("u_predicted_action"),
                "u_ai_reclaim_confidence": round(float(row["reclaim_probability"]), 2),
                "u_ai_reclaim_reason": row.get("u_ai_reclaim_reason"),
                "u_model_name": MODEL_NAME,
                "u_notification_stage": "NONE"
            }

            r = requests.post(
                f"{SERVICENOW_INSTANCE}/api/now/table/{PREDICTIONS_TABLE}",
                auth=(SN_USER, SN_PASS),
                headers=HEADERS,
                json=payload,
                timeout=30
            )

            if r.status_code == 201:
                inserted += 1
            else:
                print("Insert failed:", r.status_code, r.text)

        return {
            "status": "success",
            "records_processed": len(df),
            "records_inserted": inserted
        }

    except Exception as e:
        print("EXCEPTION OCCURRED")
        traceback.print_exc()

        return {
            "status": "error",
            "message": "Internal server error in ML service",
            "details": str(e)
        }
