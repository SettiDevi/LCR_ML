from fastapi import FastAPI
import pandas as pd
import requests
import joblib
 
app = FastAPI()
 
# ================= CONFIG =================
# ✅ YES, you MUST give ServiceNow instance details
# (but see Section 2 below for BEST PRACTICE)
 
SERVICENOW_INSTANCE = "https://YOUR_INSTANCE.service-now.com"
SN_USER = "YOUR_API_USER"
SN_PASS = "YOUR_API_PASSWORD"
 
FEATURE_STORE_TABLE = "u_lcr_feature_store"
PREDICTIONS_TABLE = "u_lcr_predictions"
MODEL_NAME = "LCR_ML_v1"
 
# ============== LOAD MODEL ===============
# ✅ File name MUST match GitHub file exactly
model = joblib.load("reclaim_model.pkl")
 
# ========== BUSINESS LOGIC ==============
def decide_action_and_reason(row):
    days = int(row["u_days_since_last_use"])
    premium_used = int(row["u_premium_feature_usage_last_30_days"])
    reclaim_prob = float(row["reclaim_probability"])
 
    if reclaim_prob >= 0.8:
        return "RECLAIM", "High reclaim risk predicted by ML"
 
    if days >= 60 and premium_used == 0:
        return "DOWNGRADE", "Premium features not used for 60+ days"
 
    if days >= 45:
        return "WARNING", "Inactive for 45+ days"
 
    return "KEEP", "License actively used"
 
# ============== API ======================
@app.post("/run_predictions")
def run_predictions():
 
    # 1️⃣ Fetch Feature Store from ServiceNow
    response = requests.get(
        f"{SERVICENOW_INSTANCE}/api/now/table/{FEATURE_STORE_TABLE}",
        auth=(SN_USER, SN_PASS),
        headers={"Accept": "application/json"}
    )
 
    if response.status_code != 200:
        return {
            "status": "error",
            "message": "Failed to fetch feature store",
            "code": response.status_code
        }
 
    data = response.json().get("result", [])
    df = pd.DataFrame(data)
 
    if df.empty:
        return {"status": "no data"}
 
    # 2️⃣ Feature columns
    feature_cols = [
        "u_days_since_last_use",
        "u_active_days_last_30_days",
        "u_active_days_last_90_days",
        "u_premium_feature_usage_last_30_days",
        "u_seasonal_user",
        "u_user_active"
    ]
 
    X = df[feature_cols].astype(float)
 
    # 3️⃣ ML inference
    df["reclaim_probability"] = model.predict_proba(X)[:, 1]
 
    # 4️⃣ Decide action + reason
    df[["action", "reason"]] = df.apply(
        lambda r: decide_action_and_reason(r),
        axis=1,
        result_type="expand"
    )
 
    # 5️⃣ Insert predictions into ServiceNow
    inserted = 0
 
    for _, row in df.iterrows():
        payload = {
            "u_user": row["u_user"],
            "u_license_sku": row["u_license_sku"],
            "u_predicted_action": row["action"],
            "u_ai_reclaim_confidence": row["reclaim_probability"],
            "u_ai_reclaim_reason": row["reason"],
            "u_model_name": MODEL_NAME,
            "u_notification_stage": "NONE"
        }
 
        r = requests.post(
            f"{SERVICENOW_INSTANCE}/api/now/table/{PREDICTIONS_TABLE}",
            auth=(SN_USER, SN_PASS),
            json=payload
        )
 
        if r.status_code == 201:
            inserted += 1
 
    return {
        "status": "success",
        "records_processed": len(df),
        "records_inserted": inserted
    }
 
