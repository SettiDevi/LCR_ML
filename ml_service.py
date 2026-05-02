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

WRITE_PARAMS = {
    "sysparm_input_display_value": "false"
}

# ================= LOAD MODEL =================
model = joblib.load("reclaim_model.pkl")


# ================= UTIL FUNCTIONS =================
def get_sys_id(val):
    """
    Handles ServiceNow reference fields safely.
    """
    if isinstance(val, dict):
        return val.get("value", "").strip()
    return str(val).strip() if val else ""


def safe_request(method, url, **kwargs):
    """
    Wrapper for requests with logging.
    """
    try:
        resp = requests.request(method, url, timeout=90, **kwargs)

        if resp.status_code not in [200, 201]:
            print(f"❌ API Error [{resp.status_code}] -> {resp.text}")

        return resp
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


# ================= HEALTH CHECK =================
@app.get("/")
def health():
    return {"status": "LCR ML service running"}


# ================= START ENDPOINT =================
@app.post("/start_predictions")
def start_predictions(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_predictions_job)
    return {
        "status": "accepted",
        "message": "Prediction job started in background"
    }


# ================= MAIN JOB =================
def run_predictions_job():
    try:
        print("🔹 LCR Prediction job started")

        # ---------- FETCH FEATURE STORE DATA ----------
        all_rows = []
        offset = 0
        limit = 1000

        while True:
            resp = safe_request(
                "GET",
                f"{SERVICENOW_INSTANCE}/api/now/table/{FEATURE_STORE_TABLE}",
                auth=(SN_USER, SN_PASS),
                headers=HEADERS,
                params={
                    "sysparm_limit": limit,
                    "sysparm_offset": offset,
                    "sysparm_query": "u_userISNOTEMPTY^u_license_skuISNOTEMPTY",
                    "sysparm_display_value": "false"
                }
            )

            if not resp:
                break

            data = resp.json().get("result", [])

            print(f"📦 Batch fetched: {len(data)} rows | offset={offset}")

            if not data:
                break

            all_rows.extend(data)
            offset += limit

        if not all_rows:
            print("⚠ No feature store data found")
            return

        print(f"✅ Total rows fetched: {len(all_rows)}")

        # ---------- DATAFRAME ----------
        df = pd.DataFrame(all_rows)

        feature_cols = [
            "u_days_since_last_use",
            "u_active_days_last_30_days",
            "u_active_days_last_90_days",
            "u_premium_feature_usage_last_30_days",
            "u_seasonal_user",
            "u_user_active"
        ]

        # fill missing columns
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # boolean conversion
        for col in ["u_seasonal_user", "u_user_active"]:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .map({"true": 1, "false": 0})
                .fillna(0)
            )

        X = df[feature_cols].fillna(0).astype(float)

        # ---------- ML PREDICTION ----------
        df["reclaim_probability"] = model.predict_proba(X)[:, 1]

        # ---------- DECISION LOGIC ----------
        def decide(row):
            days = int(float(row.get("u_days_since_last_use", 0)))
            premium = int(float(row.get("u_premium_feature_usage_last_30_days", 0)))
            prob = float(row["reclaim_probability"])

            if prob >= 0.8:
                return "RECLAIM", "High reclaim risk predicted by ML"

            if days >= 60 and premium == 0:
                return "DOWNGRADE", "Premium features not used for 60+ days"

            if days >= 45:
                return "WARNING", "Inactive for 45+ days"

            return "KEEP", "License actively used"

        df[["u_predicted_action", "u_ai_reclaim_reason"]] = df.apply(
            decide,
            axis=1,
            result_type="expand"
        )

        # ---------- DEBUG UNIQUE COMBINATIONS ----------
        unique_count = (
            df[["u_user", "u_license_sku"]]
            .astype(str)
            .drop_duplicates()
            .shape[0]
        )

        print(f"🔍 Unique user-license combinations: {unique_count}")

        # ---------- UPSERT ----------
        processed = 0
        inserted = 0
        updated = 0
        skipped = 0

        for _, row in df.iterrows():

            user_sys_id = get_sys_id(row.get("u_user"))
            license_sys_id = get_sys_id(row.get("u_license_sku"))

            if not user_sys_id or not license_sys_id:
                print(f"⚠ Skipping empty refs: user={user_sys_id}, license={license_sys_id}")
                skipped += 1
                continue

            payload = {
                "u_user": user_sys_id,
                "u_license_sku": license_sys_id,
                "u_predicted_action": row["u_predicted_action"],
                "u_ai_reclaim_confidence": round(float(row["reclaim_probability"]), 2),
                "u_ai_reclaim_reason": row["u_ai_reclaim_reason"],
                "u_model_name": MODEL_NAME,
                "u_predicted_on": datetime.utcnow().isoformat()
            }

            # check existing prediction
            check_resp = safe_request(
                "GET",
                f"{SERVICENOW_INSTANCE}/api/now/table/{PREDICTIONS_TABLE}",
                auth=(SN_USER, SN_PASS),
                headers=HEADERS,
                params={
                    "sysparm_query": f"u_user={user_sys_id}^u_license_sku={license_sys_id}",
                    "sysparm_limit": 1,
                    "sysparm_display_value": "false"
                }
            )

            existing = []
            if check_resp:
                existing = check_resp.json().get("result", [])

            if existing:
                sys_id = existing[0]["sys_id"]

                resp = safe_request(
                    "PUT",
                    f"{SERVICENOW_INSTANCE}/api/now/table/{PREDICTIONS_TABLE}/{sys_id}",
                    auth=(SN_USER, SN_PASS),
                    headers=HEADERS,
                    params=WRITE_PARAMS,
                    json=payload
                )

                if resp and resp.status_code in [200, 201]:
                    updated += 1

            else:
                resp = safe_request(
                    "POST",
                    f"{SERVICENOW_INSTANCE}/api/now/table/{PREDICTIONS_TABLE}",
                    auth=(SN_USER, SN_PASS),
                    headers=HEADERS,
                    params=WRITE_PARAMS,
                    json=payload
                )

                if resp and resp.status_code in [200, 201]:
                    inserted += 1

            processed += 1

            if processed % 100 == 0:
                print(f"🚀 Processed {processed} records...")

        # ---------- SUMMARY ----------
        print("\n========== JOB SUMMARY ==========")
        print(f"Processed : {processed}")
        print(f"Inserted  : {inserted}")
        print(f"Updated   : {updated}")
        print(f"Skipped   : {skipped}")
        print("✅ Prediction job completed successfully")

    except Exception:
        print("❌ Error during prediction job")
        traceback.print_exc()
