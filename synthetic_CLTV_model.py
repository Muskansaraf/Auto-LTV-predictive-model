import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score, r2_score, accuracy_score


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_excel("synthetic_cltv_data.xlsx")

print("Shape:", df.shape)
print(df.head())

# =========================================================
# 1B. FEATURE ENGINEERING
# =========================================================

# Vehicle age
df["vehicle_age"] = 2024 - df["vehicle_year"]

# Age buckets
df["young_driver"] = (df["driver_age"] < 25).astype(int)
df["senior_driver"] = (df["driver_age"] > 70).astype(int)

# Mileage buckets
df["high_mileage"] = (df["annual_mileage"] > 16000).astype(int)
df["low_mileage"] = (df["annual_mileage"] < 8000).astype(int)

# Safety score (simple weighted sum)
df["safety_score"] = (
    1.0 * df["has_abs"]
    + 1.2 * df["has_esc"]
    + 0.4 * df["has_collision_avoidance"]
    + 0.3 * df["has_lane_assist"]
    + 0.3 * df["has_blind_spot_monitor"]
    + 0.1 * df["airbag_count"]
)

# ZIP-based flags (convert to numeric first)
zip_num = df["zipcode"].astype(int)
# Synthetic assumption: smaller zips → more urban, larger → more rural
df["zip_urban_flag"] = (zip_num < 44000).astype(int)
df["zip_rural_flag"] = (zip_num >= 45000).astype(int)

# Weight factor & power-to-weight ratio
weight_map = {"Light": 1.0, "Medium": 1.2, "Heavy": 1.4}
df["weight_factor"] = df["weight_class"].map(weight_map)
df["power_to_weight"] = df["horsepower"] / (100.0 * df["weight_factor"].replace(0, 1))

# Risk × mileage interaction
df["risk_x_mileage"] = df["base_risk"] * (df["annual_mileage"] / 10000.0)

# Claims per ownership year (avoid division by zero)
df["claims_per_ownership_year"] = df["claims_history_count"] / df["length_of_ownership"].clip(lower=1.0)

# Value ratio
df["value_to_msrp_ratio"] = df["current_value"] / df["msrp"].replace(0, np.nan)
df["value_to_msrp_ratio"] = df["value_to_msrp_ratio"].fillna(0)

# Luxury + sporty flag
df["luxury_sport_flag"] = df["luxury_brand"] * (df["horsepower"] > 220).astype(int)

print("\nAfter feature engineering, columns:", len(df.columns))


# =========================================================
# 2. DEFINE FEATURES & TARGETS
# =========================================================

target_premium = "premium"
target_claim_flag = "claim_flag"
target_claim_amount = "claim_amount"
target_churn_flag = "churn_flag"

exclude_cols = [
    target_premium,
    target_claim_flag,
    target_claim_amount,
    target_churn_flag,
    "claim_prob_true",
    "churn_prob_true",
    # If you want the model NOT to see base_risk directly, uncomment below:
    # "base_risk",
]

feature_cols = [c for c in df.columns if c not in exclude_cols]

categorical_features = [
    "ownership_type",
    "zipcode",
    "vehicle_region",
    "body_style",
    "engine_size_category",
    "drive_type",
    "transmission",
    "fuel_type",
    "weight_class",
]

numeric_features = [c for c in feature_cols if c not in categorical_features]

print("\nNumeric features:", numeric_features)
print("\nCategorical features:", categorical_features)

X = df[feature_cols].copy()
y_premium = df[target_premium]
y_claim_flag = df[target_claim_flag]
y_claim_amount = df[target_claim_amount]
y_churn_flag = df[target_churn_flag]

(
    X_train,
    X_test,
    y_premium_train,
    y_premium_test,
    y_claim_flag_train,
    y_claim_flag_test,
    y_claim_amount_train,
    y_claim_amount_test,
    y_churn_flag_train,
    y_churn_flag_test,
) = train_test_split(
    X,
    y_premium,
    y_claim_flag,
    y_claim_amount,
    y_churn_flag,
    test_size=0.2,
    random_state=42,
    stratify=y_claim_flag,
)

# =========================================================
# 3. PREPROCESSOR FACTORY
# =========================================================

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

# =========================================================
# 4. PREMIUM MODEL (Regression)
# =========================================================

premium_pipeline = Pipeline(
    steps=[
        ("preprocess", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )),
    ]
)

print("\nTraining premium model...")
premium_pipeline.fit(X_train, y_premium_train)

pred_premium_test = premium_pipeline.predict(X_test)
mae_premium = mean_absolute_error(y_premium_test, pred_premium_test)
r2_premium = r2_score(y_premium_test, pred_premium_test)
print("Premium MAE:", round(mae_premium, 2))
print("Premium R²:", round(r2_premium, 3))


# =========================================================
# 5. CLAIM MODELS: Probability + Severity
# =========================================================

# 5a) Claim probability
claim_clf_pipeline = Pipeline(
    steps=[
        ("preprocess", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ]
)

print("\nTraining claim probability model...")
claim_clf_pipeline.fit(X_train, y_claim_flag_train)

pred_claim_prob_test = claim_clf_pipeline.predict_proba(X_test)[:, 1]
try:
    auc_claim = roc_auc_score(y_claim_flag_test, pred_claim_prob_test)
    print("Claim probability ROC AUC:", round(auc_claim, 3))
except ValueError:
    print("Not enough classes to compute ROC AUC for claim model.")

pred_claim_label_test = (pred_claim_prob_test >= 0.5).astype(int)
acc_claim = accuracy_score(y_claim_flag_test, pred_claim_label_test)
print("Claim probability accuracy:", round(acc_claim, 3))


# 5b) Claim severity (only where claim_flag == 1)
claim_mask_train = y_claim_flag_train == 1
X_train_claim = X_train[claim_mask_train]
y_claim_amount_train_nonzero = y_claim_amount_train[claim_mask_train]

claim_sev_pipeline = Pipeline(
    steps=[
        ("preprocess", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )),
    ]
)

print("\nTraining claim severity model (on claim_flag == 1)...")
if len(X_train_claim) > 0:
    claim_sev_pipeline.fit(X_train_claim, y_claim_amount_train_nonzero)
    pred_claim_severity_test = claim_sev_pipeline.predict(X_test)
else:
    print("No claims in training set, setting severity predictions to 0.")
    pred_claim_severity_test = np.zeros(len(X_test))

claim_mask_test = y_claim_flag_test == 1
if claim_mask_test.sum() > 0:
    mae_claim_sev = mean_absolute_error(
        y_claim_amount_test[claim_mask_test],
        pred_claim_severity_test[claim_mask_test]
    )
    r2_claim_sev = r2_score(
        y_claim_amount_test[claim_mask_test],
        pred_claim_severity_test[claim_mask_test]
    )
    print("Claim severity MAE (on claim_flag==1):", round(mae_claim_sev, 2))
    print("Claim severity R² (on claim_flag==1):", round(r2_claim_sev, 3))
else:
    print("No claims in test set to evaluate severity R²/MAE.")


# =========================================================
# 6. CHURN / RETENTION MODEL (Classification)
# =========================================================

churn_pipeline = Pipeline(
    steps=[
        ("preprocess", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ]
)

print("\nTraining churn model...")
churn_pipeline.fit(X_train, y_churn_flag_train)

pred_churn_prob_test = churn_pipeline.predict_proba(X_test)[:, 1]
try:
    auc_churn = roc_auc_score(y_churn_flag_test, pred_churn_prob_test)
    print("Churn ROC AUC:", round(auc_churn, 3))
except ValueError:
    print("Not enough classes to compute ROC AUC for churn model.")

pred_churn_label_test = (pred_churn_prob_test >= 0.5).astype(int)
acc_churn = accuracy_score(y_churn_flag_test, pred_churn_label_test)
print("Churn accuracy:", round(acc_churn, 3))


# =========================================================
# 7. CLTV CALCULATION
# =========================================================

eps = 1e-3
churn_prob_clipped = np.clip(pred_churn_prob_test, eps, 1.0)
expected_tenure_years = 1.0 / churn_prob_clipped
expected_tenure_years = np.clip(expected_tenure_years, 0, 10)

expected_loss = pred_claim_prob_test * pred_claim_severity_test

cltv = (pred_premium_test * expected_tenure_years) - expected_loss

# =========================================================
# 8. BUILD RESULTS DATAFRAME & SAVE TO EXCEL
# =========================================================

results = X_test.copy()
results["true_premium"] = y_premium_test.values
results["pred_premium"] = np.round(pred_premium_test, 2)

results["true_claim_flag"] = y_claim_flag_test.values
results["pred_claim_prob"] = np.round(pred_claim_prob_test, 4)
results["pred_claim_severity"] = np.round(pred_claim_severity_test, 2)

results["true_churn_flag"] = y_churn_flag_test.values
results["pred_churn_prob"] = np.round(pred_churn_prob_test, 4)
results["expected_tenure_years"] = np.round(expected_tenure_years, 2)

results["expected_loss"] = np.round(expected_loss, 2)
results["CLTV"] = np.round(cltv, 2)

results = results.sort_values("CLTV", ascending=False)

output_file = "cltv_predictions.xlsx"
results.to_excel(output_file, index=False)
print(f"\nSaved CLTV predictions to: {output_file}")
print("Top 5 rows by CLTV:")
print(results[["CLTV", "pred_premium", "pred_claim_prob", "pred_claim_severity", "pred_churn_prob"]].head())
