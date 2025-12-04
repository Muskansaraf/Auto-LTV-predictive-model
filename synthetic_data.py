import numpy as np
import pandas as pd

def generate_auto_cltv_synthetic(n=50000, random_state=42):
    np.random.seed(random_state)

    # --------------------------
    # 1. QUOTATION INPUTS
    # --------------------------
    driver_age = np.random.randint(18, 80, n)
    driving_history_score = np.random.randint(0, 11, n)  # 0 = clean, 10 = bad
    annual_mileage = np.random.normal(12000, 4000, n).clip(3000, 30000)

    vehicle_year = np.random.randint(2000, 2024, n)

    ownership_type = np.random.choice(
        ["Owned", "Financed", "Leased"],
        size=n,
        p=[0.55, 0.30, 0.15]
    )

    length_of_ownership = (
        (2024 - vehicle_year) * np.random.uniform(0.4, 1.0, size=n)
    ).clip(0, 15)

    prior_insurance = np.random.choice([0, 1], size=n, p=[0.15, 0.85])
    claims_history_count = np.random.poisson(0.2, size=n)

    # Ohio ZIP codes: numeric 43001–45999, stored as 5-digit strings
    zip_num = np.random.randint(43001, 45999, n)
    zipcode = np.char.zfill(zip_num.astype(str), 5)

    # --------------------------
    # 2. VIN-DERIVED FEATURES
    # --------------------------
    vehicle_region = np.random.choice(
        ["USA", "Asia", "Europe"],
        size=n,
        p=[0.5, 0.3, 0.2]
    )

    body_style = np.random.choice(
        ["Sedan", "SUV", "Truck", "Coupe"],
        size=n,
        p=[0.4, 0.35, 0.2, 0.05]
    )

    engine_size_category = np.random.choice(
        ["Small", "Medium", "Large"],
        size=n,
        p=[0.3, 0.5, 0.2]
    )

    drive_type = np.random.choice(
        ["FWD", "RWD", "AWD"],
        size=n,
        p=[0.5, 0.3, 0.2]
    )

    transmission = np.random.choice(
        ["Automatic", "Manual"],
        size=n,
        p=[0.85, 0.15]
    )

    fuel_type = np.random.choice(
        ["Gas", "Diesel", "Hybrid", "EV"],
        size=n,
        p=[0.7, 0.1, 0.15, 0.05]
    )

    horsepower = np.random.normal(180, 40, n).clip(80, 450)

    weight_class = np.random.choice(
        ["Light", "Medium", "Heavy"],
        size=n,
        p=[0.4, 0.45, 0.15]
    )

    # Luxury flag (simplified)
    luxury_brand = np.random.binomial(1, 0.15, size=n)

    # Safety features
    has_abs = np.random.binomial(1, 0.95, size=n)
    has_esc = np.random.binomial(1, 0.85, size=n)
    airbag_count = np.random.choice([4, 6, 8, 10], size=n, p=[0.2, 0.5, 0.25, 0.05])
    has_collision_avoidance = np.random.binomial(1, 0.4, size=n)
    has_lane_assist = np.random.binomial(1, 0.35, size=n)
    has_blind_spot_monitor = np.random.binomial(1, 0.4, size=n)

    # Theft & anti-theft
    has_immobilizer = np.random.binomial(1, 0.8, size=n)
    has_anti_theft = np.random.binomial(1, 0.6, size=n)
    theft_risk_score = np.random.beta(2, 5, size=n)  # mostly low but some high-risk cars

    # Value & repair
    # Simple MSRP model based on luxury + horsepower + body style
    body_style_value_map = {"Sedan": 1.0, "SUV": 1.2, "Truck": 1.3, "Coupe": 1.4}
    body_value_mult = np.vectorize(body_style_value_map.get)(body_style)

    msrp = (
        20000
        + luxury_brand * 15000
        + (horsepower - 150) * 120
        + body_value_mult * 3000
    )
    msrp = msrp.clip(15000, 90000)

    age_years = 2024 - vehicle_year
    depreciation_rate = np.clip(0.05 + 0.02 * age_years, 0.05, 0.8)
    current_value = msrp * (1 - depreciation_rate).clip(0.2, 0.95)

    # Repair cost index: luxury + body + EV etc.
    repair_cost_index = (
        0.3 * luxury_brand
        + 0.2 * (fuel_type == "EV").astype(int)
        + 0.1 * (body_style == "Coupe").astype(int)
        + 0.1 * (body_style == "Truck").astype(int)
        + np.random.normal(0.1, 0.05, size=n)
    )
    repair_cost_index = repair_cost_index.clip(0, 1)

    # --------------------------
    # 3. BASE RISK SCORE
    # --------------------------
    mileage_risk = (annual_mileage > 15000).astype(int) * 0.1

    age_risk = np.where(
        driver_age < 25, 0.20,
        np.where(driver_age > 70, 0.15, 0.05)
    )

    history_risk = driving_history_score / 30.0  # ~0 to 0.33
    claim_history_risk = claims_history_count * 0.15

    # Ohio zipcode risk: smooth function over range + noise
    zipcode_risk = (
        0.04
        + 0.000002 * (zip_num - 43000)
        + np.random.normal(0, 0.01, size=n)
    )
    zipcode_risk = np.clip(zipcode_risk, 0.02, 0.18)

    # Safety reduces risk
    safety_score = (
        0.05 * has_abs
        + 0.07 * has_esc
        + 0.01 * airbag_count
        + 0.04 * has_collision_avoidance
        + 0.03 * has_lane_assist
        + 0.03 * has_blind_spot_monitor
    )

    # Heavy / powerful cars slightly higher frequency risk
    vehicle_power_risk = (horsepower > 220).astype(int) * 0.08
    weight_risk = np.vectorize({"Light": 0.0, "Medium": 0.03, "Heavy": 0.05}.get)(weight_class)

    base_risk_raw = (
        age_risk
        + history_risk
        + mileage_risk
        + claim_history_risk
        + zipcode_risk
        + vehicle_power_risk
        + weight_risk
        + theft_risk_score * 0.05
        - safety_score * 0.5  # safety reduces overall risk
    )

    base_risk = base_risk_raw.clip(0, 1)

    # --------------------------
    # 4. PREMIUM MODEL TARGET
    # --------------------------
    base_premium = (
        300
        + 600 * base_risk
        + (2024 - vehicle_year) * 5
        + driving_history_score * 12
        + claims_history_count * 30
        + np.where(prior_insurance == 0, 60, -30)
        + luxury_brand * 120
        + repair_cost_index * 200
    )

    premium = base_premium + np.random.normal(0, 40, size=n)
    premium = np.clip(premium, 300, None)

    # --------------------------
    # 5. CLAIM MODEL TARGETS
    # --------------------------
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    claim_prob = sigmoid(
        -2.2
        + 3.0 * base_risk
        + 0.4 * (annual_mileage / 10000.0)
        + 0.4 * claim_history_risk
    )

    claim_flag = np.random.binomial(1, claim_prob, size=n)

    # Severity is driven by value + repair cost + luxury
    base_severity = (
        1500.0
        + 0.03 * current_value
        + 2000.0 * repair_cost_index
        + 1000.0 * luxury_brand
    )

    claim_amount = np.where(
        claim_flag == 1,
        np.random.lognormal(
            mean=np.log(base_severity.clip(500, None)),
            sigma=0.5
        ),
        0.0
    )

    # --------------------------
    # 6. RETENTION / CHURN MODEL
    # --------------------------
    churn_prob = sigmoid(
        -1.0
        + 0.003 * (premium - 500.0)           # expensive → churn
        + 0.35 * claim_flag                   # some churn after claim
        - 0.08 * length_of_ownership          # loyalty
        + np.where(prior_insurance == 0, 0.25, -0.1)
        + 0.1 * luxury_brand                  # luxury owners slightly more price-sensitive
    )

    churn_flag = np.random.binomial(1, churn_prob, size=n)

    # --------------------------
    # 7. FINAL DATAFRAME
    # --------------------------
    df = pd.DataFrame({
        # quotation-style inputs
        "driver_age": driver_age,
        "driving_history_score": driving_history_score,
        "annual_mileage": annual_mileage.astype(int),
        "vehicle_year": vehicle_year,
        "ownership_type": ownership_type,
        "length_of_ownership": length_of_ownership.round(1),
        "prior_insurance": prior_insurance,
        "claims_history_count": claims_history_count,
        "zipcode": zipcode,

        # VIN-derived / vehicle features
        "vehicle_region": vehicle_region,
        "body_style": body_style,
        "engine_size_category": engine_size_category,
        "drive_type": drive_type,
        "transmission": transmission,
        "fuel_type": fuel_type,
        "horsepower": horsepower.round(0),
        "weight_class": weight_class,
        "luxury_brand": luxury_brand,

        # safety
        "has_abs": has_abs,
        "has_esc": has_esc,
        "airbag_count": airbag_count,
        "has_collision_avoidance": has_collision_avoidance,
        "has_lane_assist": has_lane_assist,
        "has_blind_spot_monitor": has_blind_spot_monitor,

        # theft & anti-theft
        "has_immobilizer": has_immobilizer,
        "has_anti_theft": has_anti_theft,
        "theft_risk_score": theft_risk_score.round(3),

        # value & repair
        "msrp": msrp.round(0),
        "current_value": current_value.round(0),
        "depreciation_rate": depreciation_rate.round(3),
        "repair_cost_index": repair_cost_index.round(3),

        # model drivers / outputs
        "base_risk": base_risk.round(3),
        "premium": premium.round(2),
        "claim_flag": claim_flag,
        "claim_amount": claim_amount.round(2),
        "churn_flag": churn_flag,
        "claim_prob_true": claim_prob.round(4),
        "churn_prob_true": churn_prob.round(4),
    })

    return df


if __name__ == "__main__":
    df_example = generate_auto_cltv_synthetic(5000)
    print(df_example.head())
    print("Rows:", len(df_example))
    print("Columns:", len(df_example.columns))

    # SAVE TO EXCEL
    df_example.to_excel("synthetic_cltv_data.xlsx", index=False)
