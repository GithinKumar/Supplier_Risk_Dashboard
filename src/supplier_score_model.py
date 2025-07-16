import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def generate_supplier_score(
    overview_path: str,
    financial_score_path: str,
    master_data_path: str,
    output_path: str
) -> None:
    # Load data
    overview_df = pd.read_csv(overview_path)
    financial_df_raw = pd.read_csv(financial_score_path)
    required_cols = {"Supplier ID", "Quarter", "Financial Risk Score"}
    missing_cols = required_cols - set(financial_df_raw.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in financial_score_path: {missing_cols}")
    financial_df = financial_df_raw[["Supplier ID", "Quarter", "Financial Risk Score"]]
    master_df = pd.read_csv(master_data_path)

    # Merge on both Supplier ID and Quarter to retain quarter info
    df = overview_df.merge(financial_df, on=["Supplier ID"], how="left")
    df = df.merge(master_df[["Supplier ID", "Tier"]], on=["Supplier ID"], how="left")

    # Use Quarter from finacial_df
    df["Quarter"] = df["Quarter"]

    # Label encode Tier
    df["Tier"] = LabelEncoder().fit_transform(df["Tier"].astype(str))

    # Feature engineering: feature interactions
    df["delay_defect_interaction"] = df["pct_defect_rate"] * df["%_delayed"]
    df["delay_defect_interaction"] = df["pct_defect_rate"] * df["%_delayed"]
    df["volume_delay_interaction"] = df["avg_shipment_volume"] * df["avg_delay"]
    df["volume_lost_interaction"] = df["avg_shipment_volume"] * df["pct_shipment_lost"]
    df["defect_lost_interaction"] = df["pct_defect_rate"] * df["pct_shipment_lost"]

    # Optional: add log transform for skewed features
    df["log_avg_delay"] = np.log1p(df["avg_delay"])
    df["log_shipment_volume"] = np.log1p(df["avg_shipment_volume"])

    # Define features and target
    feature_cols = [
        "%_on_time", "%_delayed", "avg_delay", "avg_shipment_volume",
        "pct_shipment_lost", "pct_defect_rate", "Financial Risk Score", "Tier",
        "delay_defect_interaction", "volume_delay_interaction",
        "volume_lost_interaction", "defect_lost_interaction",
        "log_avg_delay", "log_shipment_volume"
    ]

    X = df[feature_cols]

    # Dummy target for training (simulating supervised learning)
    y = (
        0.3 * (100 - df["%_delayed"]) +
        0.2 * (100 - df["avg_delay"]) +
        0.2 * (100 - df["pct_shipment_lost"] * 100) +
        0.1 * (100 - df["pct_defect_rate"] * 100) +
        0.2 * (100 - df["Financial Risk Score"])
    )

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    # Predict and save
    df["Supplier Score"] = model.predict(scaler.transform(X))
    # Now include Quarter in the output
    df[["Supplier ID", "Quarter", "Supplier Score"]].to_csv(output_path, index=False)

    print(f"Supplier scores saved to {output_path}")