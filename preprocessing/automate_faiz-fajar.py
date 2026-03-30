# =========================================
# AUTOMATION SCRIPT - DATA PREPROCESSING
# Fatalities Israeli-Palestinian Conflict
# =========================================

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# Helper Function
# =========================
def normalize_target(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if x in ['yes', 'y', 'true', '1']:
        return 1
    if x in ['no', 'n', 'false', '0']:
        return 0
    return np.nan


# =========================
# Main Preprocessing Function
# =========================
def preprocess_data(input_path, output_dir="preprocessing_output"):
    print("=== MULAI PREPROCESSING ===")

    # Load data
    df = pd.read_csv(input_path)
    print(f"Data loaded: {df.shape}")

    # Standarisasi kolom
    df.columns = df.columns.str.strip().str.lower()

    # Hapus duplikat
    df = df.drop_duplicates()

    # =========================
    # DATE PROCESSING
    # =========================
    df['date_of_event'] = pd.to_datetime(df['date_of_event'], errors='coerce')
    if 'date_of_death' in df.columns:
        df['date_of_death'] = pd.to_datetime(df['date_of_death'], errors='coerce')

    df['event_year'] = df['date_of_event'].dt.year
    df['event_month'] = df['date_of_event'].dt.month

    if 'date_of_death' in df.columns:
        df['death_delay_days'] = (
            df['date_of_death'] - df['date_of_event']
        ).dt.days.fillna(0)

    # =========================
    # TARGET CLEANING
    # =========================
    df['took_part_in_the_hostilities'] = df[
        'took_part_in_the_hostilities'
    ].apply(normalize_target)

    df = df[df['took_part_in_the_hostilities'].isin([0, 1])].copy()

    # =========================
    # NUMERIC PROCESSING
    # =========================
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'] = df['age'].fillna(df['age'].median())

        # Clipping outlier
        df['age'] = df['age'].clip(1, 90)

    # =========================
    # CATEGORICAL PROCESSING
    # =========================
    cat_cols = [
        'gender',
        'citizenship',
        'event_location_region',
        'event_location_district',
        'type_of_injury',
        'killed_by',
        'place_of_residence_district',
        'ammunition'
    ]

    cat_cols = [c for c in cat_cols if c in df.columns]

    for col in cat_cols:
        df[col] = df[col].fillna('Unknown').astype(str).str.strip()

    # =========================
    # DROP UNUSED COLUMNS
    # =========================
    drop_cols = [
        'name',
        'notes',
        'date_of_event',
        'date_of_death',
        'event_location',
        'place_of_residence',
        'year',
        'month',
        'event_dayofweek',
        'death_delay_days'
    ]

    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # =========================
    # SPLIT X & y
    # =========================
    target = 'took_part_in_the_hostilities'
    X = df.drop(columns=[target])
    y = df[target]

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # ENCODING
    # =========================
    ohe = OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'
    )

    ohe.fit(X_train[cat_cols])

    X_train_enc = ohe.transform(X_train[cat_cols])
    X_test_enc = ohe.transform(X_test[cat_cols])

    X_train_enc = pd.DataFrame(
        X_train_enc,
        columns=ohe.get_feature_names_out(cat_cols),
        index=X_train.index
    )

    X_test_enc = pd.DataFrame(
        X_test_enc,
        columns=ohe.get_feature_names_out(cat_cols),
        index=X_test.index
    )

    X_train = X_train.drop(columns=cat_cols).join(X_train_enc)
    X_test = X_test.drop(columns=cat_cols).join(X_test_enc)

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()
    numeric_cols = ['age']
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # =========================
    # SAVE OUTPUT
    # =========================
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    joblib.dump(ohe, output_dir / "onehot_encoder.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")

    df.to_csv(output_dir / "final_dataset.csv", index=False)

    print("\n[INFO] Shape Final:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    print("=== PREPROCESSING SELESAI ===")
    print(f"Output tersimpan di folder: {output_dir}")


# =========================
# RUN SCRIPT
# =========================
if __name__ == "__main__":
    input_file = "../raw_dataset.csv"  # ganti sesuai lokasi dataset
    preprocess_data(input_file)