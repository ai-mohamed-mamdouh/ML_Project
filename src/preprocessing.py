import joblib
import numpy as np
import pandas as pd

from src.config import ONE_HOT_ENCODE_PATH


def feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    eps = 1e-9

    X["NPK_sum"] = X["N"] + X["P"] + X["K"]
    X["N_ratio"] = X["N"] / (X["NPK_sum"] + eps)
    X["P_ratio"] = X["P"] / (X["NPK_sum"] + eps)
    X["K_ratio"] = X["K"] / (X["NPK_sum"] + eps)

    X["N_to_P"] = X["N"] / (X["P"] + 1.0)
    X["N_to_K"] = X["N"] / (X["K"] + 1.0)
    X["P_to_K"] = X["P"] / (X["K"] + 1.0)

    X["NPK_balance"] = X[["N", "P", "K"]].max(axis=1) - X[["N", "P", "K"]].min(axis=1)

    X["pH_distance"] = (X["Soil_pH"] - 7.0).abs()
    X["EC_log"] = np.log1p(X["Electrical_Conductivity"].clip(lower=0))

    X["Fertility_index"] = X["Organic_Carbon"] * (X["NPK_sum"] + eps)
    X["OC_to_NPK"] = X["Organic_Carbon"] / (X["NPK_sum"] + 1.0)

    X["Moisture_Rain"] = X["Soil_Moisture"] * X["Rainfall"]
    X["Temp_Humidity"] = X["Temperature"] * X["Humidity"]
    X["Heat_Load"] = X["Temperature"] * X["Sunlight_Hours"]
    X["SunRain_ratio"] = X["Sunlight_Hours"] / (X["Rainfall"] + 1.0)
    X["Dryness_Index"] = X["Temperature"] / (X["Humidity"] + 1.0)

    X["Altitude_Temp"] = X["Altitude"] * X["Temperature"]

    if "Fertilizer_Used" in X.columns and pd.api.types.is_numeric_dtype(X["Fertilizer_Used"]):
        X["Fert_NPK_interaction"] = X["Fertilizer_Used"] * (X["NPK_sum"] + eps)

    return X


def Encode(df_origin: pd.DataFrame) -> pd.DataFrame:
    loaded_encoder = joblib.load(ONE_HOT_ENCODE_PATH)

    X = df_origin.copy()
    X.columns = X.columns.astype(str).str.strip()

    cat_cols = [
        "Soil_Type",
        "Region",
        "Season",
        "Irrigation_Type",
        "Previous_Crop",
    ]

    for col in cat_cols:
        X[col] = X[col].astype(str).str.strip()

    df_num = X.drop(columns=cat_cols)
    df_cat = X[cat_cols]

    df_cat_encoder_array = loaded_encoder.transform(df_cat)

    if hasattr(df_cat_encoder_array, "toarray"):
        df_cat_encoder_array = df_cat_encoder_array.toarray()

    df_encoder_dataFrame = pd.DataFrame(
        df_cat_encoder_array,
        columns=loaded_encoder.get_feature_names_out(cat_cols),
        index=X.index,
    )

    return pd.concat([df_num, df_encoder_dataFrame], axis=1)


def feature_selection(feature_with_extract: pd.DataFrame) -> pd.DataFrame:
    return feature_with_extract[
        [
            "Temperature",
            "SunRain_ratio",
            "Moisture_Rain",
            "Dryness_Index",
            "Heat_Load",
            "Temp_Humidity",
            "Soil_pH",
            "N",
            "Altitude_Temp",
            "pH_distance",
            "Season_Rabi",
            "N_ratio",
            "N_to_P",
            "NPK_balance",
            "NPK_sum",
            "K_ratio",
            "N_to_K",
            "Soil_Type_Sandy",
            "OC_to_NPK",
        ]
    ]