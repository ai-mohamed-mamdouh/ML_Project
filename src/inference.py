import numpy as np
import pandas as pd

from src.model_loader import load_artifacts


PIPE_COLUMNS = [
    "N",
    "P",
    "K",
    "Soil_pH",
    "Soil_Moisture",
    "Soil_Type",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Sunlight_Hours",
    "Wind_Speed",
    "Region",
    "Altitude",
    "Season",
    "Irrigation_Type",
    "Fertilizer_Used",
    "Previous_Crop",
]

CAT_COLS = [
    "Soil_Type",
    "Region",
    "Season",
    "Irrigation_Type",
    "Previous_Crop",
]

LD_COLUMNS = [
    "LD1",
    "LD2",
    "LD3",
    "LD4",
    "LD5",
    "LD6",
    "LD7",
    "LD8",
    "LD9",
]


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    missing_cols = [col for col in PIPE_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df = df[PIPE_COLUMNS].copy()

    df = df.replace(
        ["", " ", "None", "none", "NULL", "null", "NaN", "nan"],
        np.nan,
    )

    for col in PIPE_COLUMNS:
        if col in CAT_COLS:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def transform_with_pipe(df: pd.DataFrame, pipe) -> pd.DataFrame:
    transformed = pipe.transform(df)

    if isinstance(transformed, pd.DataFrame):
        transformed_df = transformed.copy()
    else:
        transformed = np.asarray(transformed)

        if transformed.ndim == 1:
            transformed = transformed.reshape(1, -1)

        transformed_df = pd.DataFrame(transformed, columns=LD_COLUMNS)

    transformed_df = transformed_df[LD_COLUMNS].copy()
    transformed_df = transformed_df.fillna(0)

    return transformed_df


def predict_one_row(row_df: pd.DataFrame) -> dict:
    pipe, first_model, second_model, _ = load_artifacts()

    first_label_map = {
        6: "Rice",
        9: "Wheat",
        -1: "Other",
    }

    second_label_map = {
        0: "Broadleaf crops",
        1: "Millet",
        2: "Sugarcane",
    }

    transformed_df = transform_with_pipe(row_df, pipe)

    first_pred = int(np.asarray(first_model.predict(transformed_df)).ravel()[0])
    first_name = first_label_map.get(first_pred, "Unknown")

    if first_pred != -1:
        return {
            "model_used": "first_model",
            "label": first_pred,
            "label_name": first_name,
        }

    second_pred = int(np.asarray(second_model.predict(transformed_df)).ravel()[0])
    second_name = second_label_map.get(second_pred, "Unknown")

    return {
        "model_used": "second_model",
        "label": second_pred,
        "label_name": second_name,
    }


def predict_csv(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = clean_input(df)

    results = []

    for i in range(len(clean_df)):
        row_df = clean_df.iloc[[i]].copy()
        pred = predict_one_row(row_df)
        results.append(pred)

    result_df = pd.concat(
        [
            clean_df.reset_index(drop=True),
            pd.DataFrame(results).reset_index(drop=True),
        ],
        axis=1,
    )

    return result_df