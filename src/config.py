from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

PIPE_LINE_PATH = MODELS_DIR / "crop_recommendation_pipeline(Preprocessing).joblib"
FIRST_MODEL_PATH = MODELS_DIR / "first_model.joblib"
SECOND_MODEL_PATH = MODELS_DIR / "second_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder(target).joblib"
ONE_HOT_ENCODE_PATH = MODELS_DIR / "onehot_encoder(features).joblib"