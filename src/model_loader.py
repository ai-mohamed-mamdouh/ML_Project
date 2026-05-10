import __main__
from functools import lru_cache

import joblib

from src.config import (
    PIPE_LINE_PATH,
    FIRST_MODEL_PATH,
    SECOND_MODEL_PATH,
    LABEL_ENCODER_PATH,
)

from src.preprocessing import (
    Encode,
    feature_extraction,
    feature_selection,
)


# مهم جدًا بسبب أن الـ joblib كان متحفظ من notebook
__main__.Encode = Encode
__main__.feature_extraction = feature_extraction
__main__.feature_selection = feature_selection


@lru_cache(maxsize=1)
def load_artifacts():
    pipe = joblib.load(PIPE_LINE_PATH)
    first_model = joblib.load(FIRST_MODEL_PATH)
    second_model = joblib.load(SECOND_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    return pipe, first_model, second_model, label_encoder