import sys
import __main__
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.preprocessing import Encode, feature_extraction, feature_selection

__main__.Encode = Encode
__main__.feature_extraction = feature_extraction
__main__.feature_selection = feature_selection
IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "pipeline.png"
from src.inference import predict_csv

st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌱",
    layout="wide",
)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(
        str(IMAGE_PATH),
        caption="Model Pipeline",
        width=400
    )

st.title("🌱 Crop Recommendation System")
st.write("Upload CSV file and get prediction results.")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.astype(str).str.strip()
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        st.subheader("Uploaded Data")
        st.dataframe(df, width="stretch")

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                result_df = predict_csv(df)

            st.subheader("🌾 Final Prediction Results")
            st.dataframe(result_df, width="stretch")

            if "label_name" in result_df.columns:
                st.markdown("---")
                st.subheader("📌 Summary")

                total_rows = len(result_df)
                most_common_crop = result_df["label_name"].mode()[0]
                most_common_count = result_df["label_name"].value_counts().iloc[0]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Samples", total_rows)

                with col2:
                    st.metric("Most Recommended Crop", most_common_crop)

                with col3:
                    st.metric("Count", most_common_count)

                st.markdown("---")
                st.subheader("🌱 Recommendations Count")

                counts_df = (
                    result_df["label_name"]
                    .value_counts()
                    .reset_index()
                )

                counts_df.columns = ["Crop", "Count"]

                st.dataframe(counts_df, width="stretch")

                st.bar_chart(
                    counts_df,
                    x="Crop",
                    y="Count"
                )

                st.markdown("---")
                st.subheader("✅ Final Output")

                for crop, count in result_df["label_name"].value_counts().items():
                    st.success(f"{crop}: {count} sample(s)")

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv,
                file_name="crop_prediction_results.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

else:
    st.info("Please upload a CSV file.")