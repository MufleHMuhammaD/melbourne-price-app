"""Melbourne House Price Prediction - Streamlit web app.

Loads the trained scikit-learn pipeline saved by the notebook
(melbourne_price_model.joblib) and predicts the price of a property.
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILE = "melbourne_price_model.joblib"
TYPICAL_ERROR = 303_000   # cross-validation RMSE of the best model (from the notebook, Step 9.2)

# The columns must be in exactly the same order as in the notebook (Step 4)
NUMERIC_FEATURES = ["Rooms", "Distance", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt"]
CATEGORICAL_FEATURES = ["Type", "Suburb"]
TYPE_LABELS = {"House": "h", "Unit / Apartment": "u", "Townhouse": "t"}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def get_suburbs(_model):
    """Read the list of suburbs the model knows from its own one-hot encoder."""
    encoder = _model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    return sorted(encoder.categories_[1])


st.set_page_config(page_title="Melbourne House Price Predictor", page_icon="🏠")
st.title("🏠 Melbourne House Price Predictor")
st.write(
    "Enter the details of a property in Melbourne and the model will predict its "
    "selling price in dollars. The model was chosen with 10-fold cross validation "
    "from 5 scikit-learn regression algorithms."
)

model = load_model()
suburbs = get_suburbs(model)

col1, col2 = st.columns(2)
with col1:
    property_type = st.selectbox("Property type", list(TYPE_LABELS))
    suburb = st.selectbox("Suburb", suburbs, index=suburbs.index("Richmond") if "Richmond" in suburbs else 0)
    rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=8, value=2)
    car_spots = st.number_input("Car spots", min_value=0, max_value=10, value=1)
with col2:
    distance = st.number_input("Distance from city centre (km)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
    landsize = st.number_input("Land size (m²)", min_value=0, max_value=10000, value=450)
    building_area = st.number_input("Building area (m²), 0 if unknown", min_value=0, max_value=1000, value=140)
    year_built = st.number_input("Year built", min_value=1800, max_value=2018, value=1990)

if st.button("Predict price", type="primary"):
    property_details = {
        "Rooms": rooms,
        "Distance": distance,
        "Bathroom": bathrooms,
        "Car": car_spots,
        "Landsize": landsize,
        # In training, a building area of 0 was treated as missing (notebook Step 3.2)
        "BuildingArea": building_area if building_area > 0 else np.nan,
        "YearBuilt": year_built,
        "Type": TYPE_LABELS[property_type],
        "Suburb": suburb,
    }
    feature_vector = pd.DataFrame([property_details])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    predicted_price = model.predict(feature_vector)[0]

    st.success("Prediction complete")
    st.metric("Predicted price", f"${predicted_price:,.0f}")
    st.caption(
        f"Typical error of this model is about ${TYPICAL_ERROR:,.0f} "
        "(cross-validation RMSE), so treat the result as an estimate."
    )
