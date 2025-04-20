import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from utils_streamlit import load_model, predict_image
import pandas as pd

st.set_page_config(page_title="🍱 Food Calorie Estimator", layout="centered")

# Title
st.title("🍽️ Food Classifier & Calorie Estimator")

# Sidebar
st.sidebar.header("Upload a Food Image")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
import pandas as pd

# Load food → calorie mapping
@st.cache_data
def load_calorie_map():
    df = pd.read_csv("/Users/nileshupraity/Python projects/foodcalorie/food-calorie-estimator/food_calorie_map.csv")
    return dict(zip(df["food"], df["calories"]))

calorie_map = load_calorie_map()

# Load model
@st.cache_resource
def get_model():
    return load_model("models/debug_efficientnetv2_food101.pth")

model, class_names = get_model()

# Image display + prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.markdown("---")
    st.subheader("🔍 Prediction")

    label, prob = predict_image(image, model, class_names)
    st.success(f"🍛 Predicted: **{label}** with {prob:.2f}% confidence")

    # (Optional) Calorie Estimation Table
    calorie_map = {
        "pizza": 266, "ice_cream": 207, "samosa": 308,
        "hamburger": 295, "caesar_salad": 190, "chicken_curry": 240,
        # Add more if you want...
    }

    if label in calorie_map:
        calories = calorie_map[label]
        st.info(f"Estimated Calories: **{calorie_map[label]} kcal** per 100g (approx)")
    else:
        st.warning("No calorie info available for this item yet.")
