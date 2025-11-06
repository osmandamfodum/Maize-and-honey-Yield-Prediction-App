import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import tensorflow as tf
from PIL import Image
import os
import json

# ------------------- Clear old cache -------------------
st.cache_resource.clear()  # Clears the old model from memory

# ------------------- Suppress TF warnings -------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ------------------- Page Config -------------------
st.set_page_config(page_title="AgriBee AI", layout="wide")

# ------------------- Custom CSS -------------------
st.markdown("""
<style>
    .stApp { background-color: white; color: black; }
    .image-container { background-color: white; padding: 10px; display: flex; justify-content: center; }
    .stImage > img { max-width: 100%; height: auto; }
    .alert-success { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb; }
    .alert-danger { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

# ------------------- Header Image -------------------
st.markdown('<div class="image-container">', unsafe_allow_html=True)
if os.path.exists("neu.jpg"):
    st.image("neu.jpg", use_container_width=True)
else:
    st.markdown("### AgriBee AI")
st.markdown('</div>', unsafe_allow_html=True)

# ------------------- MODE SELECTION -------------------
mode = st.radio("Select Prediction Mode", ["Maize", "Honey", "Bee"], index=0, horizontal=True)

# ------------------- BEE METADATA -------------------
BEE_METADATA = {
    'class_names': [
        'Varroa, Small Hive Beetles',
        'ant problems',
        'few varroa, hive beetles',
        'healthy',
        'hive being robbed',
        'missing queen'
    ],
    'image_size': (224, 224)
}

# Load from JSON if exists
if os.path.exists("bee_metadata.json"):
    try:
        with open("bee_metadata.json") as f:
            loaded = json.load(f)
            BEE_METADATA.update(loaded)
    except:
        pass

# ------------------- DIAGNOSIS TEXT -------------------
DIAGNOSIS_EXPLANATIONS = {
    'Varroa, Small Hive Beetles': "Small red or brown spots detected, indicating Varroa mite and hive beetle infestation.",
    'ant problems': "Ants detected, indicating ant-related issues in the hive.",
    'few varroa, hive beetles': "Low levels of Varroa mites and hive beetles detected.",
    'healthy': "No visible disease signs; the hive appears healthy.",
    'hive being robbed': "Signs of robbing detected, such as increased aggression or dead bees.",
    'missing queen': "Evidence of a missing queen, such as absence of eggs or queen cells."
}

FALLBACK_RESPONSES = {
    'Varroa, Small Hive Beetles': {
        'treatment': "- Treat with oxalic acid or amitraz.\n- Monitor mite and beetle populations.\n- Rotate treatments.",
        'management': "- Inspect pest levels monthly.\n- Use sticky boards."
    },
    'ant problems': {
        'treatment': "- Use ant barriers or bait traps.\n- Reduce hive entrances.\n- Remove attractants.",
        'management': "- Monitor ant activity weekly.\n- Keep the hive area clean."
    },
    'few varroa, hive beetles': {
        'treatment': "- Light treatment (e.g., thymol).\n- Use beetle traps.\n- Monitor levels.",
        'management': "- Regular hive inspections.\n- Maintain hive cleanliness."
    },
    'healthy': {
        'care': "- Provide balanced nutrition.\n- Maintain hive cleanliness.\n- Monitor queen health.",
        'next steps': "- Regular hive inspections.\n- Ensure colony strength."
    },
    'hive being robbed': {
        'management': "- Reduce hive entrances.\n- Relocate weak hives.\n- Provide supplemental feeding.",
        'treatment': "- Install anti-robbing screens.\n- Monitor aggression."
    },
    'missing queen': {
        'treatment': "- Introduce a new queen.\n- Check for queen cells.\n- Merge with a queenright colony.",
        'management': "- Monitor egg-laying patterns.\n- Ensure colony stability."
    }
}

# ------------------- LOAD BEE MODEL (Force reload) -------------------
@st.cache_resource
def load_bee_model(_model_path="bee_224_model2.h5"):
    if not os.path.exists(_model_path):
        st.error("Model file missing: bee_224_model.h5")
        st.stop()
    try:
        model = tf.keras.models.load_model(_model_path, compile=False)
        # Verify input shape
        if model.input_shape != (None, 224, 224, 3):
            st.error(f"Error: Input shape {model.input_shape}, expected (224,224,3)")
            st.stop()
        st.success("Bee model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ------------------- MAIZE / HONEY MODE -------------------
if mode in ["Maize", "Honey"]:
    has_classification = False
    if mode == "Maize":
        try:
            reg_model = joblib.load('us_maize_yield_regressor.pkl')
            preprocessor = joblib.load('preprocessor.pkl')
            historical_df = pd.read_csv('processed_us_maize_data.csv')
            st.success("Maize models loaded")
        except Exception as e:
            st.error(f"Error loading maize models: {e}")
            st.stop()
        try:
            clf_model = joblib.load('us_maize_yield_classifier.pkl')
            label_encoder = joblib.load('label_encoder.pkl')
            if len(preprocessor.get_feature_names_out()) == clf_model.n_features_in_:
                has_classification = True
                st.success("Maize classifier ready")
        except:
            st.warning("Maize classifier not available")
    else:  # Honey
        try:
            reg_model = joblib.load('honey_yield_regressor2.pkl')
            preprocessor = joblib.load('honey_preprocessor2.pkl')
            clf_model = joblib.load('honey_yield_classifier2.pkl')
            label_encoder = joblib.load('honey_label_encoder2.pkl')
            historical_df = pd.read_csv('merged_honey_weather.csv')
            has_classification = True
            st.success("Honey models loaded")
        except Exception as e:
            st.error(f"Error loading honey models: {e}")
            st.stop()

    st.title(f"{mode} Yield Prediction")
    if mode == "Maize":
        country = st.selectbox("Country", ["US"])
        crop_type = st.selectbox("Crop Type", ["Maize"])
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
        year = st.number_input("Year", 2000, 2050, 2025)
        area = st.number_input("Area (hectares)", 0.0, value=100.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, value=500.0)
        temp = st.number_input("Temperature (°C)", 0.0, value=20.0)
        tmin = st.number_input("Minimum Temperature (°C)", 0.0, value=15.0)
        tmax = st.number_input("Maximum Temperature (°C)", 0.0, value=25.0)
        rad = st.number_input("Solar Radiation (MJ/m²)", 0.0, value=15.0)
        et0 = st.number_input("Evapotranspiration (mm)", 0.0, value=5.0)
        cwb = st.number_input("Water Balance (mm)", -1000.0, value=0.0)
    else:
        state = st.selectbox("State", historical_df['state'].unique())
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
        year = st.number_input("Year", 1995, 2021, 2020)
        colonies_number = st.number_input("Number of Colonies", 0, value=50000)
        avg_temp = st.number_input("Average Temperature (°C)", 0.0, value=15.0)
        total_rainfall = st.number_input("Total Rainfall (mm)", 0.0, value=500.0)

    if st.button("Calculate Yield"):
        try:
            if mode == "Maize":
                input_data = pd.DataFrame({
                    'country': [country], 'crop_type': [crop_type], 'season': [season],
                    'year': [year], 'area': [area], 'rainfall': [rainfall], 'temp': [temp],
                    'tmin': [tmin], 'tmax': [tmax], 'rad': [rad], 'et0': [et0], 'cwb': [cwb]
                })
            else:
                input_data = pd.DataFrame({
                    'state': [state], 'season': [season], 'year': [year],
                    'colonies_number': [colonies_number], 'avg_temp': [avg_temp],
                    'total_rainfall': [total_rainfall]
                })
            X = preprocessor.transform(input_data)
            pred = reg_model.predict(X)[0]
            unit = 'tons/hectare' if mode == 'Maize' else 'pounds/colony'
            st.success(f"**Predicted Yield**: {pred:.2f} {unit}")
            total = pred * area if mode == "Maize" else pred * colonies_number / 1000
            st.success(f"**Total Yield**: {total:.2f} tons")
            if has_classification:
                cat = label_encoder.inverse_transform(clf_model.predict(X))[0]
                color = "#ccffcc" if cat == 'High' else "#ffcccc" if cat == 'Low' else "#fff3cd"
                border = "#44ff44" if cat == 'High' else "#ff4444" if cat == 'Low' else "#ffc107"
                st.markdown(f'<div style="background-color:{color}; padding:10px; border-radius:5px; border-left:5px solid {border};">'
                            f'<strong>Yield Category: {cat}</strong></div>', unsafe_allow_html=True)
            y_col = 'yield' if mode == 'Maize' else 'yield_per_colony'
            hist = historical_df.groupby('year')[y_col].mean().reset_index()
            fig1 = px.line(hist, x='year', y=y_col, title=f"Historical Yield Trend for {mode}")
            fig1.add_scatter(x=[year], y=[pred], mode='markers', name='Predicted', marker=dict(color='red', size=12))
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"Calculation failed: {e}")

# ------------------- BEE DISEASE MODE -------------------
else:
    st.title("Bee Hive Disease Diagnosis")
    st.write("Upload a hive image to detect diseases and receive treatment recommendations.")
    bee_model = load_bee_model()
    uploaded_file = st.file_uploader("Upload Hive Image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file and st.button("Analyze Hive"):
        try:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded Image", use_column_width=True)
            # Process image
            img_resized = img.resize(BEE_METADATA['image_size'])
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            # Prediction
            preds = bee_model.predict(img_array, verbose=0)[0]
            idx = np.argmax(preds)
            confidence = float(preds[idx])
            diagnosis = BEE_METADATA['class_names'][idx]
            # Confidence alerts
            if confidence >= 0.85:
                st.success(f"High Confidence: {diagnosis} ({confidence*100:.1f}%)")
            elif confidence >= 0.70:
                st.warning(f"Medium Confidence: {diagnosis} ({confidence*100:.1f}%). Check image quality.")
            else:
                st.error(f"Low Confidence: {diagnosis} ({confidence*100:.1f}%). Try a clearer image.")
            # Explanation and recommendations
            st.write("**Explanation:**", DIAGNOSIS_EXPLANATIONS.get(diagnosis, "Unknown"))
            st.subheader("Recommended Actions")
            for key, text in FALLBACK_RESPONSES.get(diagnosis, {}).items():
                st.write(f"**{key.capitalize()}**:")
                st.write(text.replace('\n', '<br>'), unsafe_allow_html=True)
            # Plot confidence chart
            conf_df = pd.DataFrame({'Condition': BEE_METADATA['class_names'], 'Confidence': preds})
            fig = px.bar(conf_df, x='Condition', y='Confidence', range_y=[0, 1], title="Diagnosis Confidence Levels")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Analysis failed: {e}")

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.header("AgriBee AI")
    st.write("**Features:**")
    st.write("- Maize and Honey yield prediction")
    st.write("- Bee hive disease diagnosis")
    st.write("**Models:** XGBoost + MobileNetV2")
    st.write("**Data:** USDA, weather, 5000+ hive images")
    st.write("Built with **Streamlit**")
