# ------------------------------------------------------------
# app.py – AgriBee AI (Maize, Honey, Bee Disease)
# 100% compatible with your newly trained bee_224_model.h5
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import tensorflow as tf
from PIL import Image
import os
import json
st.cache_resource.clear()
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
    st.image("neu.jpg", use_column_width=True)
else:
    st.markdown("### AgriBee AI")
st.markdown('</div>', unsafe_allow_html=True)

# ------------------- MODE SELECTION -------------------
mode = st.radio("Select Prediction Mode", ["Maize", "Honey", "Bee"], index=0, horizontal=True)

# ------------------- BEE METADATA (Load from JSON if exists) -------------------
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

if os.path.exists("bee_metadata.json"):
    try:
        with open("bee_metadata.json") as f:
            loaded = json.load(f)
            BEE_METADATA.update(loaded)
        st.sidebar.success("Bee metadata loaded from JSON")
    except:
        st.sidebar.warning("Using default metadata")

# ------------------- DIAGNOSIS TEXT -------------------
DIAGNOSIS_EXPLANATIONS = {
    'Varroa, Small Hive Beetles': "Small red or brown spots detected, indicating Varroa Mites and Small Hive Beetles infestation.",
    'ant problems': "Presence of ants detected, suggesting ant-related issues in the hive.",
    'few varroa, hive beetles': "Low levels of Varroa Mites and Small Hive Beetles detected.",
    'healthy': "No visible signs of disease; the hive appears healthy.",
    'hive being robbed': "Signs of robbing behavior detected, such as increased aggression or dead bees.",
    'missing queen': "Indications of a missing queen, such as lack of brood or queen cells."
}

FALLBACK_RESPONSES = {
    'Varroa, Small Hive Beetles': {
        'treatment': "- Oxalic acid or amitraz treatment.\n- Monitor mite and beetle counts.\n- Rotate treatments.",
        'management': "- Check pest levels monthly.\n- Use sticky boards."
    },
    'ant problems': {
        'treatment': "- Use ant barriers or bait traps.\n- Reduce hive entrances.\n- Remove attractants.",
        'management': "- Monitor ant activity weekly.\n- Keep hive area clean."
    },
    'few varroa, hive beetles': {
        'treatment': "- Consider light treatment (e.g., thymol).\n- Use beetle traps.\n- Monitor levels.",
        'management': "- Regular hive inspections.\n- Maintain hive hygiene."
    },
    'healthy': {
        'care': "- Provide balanced nutrition.\n- Maintain clean hives.\n- Monitor queen health.",
        'next steps': "- Regular hive inspections.\n- Ensure strong colony."
    },
    'hive being robbed': {
        'management': "- Reduce hive entrances.\n- Move weaker hives.\n- Provide supplemental feeding.",
        'treatment': "- Install robber screens.\n- Monitor for aggression."
    },
    'missing queen': {
        'treatment': "- Introduce a new queen.\n- Check for queen cells.\n- Unite with a queenright colony.",
        'management': "- Monitor brood patterns.\n- Ensure colony stability."
    }
}

# ------------------- LOAD BEE MODEL (SINGLE INPUT ONLY) -------------------
@st.cache_resource
def load_bee_model():
    model_path = 'bee_224_model2.h5'
    if not os.path.exists(model_path):
        st.error("Bee model not found: 'bee_224_model.h5' missing.")
        st.stop()

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # --- تأكد من أن النموذج له مدخل واحد فقط ---
        if isinstance(model.inputs, list):
            if len(model.inputs) != 1:
                st.error(f"Model has {len(model.inputs)} inputs. Expected 1.")
                st.stop()
        else:
            if model.input_shape != (None, 224, 224, 3):
                st.warning(f"Unexpected input shape: {model.input_shape}")

        st.success("Bee model loaded successfully (224×224, single input)")
        return model

    except Exception as e:
        st.error(f"Failed to load bee model: {e}")
        st.stop()

# ------------------- MAIZE / HONEY MODE -------------------
if mode in ["Maize", "Honey"]:
    has_classification = False

    if mode == "Maize":
        try:
            reg_model = joblib.load('us_maize_yield_regressor.pkl')
            preprocessor = joblib.load('preprocessor.pkl')
            historical_df = pd.read_csv('processed_us_maize_data.csv')
            st.success("Maize models loaded.")
        except Exception as e:
            st.error(f"Maize load error: {e}")
            st.stop()

        try:
            clf_model = joblib.load('us_maize_yield_classifier.pkl')
            label_encoder = joblib.load('label_encoder.pkl')
            if len(preprocessor.get_feature_names_out()) == clf_model.n_features_in_:
                has_classification = True
                st.success("Maize classifier ready.")
        except:
            st.warning("Maize classifier unavailable.")

    else:  # Honey
        try:
            reg_model = joblib.load('honey_yield_regressor2.pkl')
            preprocessor = joblib.load('honey_preprocessor2.pkl')
            clf_model = joblib.load('honey_yield_classifier2.pkl')
            label_encoder = joblib.load('honey_label_encoder2.pkl')
            historical_df = pd.read_csv('merged_honey_weather.csv')
            has_classification = True
            st.success("Honey models loaded.")
        except Exception as e:
            st.error(f"Honey load error: {e}")
            st.stop()

    st.title(f"{mode} Yield Prediction")
    if mode == "Maize":
        country = st.selectbox("Country", ["US"])
        crop_type = st.selectbox("Crop Type", ["Maize"])
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
        year = st.number_input("Year", 2000, 2050, 2025)
        area = st.number_input("Area (ha)", 0.0, value=100.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, value=500.0)
        temp = st.number_input("Temperature (°C)", 0.0, value=20.0)
        tmin = st.number_input("Min Temp (°C)", 0.0, value=15.0)
        tmax = st.number_input("Max Temp (°C)", 0.0, value=25.0)
        rad = st.number_input("Solar Radiation (MJ/m²)", 0.0, value=15.0)
        et0 = st.number_input("Evapotranspiration (mm)", 0.0, value=5.0)
        cwb = st.number_input("Climatic Water Balance (mm)", -1000.0, value=0.0)
    else:
        state = st.selectbox("State", historical_df['state'].unique())
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
        year = st.number_input("Year", 1995, 2021, 2020)
        colonies_number = st.number_input("Number of Colonies", 0, value=50000)
        avg_temp = st.number_input("Average Temperature (°C)", 0.0, value=15.0)
        total_rainfall = st.number_input("Total Rainfall (mm)", 0.0, value=500.0)

    if st.button("Predict Yield"):
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
            unit = 't/ha' if mode == 'Maize' else 'lbs/colony'
            st.success(f"**Predicted Yield**: {pred:.2f} {unit}")

            total = pred * area if mode == "Maize" else pred * colonies_number / 1000
            st.success(f"**Total {'Crop' if mode == 'Maize' else 'Honey'}**: {total:.2f} tons")

            if has_classification:
                cat = label_encoder.inverse_transform(clf_model.predict(X))[0]
                color = "#ccffcc" if cat == 'High' else "#ffcccc" if cat == 'Low' else "#fff3cd"
                border = "#44ff44" if cat == 'High' else "#ff4444" if cat == 'Low' else "#ffc107"
                st.markdown(f'<div style="background-color:{color}; padding:10px; border-radius:5px; border-left:5px solid {border};">'
                            f'<strong>Yield Category: {cat}</strong></div>', unsafe_allow_html=True)

            y_col = 'yield' if mode == 'Maize' else 'yield_per_colony'
            hist = historical_df.groupby('year')[y_col].mean().reset_index()
            fig1 = px.line(hist, x='year', y=y_col, title=f"Historical {mode} Yield")
            fig1.add_scatter(x=[year], y=[pred], mode='markers', name='Predicted', marker=dict(color='red', size=12))
            st.plotly_chart(fig1, use_container_width=True)

            mean_y = historical_df[y_col].mean()
            comp = pd.DataFrame({'Type': ['Historical Avg', 'Predicted'], 'Yield': [mean_y, pred]})
            st.plotly_chart(px.bar(comp, x='Type', y='Yield', color='Type'), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------------- BEE DISEASE MODE -------------------
else:
    st.title("Bee Hive Disease Diagnosis")
    st.write("Upload a hive image to detect diseases and get management advice.")

    bee_model = load_bee_model()
    uploaded_file = st.file_uploader("Upload Hive Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file and st.button("Analyze Hive"):
        try:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Preprocess
            img_resized = img.resize(BEE_METADATA['image_size'])
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

            # Predict
            preds = bee_model.predict(img_array, verbose=0)[0]
            idx = np.argmax(preds)
            confidence = float(preds[idx])
            diagnosis = BEE_METADATA['class_names'][idx]

            # --- Confidence Alerts ---
            if confidence >= 0.85:
                st.success(f"موثوق: {diagnosis} ({confidence*100:.1f}%)")
            elif confidence >= 0.70:
                st.warning(f"ثقة متوسطة: {diagnosis} ({confidence*100:.1f}%). تحقق من جودة الصورة.")
            else:
                st.error(f"ثقة منخفضة: {diagnosis} ({confidence*100:.1f}%). جرب صورة أوضح.")

            # --- Explanation & Advice ---
            st.write("**Explanation:**", DIAGNOSIS_EXPLANATIONS.get(diagnosis, "غير معروف"))

            st.subheader("Recommended Actions")
            for key, text in FALLBACK_RESPONSES.get(diagnosis, {}).items():
                st.write(f"**{key.capitalize()}**:")
                st.write(text.replace('\n', '<br>'), unsafe_allow_html=True)

            # --- Confidence Chart ---
            conf_df = pd.DataFrame({'Condition': BEE_METADATA['class_names'], 'Confidence': preds})
            fig = px.bar(conf_df, x='Condition', y='Confidence', range_y=[0, 1], title="Diagnosis Confidence")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Analysis failed: {e}")

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.header("AgriBee AI")
    st.write("**Features:**")
    st.write("- Maize & Honey Yield Prediction")
    st.write("- Bee Hive Disease Diagnosis")
    st.write("**Models:** XGBoost + MobileNetV2")
    st.write("**Data:** USDA, Weather, 5k+ Hive Images")
    st.write("Built with **Streamlit**")
