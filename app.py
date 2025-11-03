import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import tensorflow as tf
from PIL import Image
import os
import json
import logging

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Custom DepthwiseConv2D layer
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# --- CONFIG ---
st.set_page_config(page_title="AgriBee AI", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: white; color: black; }
    .image-container { background-color: white; padding: 10px; display: flex; justify-content: center; }
    .stImage > img { max-width: 100%; height: auto; }
    .alert-success { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb; }
    .alert-danger { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

# Header Image
st.markdown('<div class="image-container">', unsafe_allow_html=True)
st.image("neu.jpg", use_column_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- MODE SELECTION ---
mode = st.radio("Select Prediction Mode", ["Maize", "Honey", "Bee"], index=0, horizontal=True)

# --- BEE MODEL & METADATA (Only load once) ---
@st.cache_resource
def load_bee_model():
    model_path = 'bee_224_model.h5'
    if not os.path.exists(model_path):
        st.error("Bee model not found. Place 'Final_bee_model.h5' in the root directory.")
        st.stop()
    model = tf.keras.models.load_model(model_path)
    return model

BEE_METADATA = {
    'class_names': [
        'Varroa, Small Hive Beetles', 'ant problems', 'few varroa, hive beetles',
        'healthy', 'hive being robbed', 'missing queen'
    ],
    'image_size': (224, 224)
}

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

# --- MAIZE / HONEY MODE ---
if mode in ["Maize", "Honey"]:
    has_classification = False

    if mode == "Maize":
        try:
            reg_model = joblib.load('us_maize_yield_regressor.pkl')
            preprocessor = joblib.load('preprocessor.pkl')
            historical_df = pd.read_csv('processed_us_maize_data.csv')
            st.success("Maize model and data loaded.")
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        try:
            clf_model = joblib.load('us_maize_yield_classifier.pkl')
            label_encoder = joblib.load('label_encoder.pkl')
            expected_features = len(preprocessor.get_feature_names_out())
            if hasattr(clf_model, 'n_features_in_') and clf_model.n_features_in_ == expected_features:
                has_classification = True
                st.success("Maize classifier loaded.")
            else:
                st.warning("Classifier feature mismatch.")
        except:
            st.warning("Maize classifier not available.")

    elif mode == "Honey":
        try:
            reg_model = joblib.load('honey_yield_regressor2.pkl')
            preprocessor = joblib.load('honey_preprocessor2.pkl')
            clf_model = joblib.load('honey_yield_classifier2.pkl')
            label_encoder = joblib.load('honey_label_encoder2.pkl')
            historical_df = pd.read_csv('merged_honey_weather.csv')
            has_classification = True
            st.success("Honey models and data loaded.")
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.title(f"{mode} Yield Prediction")
    st.write(f"Enter details to predict {mode.lower()} yield.")

    # --- INPUTS ---
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

    else:  # Honey
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

            input_preprocessed = preprocessor.transform(input_data)
            reg_prediction = reg_model.predict(input_preprocessed)[0]

            unit = 't/ha' if mode == 'Maize' else 'lbs/colony'
            st.success(f"**Predicted Yield**: {reg_prediction:.2f} {unit}")

            if mode == "Maize":
                total_yield = reg_prediction * area
                st.success(f"**Total Crop**: {total_yield:.2f} tons")
            else:
                total_yield = reg_prediction * colonies_number / 1000
                st.success(f"**Total Honey**: {total_yield:.2f} tons")

            # Classification
            if has_classification:
                clf_pred = clf_model.predict(input_preprocessed)[0]
                clf_label = label_encoder.inverse_transform([clf_pred])[0]
                color = "#ccffcc" if clf_label == 'High' else "#ffcccc" if clf_label == 'Low' else "#fff3cd"
                border = "#44ff44" if clf_label == 'High' else "#ff4444" if clf_label == 'Low' else "#ffc107"
                st.markdown(f'<div style="background-color:{color}; padding:10px; border-radius:5px; border-left:5px solid {border};">'
                            f'<strong>Yield Category: {clf_label}</strong></div>', unsafe_allow_html=True)
            else:
                st.info("Classification not available.")

            # --- CHARTS ---
            st.header("Yield Report")
            y_col = 'yield' if mode == 'Maize' else 'yield_per_colony'
            hist_avg = historical_df.groupby('year')[y_col].mean().reset_index()
            fig1 = px.line(hist_avg, x='year', y=y_col, title=f"Historical {mode} Yield Trend")
            fig1.add_scatter(x=[year], y=[reg_prediction], mode='markers', name='Predicted', marker=dict(color='red', size=10))
            st.plotly_chart(fig1, use_container_width=True)

            mean_y = historical_df[y_col].mean()
            comp_df = pd.DataFrame({'Type': ['Historical Avg', 'Predicted'], 'Yield': [mean_y, reg_prediction]})
            fig2 = px.bar(comp_df, x='Type', y='Yield', color='Type', title="Predicted vs Historical")
            st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.histogram(historical_df, x=y_col, nbins=30, title="Yield Distribution")
            fig3.add_vline(x=reg_prediction, line_dash="dash", line_color="red", annotation_text=f"Predicted: {reg_prediction:.2f}")
            st.plotly_chart(fig3, use_container_width=True)

            # Feature Importance
            cat_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(
                ['country', 'crop_type', 'season'] if mode == 'Maize' else ['state', 'season']
            ))
            num_features = ['year', 'rainfall' if mode == 'Maize' else 'total_rainfall',
                            'temp' if mode == 'Maize' else 'avg_temp'] + (
                ['tmin', 'tmax', 'rad', 'et0', 'cwb', 'area'] if mode == 'Maize' else ['colonies_number']
            )
            feature_names = num_features + cat_features
            feature_names = [f for f in feature_names if f in [col.split('__')[-1] for col in preprocessor.get_feature_names_out()]]
            imp_df = pd.DataFrame({
                'Feature': feature_names[:len(reg_model.feature_importances_)],
                'Importance': reg_model.feature_importances_
            })
            fig4 = px.bar(imp_df.sort_values('Importance'), x='Importance', y='Feature', orientation='h', title="Feature Importance")
            st.plotly_chart(fig4, use_container_width=True)

            st.write(f"Predicted yield is **{reg_prediction - mean_y:+.2f} {unit}** than historical average.")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# --- BEE DISEASE MODE ---
else:  # mode == "Bee"
    st.title("Bee Hive Disease Diagnosis")
    st.write("Upload a hive image to detect diseases and get management advice.")

    bee_model = load_bee_model()

    uploaded_file = st.file_uploader("Upload Hive Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file and st.button("Analyze Hive"):
    try:
        img = Image.open(uploaded_file).convert('L')  # Grayscale
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Resize to 28x28 and preprocess
        img_resized = img.resize((28, 28))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        
        # Option A: Flatten for (None, 784) input
        img_array = img_array.reshape(1, -1) / 255.0

        # Option B: Use (1, 28, 28, 1) if model expects 4D
        # img_array = np.expand_dims(img_array, axis=-1)
        # img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = bee_model.predict(img_array)[0]
        pred_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        diagnosis = BEE_METADATA['class_names'][pred_idx]
        ...

            # Display result
            is_healthy = diagnosis == 'healthy'
            alert_class = "alert-success" if is_healthy else "alert-danger"
            st.markdown(f'<div class="{alert_class}"><strong>Diagnosis: {diagnosis}</strong> '
                        f'({confidence*100:.1f}% confidence)</div>', unsafe_allow_html=True)

            # Explanation
            st.write("**Explanation:**", DIAGNOSIS_EXPLANATIONS.get(diagnosis, "No details available."))

            # Advice
            st.subheader("Recommended Actions")
            responses = FALLBACK_RESPONSES.get(diagnosis, {})
            for key, advice in responses.items():
                st.write(f"**{key.capitalize()}**:")
                st.write(advice.replace('\n', '<br>'), unsafe_allow_html=True)

            # Confidence bar
            conf_df = pd.DataFrame({
                'Condition': BEE_METADATA['class_names'],
                'Confidence': prediction
            })
            fig = px.bar(conf_df, x='Condition', y='Confidence', title="Diagnosis Confidence")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Analysis failed: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("About")
    st.write("**AgriBee AI** combines:")
    st.write("- Crop yield prediction (Maize/Honey)")
    st.write("- Bee hive health diagnosis via deep learning")
    st.write("**Models**: XGBoost, TensorFlow CNN")
    st.write("**Data**: USDA, Weather, Hive Images")
    st.write("Built with Streamlit")
