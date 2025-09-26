import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Custom CSS for modern UI
st.markdown(
    """
    <style>
    .stApp { background-color: white; color: black; }
    .image-container { background-color: white; padding: 10px; display: flex; justify-content: center; }
    .stImage > img { max-width: 100%; height: auto; background-color: transparent; }
    </style>
    """,
    unsafe_allow_html=True
)

# Display image at the top
st.markdown('<div class="image-container">', unsafe_allow_html=True)
st.image("neu.jpg", width='stretch')
st.markdown('</div>', unsafe_allow_html=True)

# Switch button using radio for mode selection
mode = st.radio("Switch Prediction Mode", ["Maize", "Honey"], index=0, horizontal=True)

# Load models based on mode
has_classification = False
if mode == "Maize":
    try:
        reg_model = joblib.load('us_maize_yield_regressor.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        historical_df = pd.read_csv('processed_us_maize_data.csv')
        st.success("Maize model and historical data loaded successfully.")
    except Exception as e:
        st.error(f"Error loading maize model or data: {e}")
        st.stop()
    try:
        clf_model = joblib.load('us_maize_yield_classifier.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        expected_features = len(preprocessor.get_feature_names_out())
        if hasattr(clf_model, 'n_features_in_') and clf_model.n_features_in_ == expected_features:
            has_classification = True
            st.success("Maize classification model loaded successfully.")
        else:
            st.warning("Maize classification model feature count mismatch. Skipping classification predictions.")
    except Exception:
        st.warning("Maize classification model not found. Skipping classification predictions.")
elif mode == "Honey":
    try:
        reg_model = joblib.load('honey_yield_regressor2.pkl')
        preprocessor = joblib.load('honey_preprocessor2.pkl')
        clf_model = joblib.load('honey_yield_classifier2.pkl')
        label_encoder = joblib.load('honey_label_encoder2.pkl')
        historical_df = pd.read_csv('merged_honey_weather.csv')
        has_classification = True
        st.success("Honey model and historical data loaded successfully.")
    except Exception as e:
        st.error(f"Error loading honey model or data: {e}")
        st.stop()

st.title(f"{mode} Yield Prediction App")
st.write(f"Enter the details below to predict {mode.lower()} yield and get a full report with charts.")

# User inputs
if mode == "Maize":
    country = st.selectbox("Country", ["US"])
    crop_type = st.selectbox("Crop Type", ["Maize"])
    season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
    year = st.number_input("Year", min_value=2000, max_value=2050, value=2025)
    area = st.number_input("Area (ha)", min_value=0.0, value=100.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=500.0)
    temp = st.number_input("Temperature (°C)", min_value=0.0, value=20.0)
    tmin = st.number_input("Min Temperature (°C)", min_value=0.0, value=15.0)
    tmax = st.number_input("Max Temperature (°C)", min_value=0.0, value=25.0)
    rad = st.number_input("Solar Radiation (MJ/m²)", min_value=0.0, value=15.0)
    et0 = st.number_input("Evapotranspiration (mm)", min_value=0.0, value=5.0)
    cwb = st.number_input("Climatic Water Balance (mm)", min_value=-1000.0, value=0.0)
elif mode == "Honey":
    state = st.selectbox("State", historical_df['state'].unique())
    season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
    year = st.number_input("Year", min_value=1995, max_value=2021, value=2020)
    colonies_number = st.number_input("Number of Colonies", min_value=0, value=50000)
    avg_temp = st.number_input("Average Temperature (°C)", min_value=0.0, value=15.0)
    total_rainfall = st.number_input("Total Rainfall (mm)", min_value=0.0, value=500.0)

if st.button("Predict Yield"):
    # Create input DataFrame
    if mode == "Maize":
        input_data = pd.DataFrame({
            'country': [country], 'crop_type': [crop_type], 'season': [season],
            'year': [year], 'area': [area], 'rainfall': [rainfall], 'temp': [temp],
            'tmin': [tmin], 'tmax': [tmax], 'rad': [rad], 'et0': [et0], 'cwb': [cwb]
        })
    elif mode == "Honey":
        input_data = pd.DataFrame({
            'state': [state], 'season': [season], 'year': [year],
            'colonies_number': [colonies_number], 'avg_temp': [avg_temp],
            'total_rainfall': [total_rainfall]
        })
    
    try:
        # Preprocess input
        input_preprocessed = preprocessor.transform(input_data)
        
        # Regression prediction
        reg_prediction = reg_model.predict(input_preprocessed)[0]
        st.success(f"**Predicted {mode} Yield**: {reg_prediction:.2f} {'t/ha' if mode == 'Maize' else 'lbs/colony'}")
        if mode == "Maize":
            total_yield = reg_prediction * area
            st.success(f"**Total Crop Quantity**: {total_yield:.2f} tons")
        elif mode == "Honey":
            total_yield = reg_prediction * colonies_number / 1000  # Convert to tons
            st.success(f"**Total Honey Production**: {total_yield:.2f} tons")
        
        # Classification prediction
        if has_classification:
            clf_prediction = clf_model.predict(input_preprocessed)[0]
            clf_label = label_encoder.inverse_transform([clf_prediction])[0]
            if clf_label == 'Low':
                st.markdown(f'<div style="background-color: #ffcccc; padding: 10px; border-radius: 5px; border-left: 5px solid #ff4444; color: #cc0000;"><strong>Predicted Yield Category: **{clf_label}**</strong> (Unfavorable conditions)</div>', unsafe_allow_html=True)
            elif clf_label == 'High':
                st.markdown(f'<div style="background-color: #ccffcc; padding: 10px; border-radius: 5px; border-left: 5px solid #44ff44; color: #006600;"><strong>Predicted Yield Category: **{clf_label}**</strong> (Favorable conditions)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107; color: #856404;"><strong>Predicted Yield Category: **{clf_label}**</strong> (Moderate conditions)</div>', unsafe_allow_html=True)
        else:
            st.info("Classification predictions unavailable (model not loaded).")
        
        # Full Report with Charts
        st.header(f"Full {mode} Prediction Report")
        
        # Chart 1: Historical Yield Trend
        historical_avg = historical_df.groupby('year')[f'yield' if mode == 'Maize' else 'yield_per_colony'].mean().reset_index()
        fig1 = px.line(historical_avg, x='year', y=f'yield' if mode == 'Maize' else 'yield_per_colony', title=f"Historical {mode} Yield Trend (Average per Year)")
        fig1.add_scatter(x=[year], y=[reg_prediction], mode='markers', name='Predicted Yield', marker=dict(color='red', size=10))
        fig1.update_layout(showlegend=True)
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Predicted Yield vs. Historical Average
        historical_mean = historical_df[f'yield' if mode == 'Maize' else 'yield_per_colony'].mean()
        comparison_df = pd.DataFrame({
            'Type': ['Historical Average', 'Predicted'],
            'Yield': [historical_mean, reg_prediction]
        })
        fig2 = px.bar(comparison_df, x='Type', y='Yield', title=f"Predicted {mode} Yield vs. Historical Average", color='Type')
        st.plotly_chart(fig2, use_container_width=True)

        # Chart 3: Yield Distribution (Historical vs. Predicted)
        fig3 = px.histogram(historical_df, x=f'yield' if mode == 'Maize' else 'yield_per_colony', nbins=30, title=f"Historical {mode} Yield Distribution")
        fig3.add_vline(x=reg_prediction, line_dash="dash", line_color="red", annotation_text=f"Predicted: {reg_prediction:.2f} {'t/ha' if mode == 'Maize' else 'lbs/colony'}")
        st.plotly_chart(fig3, use_container_width=True)

        # Chart 4: Feature Importance
        feature_names = ['year', 'rainfall' if mode == 'Maize' else 'total_rainfall', 'temp' if mode == 'Maize' else 'avg_temp', 
                         'tmin' if mode == 'Maize' else None, 'tmax' if mode == 'Maize' else None, 
                         'rad' if mode == 'Maize' else None, 'et0' if mode == 'Maize' else None, 
                         'cwb' if mode == 'Maize' else None, 'area' if mode == 'Maize' else 'colonies_number'] + \
                        list(preprocessor.named_transformers_['cat'].get_feature_names_out(
                            ['country', 'crop_type', 'season'] if mode == 'Maize' else ['state', 'season']))
        feature_names = [x for x in feature_names if x is not None]  # Remove None values
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(reg_model.feature_importances_)],
            'Importance': reg_model.feature_importances_
        })
        fig4 = px.bar(importance_df.sort_values(by='Importance', ascending=True), 
                      x='Importance', y='Feature', title=f"Feature Importance ({mode} Regression Model)", orientation='h')
        st.plotly_chart(fig4, use_container_width=True)

        # Additional Info
        change = reg_prediction - historical_mean
        st.write(f"The predicted yield for {year} is {reg_prediction:.2f} {'t/ha' if mode == 'Maize' else 'lbs/colony'}, which is {change:.2f} {'t/ha' if mode == 'Maize' else 'lbs/colony'} {'higher' if change > 0 else 'lower'} than the historical average of {historical_mean:.2f} {'t/ha' if mode == 'Maize' else 'lbs/colony'}.")
        if mode == "Maize":
            st.write(f"**Total Crop Quantity**: {total_yield:.2f} tons (based on {area} ha area).")
        elif mode == "Honey":
            st.write(f"**Total Honey Production**: {total_yield:.2f} tons (based on {colonies_number} colonies).")
        
        if has_classification:
            st.write(f"**Yield Category**: {clf_label} (indicating {'favorable' if clf_label == 'High' else 'moderate' if clf_label == 'Medium' else 'unfavorable'} conditions).")
        
        # Historical Comparison Table
        recent_years = historical_df[historical_df['year'] >= year - 5].groupby('year')[f'yield' if mode == 'Maize' else 'yield_per_colony'].mean().reset_index()
        recent_comparison = pd.DataFrame({
            'Year': recent_years['year'].tolist() + [year],
            'Average Yield': recent_years[f'yield' if mode == 'Maize' else 'yield_per_colony'].tolist() + [reg_prediction]
        })
        st.subheader(f"Recent Years {mode} Yield Comparison")
        st.dataframe(recent_comparison, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Sidebar for Model Info
with st.sidebar:
    st.header("Model Information")
    st.write(f"**{mode} Model**: XGBoost Regressor and Classifier (if available) trained on {historical_df.shape[0]} samples.")
    st.write("**Features**: Year, rainfall/temp, area/colonies, and categorical vars (state/season for Honey, country/crop/season for Maize).")
    st.write(f"**Data Range**: 2000–2023 (Maize) or 1995–2021 (Honey).")

    st.write("**Charts**: Historical trends, comparisons, distribution, and feature importance.")

