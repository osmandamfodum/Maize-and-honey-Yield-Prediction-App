import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import tensorflow as tf
from PIL import Image
import os
import requests
import logging
import json

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .stApp { background-color: white; color: black; }
    .image-container { background-color: white; padding: 10px; display: flex; justify-content: center; }
    .stImage > img { max-width: 100%; height: auto; }
    .chat-container { max-width: 700px; margin: 20px auto; padding: 15px; border: 1px solid #ddd; border-radius: 10px; }
    .chat-bubble { padding: 10px 15px; margin: 8px 0; border-radius: 15px; max-width: 80%; }
    .user-bubble { background-color: #007bff; color: white; align-self: flex-end; }
    .bot-bubble { background-color: #e9ecef; color: black; align-self: flex-start; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header Image ---
try:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image("neu.jpg", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
except Exception as e:
    st.warning("Header image not found. Continuing without it.")
    logger.error(f"Header image error: {e}")

# --- Mode Selection ---
mode = st.radio("Select App Mode", ["Maize", "Honey", "Bee"], index=0, horizontal=True)

# --- BEE MODEL SETUP ---
@st.cache_resource
def load_bee_model():
    try:
        class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(**kwargs)
        tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D
        
        model_path = 'bee.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load bee model: {e}")
        logger.error(f"Bee model load error: {e}")
        st.stop()

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

# --- DeepInfra + Fallback API Handling ---
DI_API_TOKEN = st.secrets.get("DI_API_TOKEN", os.getenv("DI_API_TOKEN", ""))
HF_API_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))

DI_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
HF_API_URL = "https://api-inference.huggingface.co/models"
DI_MODELS = {
    'DeepSeek-V3': 'deepseek/DeepSeek-V3',
    'Qwen3': 'qwen/Qwen-32B',
    'Mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1'
}

def query_di_api(prompt, model='DeepSeek-V3'):
    """Query DeepInfra first, fallback to Hugging Face if it fails."""
    token = DI_API_TOKEN
    if not token:
        st.warning("⚠️ DeepInfra API token not found. Using public fallback.")
        return query_hf_fallback(prompt, model), "HuggingFace"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        payload = {
            "model": DI_MODELS[model],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.7
        }
        response = requests.post(DI_API_URL, headers=headers, json=payload, timeout=20)
        if response.status_code == 403:
            logger.warning("403 Forbidden from DeepInfra — switching to Hugging Face fallback.")
            return query_hf_fallback(prompt, model), "HuggingFace"
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'], "DeepInfra"
    except requests.exceptions.RequestException as e:
        logger.error(f"DeepInfra request failed: {e}")
        return query_hf_fallback(prompt, model), "HuggingFace"
    except Exception as e:
        logger.error(f"Unexpected DeepInfra error: {e}")
        return None, "None"

def query_hf_fallback(prompt, model='Mixtral'):
    """Use Hugging Face Inference API as fallback if DeepInfra fails."""
    try:
        model_name = DI_MODELS.get(model, 'mistralai/Mixtral-8x7B-Instruct-v0.1')
        url = f"{HF_API_URL}/{model_name}"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}

        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 250, "temperature": 0.7}}
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code == 403:
            logger.warning("403 Forbidden from Hugging Face — using rule-based fallback.")
            return None
        response.raise_for_status()

        result = response.json()
        if isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text']
        else:
            return json.dumps(result)[:1000]
    except Exception as e:
        logger.error(f"Hugging Face fallback failed: {e}")
        return None

# --- Prediction function ---
def predict_bee_image(img, model):
    try:
        img = img.resize(BEE_METADATA['image_size'])
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        pred = model.predict(img_array)[0]
        idx = np.argmax(pred)
        diagnosis = BEE_METADATA['class_names'][idx]
        confidence = float(pred[idx])
        return diagnosis, confidence
    except Exception as e:
        st.error(f"Image prediction failed: {e}")
        logger.error(f"Prediction error: {e}")
        return None, None

# --- MAIN APP LOGIC ---
if mode == "Bee":
    st.title("🐝 Bee Hive Health Detector")
    st.write("Upload a hive image to detect diseases and ask follow-up questions.")

    bee_model = load_bee_model()
    uploaded_file = st.file_uploader("Upload Hive Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Analyzing hive health..."):
                diagnosis, confidence = predict_bee_image(img, bee_model)
                if diagnosis is None:
                    raise ValueError("Prediction returned None")
                conf_pct = f"{confidence * 100:.1f}%"

                if diagnosis == 'healthy':
                    st.success(f"✅ Healthy Hive ({conf_pct} confidence)")
                else:
                    st.error(f"⚠️ Issue Detected: {diagnosis} ({conf_pct})")

                st.write(f"**Explanation:** {DIAGNOSIS_EXPLANATIONS.get(diagnosis, 'N/A')}")
        except Exception as e:
            st.error(f"Image upload/processing failed: {e}. Try a different image.")
            logger.error(f"Upload error: {e}")

        # Chatbot
        if diagnosis:
            st.markdown("---")
            st.subheader("Ask a Bee Health Expert")

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Ask about treatment, prevention, or next steps..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            ai_prompt = f"""
                            You are a bee health expert. Hive diagnosis: '{diagnosis}'.
                            User asks: '{prompt}'
                            Respond in bullet points with practical advice on treatment, management, and prevention.
                            """

                            response, source = query_di_api(ai_prompt)
                            if not response:
                                # Rule-based fallback
                                response = None
                                for key in FALLBACK_RESPONSES.get(diagnosis, {}):
                                    if key in prompt.lower():
                                        response = FALLBACK_RESPONSES[diagnosis][key]
                                        break
                                if not response:
                                    response = "Consult a local beekeeping expert for hands-on advice."
                                source = "Rule-based"

                            st.markdown(response)
                            st.caption(f"Response Source: {source}")
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Chatbot response error: {e}")
                            logger.error(f"Chatbot error: {e}")

else:
    st.info("Switch to **Bee** mode to test the updated DeepInfra + Hugging Face fallback logic.")

# --- Sidebar Info ---
with st.sidebar:
    st.header("About")
    if mode == "Bee":
        st.write("**Model**: CNN trained on 6 hive conditions")
        st.write("**Classes**: Varroa, ants, robbing, queenless, etc.")
        st.write("**Chatbot**: DeepInfra → HuggingFace → Rule-based fallback")
    else:
        st.write("**Yield Prediction**: XGBoost models")
        st.write("**Data**: Historical weather + yield records")
