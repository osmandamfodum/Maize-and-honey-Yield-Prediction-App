import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import tensorflow as tf
from PIL import Image
import os
import json
# ------------------- إجبار حذف الـ cache القديم -------------------
st.cache_resource.clear() # يحذف النموذج القديم من الذاكرة
# ------------------- Suppress TF warnings -------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# ------------------- Page Config -------------------
st.set_page_config(page_title="AgriBee AI", layout="wide")
# ------------------- Custom CSS -------------------
st.markdown("""
<style>
&nbsp;&nbsp;&nbsp;&nbsp;.stApp { background-color: white; color: black; }
&nbsp;&nbsp;&nbsp;&nbsp;.image-container { background-color: white; padding: 10px; display: flex; justify-content: center; }
&nbsp;&nbsp;&nbsp;&nbsp;.stImage > img { max-width: 100%; height: auto; }
&nbsp;&nbsp;&nbsp;&nbsp;.alert-success { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb; }
&nbsp;&nbsp;&nbsp;&nbsp;.alert-danger { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb; }
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
mode = st.radio("اختر وضع التنبؤ", ["Maize", "Honey", "Bee"], index=0, horizontal=True)
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
# تحميل من JSON إذا وُجد
if os.path.exists("bee_metadata.json"):
    try:
        with open("bee_metadata.json") as f:
            loaded = json.load(f)
            BEE_METADATA.update(loaded)
    except:
        pass
# ------------------- DIAGNOSIS TEXT -------------------
DIAGNOSIS_EXPLANATIONS = {
    'Varroa, Small Hive Beetles': "تم اكتشاف بقع حمراء أو بنية صغيرة، تشير إلى إصابة بعث الـ Varroa وخنافس الخلية.",
    'ant problems': "تم اكتشاف وجود نمل، مما يشير إلى مشاكل متعلقة بالنمل في الخلية.",
    'few varroa, hive beetles': "تم اكتشاف مستويات منخفضة من عث الـ Varroa وخنافس الخلية.",
    'healthy': "لا توجد علامات مرضية مرئية؛ الخلية تبدو سليمة.",
    'hive being robbed': "تم اكتشاف علامات نهب، مثل زيادة العدوانية أو وجود نحل ميت.",
    'missing queen': "هناك دلائل على فقدان الملكة، مثل غياب البيض أو خلايا الملكة."
}
FALLBACK_RESPONSES = {
    'Varroa, Small Hive Beetles': {
        'treatment': "- علاج بحمض الأكساليك أو الأميتراز.\n- مراقبة أعداد العث والخنافس.\n- تدوير العلاجات.",
        'management': "- فحص مستويات الآفات شهريًا.\n- استخدام لوحات لزجة."
    },
    'ant problems': {
        'treatment': "- استخدام حواجز النمل أو مصائد الطعم.\n- تقليل فتحات الخلية.\n- إزالة المغريات.",
        'management': "- مراقبة نشاط النمل أسبوعيًا.\n- الحفاظ على نظافة منطقة الخلية."
    },
    'few varroa, hive beetles': {
        'treatment': "- علاج خفيف (مثل الثيمول).\n- استخدام مصائد الخنافس.\n- مراقبة المستويات.",
        'management': "- فحوصات دورية للخلية.\n- الحفاظ على نظافة الخلية."
    },
    'healthy': {
        'care': "- توفير تغذية متوازنة.\n- الحفاظ على نظافة الخلايا.\n- مراقبة صحة الملكة.",
        'next steps': "- فحوصات دورية للخلية.\n- ضمان قوة المستعمرة."
    },
    'hive being robbed': {
        'management': "- تقليل فتحات الخلية.\n- نقل الخلايا الضعيفة.\n- توفير تغذية إضافية.",
        'treatment': "- تركيب شاشات مضادة للنهب.\n- مراقبة العدوانية."
    },
    'missing queen': {
        'treatment': "- إدخال ملكة جديدة.\n- التحقق من خلايا الملكة.\n- دمج مع مستعمرة لها ملكة.",
        'management': "- مراقبة أنماط البيض.\n- ضمان استقرار المستعمرة."
    }
}
# ------------------- LOAD BEE MODEL (مع إجبار إعادة التحميل) -------------------
@st.cache_resource
def load_bee_model(_model_path="bee_224_model2.h5"):
    if not os.path.exists(_model_path):
        st.error("ملف النموذج مفقود: bee_224_model.h5")
        st.stop()
    try:
        model = tf.keras.models.load_model(_model_path, compile=False)
        # تحقق من المدخل
        if model.input_shape != (None, 224, 224, 3):
            st.error(f"خطأ: حجم المدخل {model.input_shape}، مطلوب (224,224,3)")
            st.stop()
        st.success("تم تحميل نموذج النحل بنجاح")
        return model
    except Exception as e:
        st.error(f"فشل تحميل النموذج: {e}")
        st.stop()
# ------------------- MAIZE / HONEY MODE -------------------
if mode in ["Maize", "Honey"]:
    has_classification = False
    if mode == "Maize":
        try:
            reg_model = joblib.load('us_maize_yield_regressor.pkl')
            preprocessor = joblib.load('preprocessor.pkl')
            historical_df = pd.read_csv('processed_us_maize_data.csv')
            st.success("تم تحميل نماذج الذرة")
        except Exception as e:
            st.error(f"خطأ في تحميل الذرة: {e}")
            st.stop()
        try:
            clf_model = joblib.load('us_maize_yield_classifier.pkl')
            label_encoder = joblib.load('label_encoder.pkl')
            if len(preprocessor.get_feature_names_out()) == clf_model.n_features_in_:
                has_classification = True
                st.success("مُصنّف الذرة جاهز")
        except:
            st.warning("مُصنّف الذرة غير متوفر")
    else: # Honey
        try:
            reg_model = joblib.load('honey_yield_regressor2.pkl')
            preprocessor = joblib.load('honey_preprocessor2.pkl')
            clf_model = joblib.load('honey_yield_classifier2.pkl')
            label_encoder = joblib.load('honey_label_encoder2.pkl')
            historical_df = pd.read_csv('merged_honey_weather.csv')
            has_classification = True
            st.success("تم تحميل نماذج العسل")
        except Exception as e:
            st.error(f"خطأ في تحميل العسل: {e}")
            st.stop()
    st.title(f"تنبؤ إنتاجية {mode}")
    if mode == "Maize":
        country = st.selectbox("البلد", ["US"])
        crop_type = st.selectbox("نوع المحصول", ["Maize"])
        season = st.selectbox("الموسم", ["Spring", "Summer", "Fall", "Winter"])
        year = st.number_input("السنة", 2000, 2050, 2025)
        area = st.number_input("المساحة (هكتار)", 0.0, value=100.0)
        rainfall = st.number_input("الأمطار (مم)", 0.0, value=500.0)
        temp = st.number_input("درجة الحرارة (°C)", 0.0, value=20.0)
        tmin = st.number_input("أدنى درجة (°C)", 0.0, value=15.0)
        tmax = st.number_input("أعلى درجة (°C)", 0.0, value=25.0)
        rad = st.number_input("الإشعاع الشمسي (MJ/m²)", 0.0, value=15.0)
        et0 = st.number_input("التبخر (مم)", 0.0, value=5.0)
        cwb = st.number_input("توازن الماء (مم)", -1000.0, value=0.0)
    else:
        state = st.selectbox("الولاية", historical_df['state'].unique())
        season = st.selectbox("الموسم", ["Spring", "Summer", "Fall", "Winter"])
        year = st.number_input("السنة", 1995, 2021, 2020)
        colonies_number = st.number_input("عدد المستعمرات", 0, value=50000)
        avg_temp = st.number_input("متوسط درجة الحرارة (°C)", 0.0, value=15.0)
        total_rainfall = st.number_input("إجمالي الأمطار (مم)", 0.0, value=500.0)
    if st.button("احسب الإنتاجية"):
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
            unit = 'طن/هكتار' if mode == 'Maize' else 'رطل/مستعمرة'
            st.success(f"**الإنتاجية المتوقعة**: {pred:.2f} {unit}")
            total = pred * area if mode == "Maize" else pred * colonies_number / 1000
            st.success(f"**الإجمالي**: {total:.2f} طن")
            if has_classification:
                cat = label_encoder.inverse_transform(clf_model.predict(X))[0]
                color = "#ccffcc" if cat == 'High' else "#ffcccc" if cat == 'Low' else "#fff3cd"
                border = "#44ff44" if cat == 'High' else "#ff4444" if cat == 'Low' else "#ffc107"
                st.markdown(f'<div style="background-color:{color}; padding:10px; border-radius:5px; border-left:5px solid {border};">'
                            f'<strong>فئة الإنتاجية: {cat}</strong></div>', unsafe_allow_html=True)
            y_col = 'yield' if mode == 'Maize' else 'yield_per_colony'
            hist = historical_df.groupby('year')[y_col].mean().reset_index()
            fig1 = px.line(hist, x='year', y=y_col, title=f"الاتجاه التاريخي لإنتاجية {mode}")
            fig1.add_scatter(x=[year], y=[pred], mode='markers', name='المتوقع', marker=dict(color='red', size=12))
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"فشل الحساب: {e}")
# ------------------- BEE DISEASE MODE -------------------
else:
    st.title("تشخيص أمراض خلية النحل")
    st.write("ارفع صورة الخلية للكشف عن الأمراض وتلقي نصائح العلاج.")
    bee_model = load_bee_model()
    uploaded_file = st.file_uploader("ارفع صورة الخلية", type=['png', 'jpg', 'jpeg'])
    if uploaded_file and st.button("تحليل الخلية"):
        try:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="الصورة المرفوعة", use_column_width=True)
            # معالجة الصورة
            img_resized = img.resize(BEE_METADATA['image_size'])
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            # التنبؤ
            preds = bee_model.predict(img_array, verbose=0)[0]
            idx = np.argmax(preds)
            confidence = float(preds[idx])
            diagnosis = BEE_METADATA['class_names'][idx]
            # تنبيهات الثقة
            if confidence >= 0.85:
                st.success(f"موثوق: {diagnosis} ({confidence*100:.1f}%)")
            elif confidence >= 0.70:
                st.warning(f"ثقة متوسطة: {diagnosis} ({confidence*100:.1f}%). تحقق من جودة الصورة.")
            else:
                st.error(f"ثقة منخفضة: {diagnosis} ({confidence*100:.1f}%). جرب صورة أوضح.")
            # الشرح والنصائح
            st.write("**الشرح:**", DIAGNOSIS_EXPLANATIONS.get(diagnosis, "غير معروف"))
            st.subheader("الإجراءات الموصى بها")
            for key, text in FALLBACK_RESPONSES.get(diagnosis, {}).items():
                st.write(f"**{key.capitalize()}**:")
                st.write(text.replace('\n', '
'), unsafe_allow_html=True)
            # رسم بياني
            conf_df = pd.DataFrame({'الحالة': BEE_METADATA['class_names'], 'الثقة': preds})
            fig = px.bar(conf_df, x='الحالة', y='الثقة', range_y=[0, 1], title="مستوى الثقة في التشخيص")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"فشل التحليل: {e}")
# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.header("AgriBee AI")
    st.write("**الميزات:**")
    st.write("- تنبؤ إنتاجية الذرة والعسل")
    st.write("- تشخيص أمراض خلية النحل")
    st.write("**النماذج:** XGBoost + MobileNetV2")
    st.write("**البيانات:** USDA، الطقس، 5000+ صورة خلية")
    st.write("مبرمج بـ **Streamlit**")
