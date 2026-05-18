import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="CardioGuard - Health Monitor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== ESTILOS CSS - MODERNO / SALUD ==========
st.markdown("""
<style>
    /* Fondo limpio */
    .stApp {
        background: #f5f7fb;
    }
    
    /* Tarjetas principales */
    .health-card {
        background: white;
        border-radius: 24px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #eef2f6;
    }
    
    /* Tarjeta de resultado */
    .result-card {
        background: white;
        border-radius: 28px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        border: 1px solid #f0f0f0;
    }
    
    .result-card:hover {
        transform: translateY(-3px);
    }
    
    /* Títulos de tarjetas */
    .card-title {
        font-size: 18px;
        font-weight: 600;
        color: #6c5ce7;
        margin-bottom: 15px;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 42px;
        font-weight: 700;
        margin: 15px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        font-weight: 500;
    }
    
    .probability-bar {
        background: #e9ecef;
        border-radius: 12px;
        height: 8px;
        margin: 15px 0;
        overflow: hidden;
    }
    
    .probability-fill {
        background: #6c5ce7;
        height: 100%;
        border-radius: 12px;
        width: 0%;
    }
    
    /* Header principal */
    .main-header {
        background: white;
        padding: 20px 30px;
        border-radius: 28px;
        margin-bottom: 30px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        border: 1px solid #f0f0f0;
    }
    
    .main-header h1 {
        font-size: 28px;
        font-weight: 700;
        color: #2d3436;
        margin: 0;
    }
    
    .main-header p {
        color: #6c757d;
        margin: 5px 0 0;
        font-size: 14px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #f0f0f0;
    }
    
    /* Inputs */
    .stSlider > div > div > div {
        background: #6c5ce7;
    }
    
    .stSelectbox > div > div {
        background: white;
        border-radius: 12px;
    }
    
    /* Botón */
    .stButton > button {
        background: #6c5ce7;
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 40px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: #5b4bc4;
        transform: scale(0.98);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        background: white;
        border-radius: 20px;
        color: #6c757d;
        font-size: 13px;
        border: 1px solid #f0f0f0;
    }
    
    /* Radio buttons */
    .stRadio > div {
        display: flex;
        gap: 20px;
    }
    
    /* Títulos en sidebar */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #2d3436;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
<div class="main-header">
    <h1>❤️ CardioGuard</h1>
    <p>Monitoreo inteligente de salud cardiovascular · Basado en Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## 📋 Datos del paciente")
    st.markdown("---")
    
    age = st.slider("Edad", 20, 100, 50)
    sex = st.radio("Sexo", ["Femenino", "Masculino"], horizontal=True)
    sex = 1 if sex == "Masculino" else 0
    
    st.markdown("### ❤️ Síntomas principales")
    cp = st.selectbox("Tipo de dolor de pecho", [
        "Asintomático", "Angina atípica", "Angina no anginal", "Angina típica"
    ])
    cp_map = {"Asintomático": 0, "Angina atípica": 1, "Angina no anginal": 2, "Angina típica": 3}
    cp = cp_map[cp]
    
    trestbps = st.slider("Presión arterial (mm Hg)", 80, 200, 120)
    chol = st.slider("Colesterol (mg/dl)", 100, 400, 200)
    
    st.markdown("### 📊 Estudios")
    fbs = st.radio("Azúcar en ayunas > 120", ["No", "Sí"], horizontal=True)
    fbs = 1 if fbs == "Sí" else 0
    
    restecg = st.selectbox("ECG en reposo", ["Normal", "Anomalía ST-T", "Hipertrofia ventricular"])
    restecg_map = {"Normal": 0, "Anomalía ST-T": 1, "Hipertrofia ventricular": 2}
    restecg = restecg_map[restecg]
    
    thalach = st.slider("Frec. cardíaca máxima", 60, 220, 150)
    exang = st.radio("Angina por ejercicio", ["No", "Sí"], horizontal=True)
    exang = 1 if exang == "Sí" else 0

# ========== COLUMNAS CAMPOS RESTANTES ==========
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown('<div class="health-card">', unsafe_allow_html=True)
        st.markdown("#### 🔬 Parámetros específicos")
        oldpeak = st.slider("Depresión ST", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Pendiente ST", ["Ascendente", "Plana", "Descendente"])
        slope_map = {"Ascendente": 0, "Plana": 1, "Descendente": 2}
        slope = slope_map[slope]
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="health-card">', unsafe_allow_html=True)
        st.markdown("#### 🫀 Hallazgos clínicos")
        ca = st.slider("Vasos coloreados", 0, 3, 0)
        thal = st.selectbox("Talasemia", ["Normal", "Defecto fijo", "Defecto reversible"])
        thal_map = {"Normal": 1, "Defecto fijo": 2, "Defecto reversible": 3}
        thal = thal_map[thal]
        st.markdown('</div>', unsafe_allow_html=True)

# ========== BOTÓN ==========
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predecir = st.button("📊 Evaluar riesgo cardiovascular", use_container_width=True)

# ========== CARGAR MODELOS ==========
@st.cache_resource
def cargar_modelos():
    try:
        rf = joblib.load("modelos/modelo_rf_heart.pkl")
        svm = joblib.load("modelos/modelo_svm_heart.pkl")
        scaler = joblib.load("modelos/scaler_heart.pkl")
        return rf, svm, scaler
    except:
        return None, None, None

modelo_rf, modelo_svm, scaler = cargar_modelos()

# ========== PREDICCIÓN ==========
if predecir:
    if modelo_rf is None:
        st.error("❌ Error al cargar los modelos. Verifica la carpeta 'modelos'.")
    else:
        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal
        ]])
        
        features_scaled = scaler.transform(features)
        
        pred_rf = modelo_rf.predict(features_scaled)[0]
        pred_svm = modelo_svm.predict(features_scaled)[0]
        
        proba_rf = modelo_rf.predict_proba(features_scaled)[0]
        proba_svm = modelo_svm.predict_proba(features_scaled)[0]
        
        riesgo_rf = proba_rf[1] * 100
        riesgo_svm = proba_svm[1] * 100
        riesgo_promedio = (riesgo_rf + riesgo_svm) / 2
        
        # ========== RESULTADOS ==========
        st.markdown("---")
        st.markdown("## 📈 Resultados del análisis")
        
        # Métricas principales
        col_met1, col_met2, col_met3 = st.columns(3)
        
        with col_met1:
            st.markdown(f"""
            <div class="result-card">
                <div class="card-title">🎯 RIESGO PROMEDIO</div>
                <div class="metric-value" style="color: {'#e74c3c' if riesgo_promedio > 50 else '#2ecc71'};">{riesgo_promedio:.1f}%</div>
                <div class="probability-bar"><div class="probability-fill" style="width: {riesgo_promedio}%; background: {'#e74c3c' if riesgo_promedio > 50 else '#2ecc71'};"></div></div>
                <div class="metric-label">{'⚠️ Atención requerida' if riesgo_promedio > 50 else '✅ Dentro de lo esperado'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_met2:
            st.markdown(f"""
            <div class="result-card">
                <div class="card-title">🌲 Random Forest</div>
                <div class="metric-value" style="color: {'#e74c3c' if pred_rf == 1 else '#2ecc71'};">{'En riesgo' if pred_rf == 1 else 'Sin riesgo'}</div>
                <div class="metric-label">Probabilidad: {riesgo_rf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_met3:
            st.markdown(f"""
            <div class="result-card">
                <div class="card-title">🤖 SVM</div>
                <div class="metric-value" style="color: {'#e74c3c' if pred_svm == 1 else '#2ecc71'};">{'En riesgo' if pred_svm == 1 else 'Sin riesgo'}</div>
                <div class="metric-label">Probabilidad: {riesgo_svm:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Recomendación
        if riesgo_promedio > 70:
            st.warning("🟠 **Recomendación:** Acude a un especialista cardiovascular lo antes posible.")
        elif riesgo_promedio > 40:
            st.info("🟡 **Recomendación:** Programa una consulta de control y mejora hábitos de vida.")
        else:
            st.success("🟢 **Recomendación:** Mantén tus hábitos saludables. Sigue con controles periódicos.")

# ========== FOOTER ==========
st.markdown(f"""
<div class="footer">
    <p>🔗 <a href="https://colab.research.google.com/drive/1uo0Sb4xdyYNEVlsn71h-7_QYlIFdNIPP?usp=sharing" target="_blank">📓 Ver cuaderno en Google COLAB</a></p>
    <p><strong>Neil Pariona</strong> | Código ISIL: 6816</p>
    <p>Heart Disease UCI Dataset · Random Forest & SVM</p>
</div>
""", unsafe_allow_html=True)
