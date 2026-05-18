import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import base64

# Configuración de la página (DEBE SER EL PRIMER COMANDO)
st.set_page_config(
    page_title="CardioPredict - Enfermedad Cardíaca",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== ESTILOS CSS PERSONALIZADOS ==========
st.markdown("""
<style>
    /* Fondo y colores principales */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tarjetas de resultados */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        color: white;
    }
    
    .result-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .result-value {
        font-size: 48px;
        font-weight: bold;
        margin: 15px 0;
    }
    
    .probability {
        font-size: 18px;
        background: rgba(255,255,255,0.2);
        border-radius: 50px;
        padding: 8px;
        margin-top: 10px;
    }
    
    /* Encabezado principal */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 30px;
        margin-bottom: 30px;
        color: white;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 48px;
        margin-bottom: 10px;
    }
    
    .main-header p {
        font-size: 18px;
        opacity: 0.9;
    }
    
    /* Tarjeta de entrada de datos */
    .input-card {
        background: rgba(255,255,255,0.95);
        border-radius: 25px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    /* Botón de predicción */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 15px 40px;
        border-radius: 50px;
        border: none;
        width: 100%;
        transition: transform 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Sliders personalizados */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Información del estudiante */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        background: rgba(0,0,0,0.2);
        border-radius: 20px;
        color: white;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER PRINCIPAL ==========
st.markdown("""
<div class="main-header">
    <h1>❤️ CardioPredict</h1>
    <p>Predicción inteligente de enfermedad cardíaca basada en Machine Learning</p>
    <p style="font-size:14px; opacity:0.8;">Random Forest & SVM - Modelos entrenados con datos clínicos reales</p>
</div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## 📊 Información del Paciente")
    st.markdown("---")
    
    # Datos personales
    st.markdown("### 👤 Datos Personales")
    age = st.slider("📅 Edad", 20, 100, 50, help="Edad del paciente en años")
    
    sex = st.radio("⚥ Sexo", ["Femenino", "Masculino"], horizontal=True)
    sex = 1 if sex == "Masculino" else 0
    
    st.markdown("---")
    
    # Datos clínicos principales
    st.markdown("### ❤️ Síntomas y Signos")
    
    cp = st.selectbox("💢 Tipo de dolor de pecho", [
        "Asintomático", 
        "Angina atípica", 
        "Angina no anginal", 
        "Angina típica"
    ])
    cp_map = {"Asintomático": 0, "Angina atípica": 1, "Angina no anginal": 2, "Angina típica": 3}
    cp = cp_map[cp]
    
    trestbps = st.slider("📏 Presión arterial en reposo (mm Hg)", 80, 200, 120, help="Valor normal: 120")
    
    chol = st.slider("🩸 Colesterol sérico (mg/dl)", 100, 400, 200, help="Valor normal: <200")
    
    st.markdown("---")
    
    # Medidas avanzadas
    st.markdown("### 📈 Medidas Avanzadas")
    
    fbs = st.radio("🍬 Azúcar en sangre en ayunas > 120 mg/dl", ["No", "Sí"], horizontal=True)
    fbs = 1 if fbs == "Sí" else 0
    
    restecg = st.selectbox("📊 Electrocardiograma en reposo", [
        "Normal", 
        "Anomalía ST-T", 
        "Hipertrofia ventricular"
    ])
    restecg_map = {"Normal": 0, "Anomalía ST-T": 1, "Hipertrofia ventricular": 2}
    restecg = restecg_map[restecg]
    
    thalach = st.slider("🏃 Frecuencia cardíaca máxima", 60, 220, 150, help="Frecuencia máxima alcanzada durante ejercicio")
    
    exang = st.radio("🏋️ Angina inducida por ejercicio", ["No", "Sí"], horizontal=True)
    exang = 1 if exang == "Sí" else 0

# ========== COLUMNAS PARA MÁS CAMPOS ==========
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Parámetros Específicos")
    
    oldpeak = st.slider("📉 Depresión ST inducida por ejercicio", 0.0, 6.0, 1.0, step=0.1, 
                        help="Indica depresión del segmento ST")
    
    slope = st.selectbox("📐 Pendiente del segmento ST", ["Ascendente", "Plana", "Descendente"])
    slope_map = {"Ascendente": 0, "Plana": 1, "Descendente": 2}
    slope = slope_map[slope]
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("### 🫀 Hallazgos Clínicos")
    
    ca = st.slider("🔬 Vasos coloreados (0-3)", 0, 3, 0, 
                   help="Número de vasos principales coloreados por fluoroscopia")
    
    thal = st.selectbox("💊 Talasemia", ["Normal", "Defecto fijo", "Defecto reversible"])
    thal_map = {"Normal": 1, "Defecto fijo": 2, "Defecto reversible": 3}
    thal = thal_map[thal]
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== BOTÓN DE PREDICCIÓN ==========
st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predecir = st.button("🔍 PREDECIR RIESGO CARDÍACO", use_container_width=True)

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

# ========== REALIZAR PREDICCIÓN ==========
if predecir:
    if modelo_rf is None:
        st.error("❌ Error: No se pudieron cargar los modelos. Verifica que los archivos .pkl estén en la carpeta 'modelos'")
    else:
        # Crear array con las características en el orden correcto
        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal
        ]])
        
        # Escalar características
        features_scaled = scaler.transform(features)
        
        # Predicciones
        pred_rf = modelo_rf.predict(features_scaled)[0]
        pred_svm = modelo_svm.predict(features_scaled)[0]
        
        # Probabilidades
        proba_rf = modelo_rf.predict_proba(features_scaled)[0]
        proba_svm = modelo_svm.predict_proba(features_scaled)[0]
        
        # ========== MOSTRAR RESULTADOS ==========
        st.markdown("---")
        st.markdown("## 📋 Resultados de la Predicción")
        
        col_rf, col_svm = st.columns(2)
        
        with col_rf:
            riesgo_rf = proba_rf[1] * 100
            color_rf = "#ff4757" if pred_rf == 1 else "#2ed573"
            icono_rf = "⚠️" if pred_rf == 1 else "✅"
            texto_rf = "ALTO RIESGO" if pred_rf == 1 else "BAJO RIESGO"
            
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, {color_rf}dd, {color_rf}99);">
                <div class="result-title">🌲 Random Forest {icono_rf}</div>
                <div class="result-value">{texto_rf}</div>
                <div class="probability">Probabilidad de enfermedad: <b>{riesgo_rf:.1f}%</b></div>
                <div style="font-size:14px; margin-top:15px;">
                    {"🔴 Se recomienda consultar a un especialista" if pred_rf == 1 else "🟢 Perfil de riesgo bajo - Mantener hábitos saludables"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_svm:
            riesgo_svm = proba_svm[1] * 100
            color_svm = "#ff4757" if pred_svm == 1 else "#2ed573"
            icono_svm = "⚠️" if pred_svm == 1 else "✅"
            texto_svm = "ALTO RIESGO" if pred_svm == 1 else "BAJO RIESGO"
            
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, {color_svm}dd, {color_svm}99);">
                <div class="result-title">🤖 SVM {icono_svm}</div>
                <div class="result-value">{texto_svm}</div>
                <div class="probability">Probabilidad de enfermedad: <b>{riesgo_svm:.1f}%</b></div>
                <div style="font-size:14px; margin-top:15px;">
                    {"🔴 Se recomienda consultar a un especialista" if pred_svm == 1 else "🟢 Perfil de riesgo bajo - Mantener hábitos saludables"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Barra de probabilidad promedio
        riesgo_promedio = (riesgo_rf + riesgo_svm) / 2
        st.markdown("---")
        st.markdown("### 📊 Nivel de riesgo promedio")
        st.progress(int(riesgo_promedio))
        st.caption(f"Riesgo cardiovascular estimado: {riesgo_promedio:.1f}%")

# ========== FOOTER ==========
st.markdown("""
<div class="footer">
    <p>🔗 <a href="https://colab.research.google.com/drive/TU_ENLACE_AQUI" target="_blank" style="color:white;">Ver cuaderno en Google COLAB</a></p>
    <p><strong>Nombre:</strong> Neil Pariona | <strong>Código ISIL:</strong> TU_CODIGO</p>
    <p style="font-size:12px; opacity:0.7;">Modelos entrenados con Heart Disease UCI Dataset | Random Forest & SVM</p>
</div>
""", unsafe_allow_html=True)
