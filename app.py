import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Configuración de la página (DEBE SER EL PRIMER COMANDO)
st.set_page_config(
    page_title="Solo Leveling - Cardio Predictor",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== ESTILOS CSS - TEMÁTICA SOLO LEVELING ==========
st.markdown("""
<style>
    /* Fondo principal oscuro - estilo solo leveling */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f0f1a 100%);
    }
    
    /* Fondo de los widgets */
    .stApp > header {
        background: rgba(0,0,0,0.8);
    }
    
    /* Tarjetas de resultados - estilo sombra/oscuro */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0,255,255,0.2);
        border: 1px solid rgba(0,255,255,0.3);
        color: white;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 30px rgba(0,255,255,0.4);
        border-color: #00ffff;
    }
    
    .result-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        font-family: monospace;
        text-shadow: 0 0 5px cyan;
    }
    
    .result-value {
        font-size: 42px;
        font-weight: bold;
        margin: 15px 0;
        font-family: monospace;
    }
    
    .probability {
        font-size: 18px;
        background: rgba(0,255,255,0.1);
        border-radius: 50px;
        padding: 8px;
        margin-top: 10px;
        border: 1px solid cyan;
    }
    
    /* Encabezado principal */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        color: white;
        box-shadow: 0 0 30px rgba(0,255,255,0.2);
        border: 1px solid rgba(0,255,255,0.5);
    }
    
    .main-header h1 {
        font-size: 54px;
        margin-bottom: 10px;
        font-family: monospace;
        text-shadow: 0 0 10px cyan, 0 0 20px blue;
        letter-spacing: 3px;
    }
    
    .main-header p {
        font-size: 16px;
        opacity: 0.9;
        color: #aaa;
    }
    
    /* Tarjeta de entrada de datos */
    .input-card {
        background: rgba(10,10,26,0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.5);
        border: 1px solid rgba(0,255,255,0.3);
    }
    
    /* Botón de predicción */
    .stButton > button {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        color: cyan;
        font-size: 20px;
        font-weight: bold;
        padding: 15px 40px;
        border-radius: 50px;
        border: 2px solid cyan;
        width: 100%;
        transition: all 0.3s;
        font-family: monospace;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        background: cyan;
        color: black;
        box-shadow: 0 0 30px cyan;
        border-color: black;
    }
    
    /* Sliders personalizados */
    .stSlider > div > div > div {
        background: cyan;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: #0a0a0a;
        color: cyan;
        border-color: cyan;
    }
    
    /* Radio buttons */
    .stRadio > div {
        color: white;
    }
    
    /* Información del estudiante */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        background: rgba(0,0,0,0.7);
        border-radius: 20px;
        color: #aaa;
        border: 1px solid rgba(0,255,255,0.2);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(0,0,0,0.8);
        backdrop-filter: blur(10px);
        border-right: 2px solid cyan;
    }
    
    /* Títulos dentro del sidebar */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: cyan;
        font-family: monospace;
    }
    
    /* Labels */
    label {
        color: cyan !important;
        font-family: monospace !important;
    }
    
    /* Números de los sliders */
    .stSlider label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER PRINCIPAL - ESTILO SOLO LEVELING ==========
st.markdown("""
<div class="main-header">
    <h1>⚔️ SHADOW SOLDIER ⚔️</h1>
    <h2 style="color:cyan; font-family:monospace;">CARDIO PREDICTOR</h2>
    <p>「 El sistema de evaluación del gremio - Predicción de enfermedad cardíaca 」</p>
    <p style="font-size:12px;">Modelos: Random Forest | SVM — Basado en datos clínicos reales</p>
</div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## 🧬 PERFIL DEL CAZADOR")
    st.markdown("---")
    
    st.markdown("### 👤 DATOS BÁSICOS")
    age = st.slider("📅 EDAD", 20, 100, 50, help="Rango de edad del cazador")
    
    sex = st.radio("⚥ GÉNERO", ["Femenino", "Masculino"], horizontal=True)
    sex = 1 if sex == "Masculino" else 0
    
    st.markdown("---")
    st.markdown("### ❤️ SÍNTOMAS")
    
    cp = st.selectbox("💢 DOLOR DE PECHO", [
        "Asintomático", 
        "Angina atípica", 
        "Angina no anginal", 
        "Angina típica"
    ])
    cp_map = {"Asintomático": 0, "Angina atípica": 1, "Angina no anginal": 2, "Angina típica": 3}
    cp = cp_map[cp]
    
    trestbps = st.slider("📏 PRESIÓN ARTERIAL", 80, 200, 120, help="mm Hg - Valor normal: 120")
    chol = st.slider("🩸 COLESTEROL", 100, 400, 200, help="mg/dl - Valor normal: <200")
    
    st.markdown("---")
    st.markdown("### 📊 ESTADOS AVANZADOS")
    
    fbs = st.radio("🍬 AZÚCAR EN AYUNAS >120", ["No", "Sí"], horizontal=True)
    fbs = 1 if fbs == "Sí" else 0
    
    restecg = st.selectbox("📈 ELECTROCARDIOGRAMA", [
        "Normal", 
        "Anomalía ST-T", 
        "Hipertrofia ventricular"
    ])
    restecg_map = {"Normal": 0, "Anomalía ST-T": 1, "Hipertrofia ventricular": 2}
    restecg = restecg_map[restecg]
    
    thalach = st.slider("🏃 FRECUENCIA CARDÍACA MÁX", 60, 220, 150)
    exang = st.radio("⚡ ANGINA POR EJERCICIO", ["No", "Sí"], horizontal=True)
    exang = 1 if exang == "Sí" else 0

# ========== COLUMNAS PARA MÁS CAMPOS ==========
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("### 🔬 PARÁMETROS AVANZADOS")
    
    oldpeak = st.slider("📉 DEPRESIÓN ST", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("📐 PENDIENTE ST", ["Ascendente", "Plana", "Descendente"])
    slope_map = {"Ascendente": 0, "Plana": 1, "Descendente": 2}
    slope = slope_map[slope]
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("### 🫀 HALLAZGOS")
    
    ca = st.slider("🔬 VASOS COLOREADOS", 0, 3, 0)
    thal = st.selectbox("💊 TALASEMIA", ["Normal", "Defecto fijo", "Defecto reversible"])
    thal_map = {"Normal": 1, "Defecto fijo": 2, "Defecto reversible": 3}
    thal = thal_map[thal]
    st.markdown('</div>', unsafe_allow_html=True)

# ========== BOTÓN DE PREDICCIÓN ==========
st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predecir = st.button("🔮 EJECUTAR ANÁLISIS DEL SISTEMA", use_container_width=True)

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
        st.error("❌ Error: El sistema no puede cargar los modelos. Contacta al gremio.")
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
        
        # ========== MOSTRAR RESULTADOS ==========
        st.markdown("---")
        st.markdown("## ⚡ RESULTADOS DEL SISTEMA ⚡")
        
        col_rf, col_svm = st.columns(2)
        
        with col_rf:
            riesgo_rf = proba_rf[1] * 100
            color_rf = "#ff0055" if pred_rf == 1 else "#00ffff"
            icono_rf = "☠️" if pred_rf == 1 else "✅"
            texto_rf = "NIVEL: AMENAZA" if pred_rf == 1 else "NIVEL: SEGURO"
            
            st.markdown(f"""
            <div class="result-card" style="border-color: {color_rf}; box-shadow: 0 0 20px {color_rf}80;">
                <div class="result-title">🌲 RANDOM FOREST {icono_rf}</div>
                <div class="result-value" style="color: {color_rf};">{texto_rf}</div>
                <div class="probability">Probabilidad de ataque: <b>{riesgo_rf:.1f}%</b></div>
                <div style="font-size:14px; margin-top:15px; color:{color_rf};">
                    {"⚠️ Se requiere invocación de sanador ⚠️" if pred_rf == 1 else "🛡️ El cazador está a salvo 🛡️"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_svm:
            riesgo_svm = proba_svm[1] * 100
            color_svm = "#ff0055" if pred_svm == 1 else "#00ffff"
            icono_svm = "☠️" if pred_svm == 1 else "✅"
            texto_svm = "NIVEL: AMENAZA" if pred_svm == 1 else "NIVEL: SEGURO"
            
            st.markdown(f"""
            <div class="result-card" style="border-color: {color_svm}; box-shadow: 0 0 20px {color_svm}80;">
                <div class="result-title">🤖 SUPPORT VECTOR MACHINE {icono_svm}</div>
                <div class="result-value" style="color: {color_svm};">{texto_svm}</div>
                <div class="probability">Probabilidad de ataque: <b>{riesgo_svm:.1f}%</b></div>
                <div style="font-size:14px; margin-top:15px; color:{color_svm};">
                    {"⚠️ Se requiere invocación de sanador ⚠️" if pred_svm == 1 else "🛡️ El cazador está a salvo 🛡️"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        riesgo_promedio = (riesgo_rf + riesgo_svm) / 2
        
        st.markdown("---")
        st.markdown("### 📊 RIESGO PROMEDIO ESTIMADO POR EL SISTEMA")
        st.progress(int(riesgo_promedio))
        st.caption(f"**{riesgo_promedio:.1f}%** - {'⚠️ ZONA DE PELIGRO' if riesgo_promedio > 50 else '🟢 ZONA SEGURA'}")

# ========== FOOTER ==========
st.markdown(f"""
<div class="footer">
    <p>🔗 <a href="https://colab.research.google.com/drive/1uo0Sb4xdyYNEVlsn71h-7_QYlIFdNIPP?usp=sharing" target="_blank" style="color:cyan;">📓 ACCEDER AL CUADERNO DEL GREMIO (COLAB)</a></p>
    <p><strong>🏷️ CAZADOR:</strong> Neil Pariona | <strong>🆔 CÓDIGO ISIL:</strong> 6816</p>
    <p style="font-size:12px; opacity:0.6;">⚔️ Solo Leveling Style - Basado en Heart Disease UCI Dataset | Random Forest & SVM ⚔️</p>
</div>
""", unsafe_allow_html=True)
