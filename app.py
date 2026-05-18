import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="CardioRisk - Evaluación Cardiovascular",
    page_icon="❤️",
    layout="wide"
)

# ========== ESTILOS MÍNIMOS Y LIMPIOS ==========
st.markdown("""
<style>
    /* Fuente limpia y profesional */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, .stApp, div, p, span, label, h1, h2, h3 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Márgenes adecuados */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Títulos */
    h1 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #1a1a2e !important;
        margin-bottom: 0.25rem !important;
    }
    
    h2 {
        font-size: 1.25rem !important;
        font-weight: 500 !important;
        color: #2d3436 !important;
        margin: 1rem 0 0.5rem 0 !important;
    }
    
    /* Separador sutil */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Tarjetas de resultado más limpias */
    .result-box {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    .result-risk {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .risk-high { color: #e74c3c; }
    .risk-moderate { color: #f39c12; }
    .risk-low { color: #27ae60; }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.title("CardioRisk")
st.caption("Evaluación de riesgo cardiovascular · Herramienta de apoyo clínico")

st.markdown("---")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("📋 Datos del Paciente")
    
    age = st.slider("Edad (años)", 20, 100, 50)
    sex = st.radio("Sexo", ["Femenino", "Masculino"])
    sex = 1 if sex == "Masculino" else 0
    
    st.subheader("❤️ Signos Vitales")
    
    cp = st.selectbox("Tipo de dolor torácico", [
        "Asintomático", "Angina atípica", "Angina no anginal", "Angina típica"
    ])
    cp_map = {"Asintomático": 0, "Angina atípica": 1, "Angina no anginal": 2, "Angina típica": 3}
    cp = cp_map[cp]
    
    col1, col2 = st.columns(2)
    with col1:
        trestbps = st.number_input("Presión arterial (mmHg)", 80, 200, 120)
    with col2:
        chol = st.number_input("Colesterol (mg/dl)", 100, 400, 200)
    
    st.subheader("📊 Estudios")
    
    fbs = st.radio("Glucemia basal > 120 mg/dl", ["No", "Sí"])
    fbs = 1 if fbs == "Sí" else 0
    
    restecg = st.selectbox("ECG en reposo", ["Normal", "Anomalía ST-T", "Hipertrofia ventricular"])
    restecg_map = {"Normal": 0, "Anomalía ST-T": 1, "Hipertrofia ventricular": 2}
    restecg = restecg_map[restecg]
    
    thalach = st.number_input("Frecuencia cardíaca máxima", 60, 220, 150)
    exang = st.radio("Angina por ejercicio", ["No", "Sí"])
    exang = 1 if exang == "Sí" else 0

# ========== COLUMNAS PRINCIPALES ==========
col1, col2 = st.columns(2)

with col1:
    oldpeak = st.slider("Depresión ST inducida por ejercicio", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Pendiente del segmento ST", ["Ascendente", "Plana", "Descendente"])
    slope_map = {"Ascendente": 0, "Plana": 1, "Descendente": 2}
    slope = slope_map[slope]

with col2:
    ca = st.slider("Vasos coloreados (fluoroscopia)", 0, 3, 0)
    thal = st.selectbox("Talasemia", ["Normal", "Defecto fijo", "Defecto reversible"])
    thal_map = {"Normal": 1, "Defecto fijo": 2, "Defecto reversible": 3}
    thal = thal_map[thal]

st.markdown("---")

# ========== BOTÓN ==========
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predecir = st.button("🔍 Evaluar Riesgo Cardiovascular", type="primary", use_container_width=True)

# ========== CARGAR MODELOS ==========
@st.cache_resource
def cargar_modelos():
    try:
        rf = joblib.load("modelos/modelo_rf_heart.pkl")
        svm = joblib.load("modelos/modelo_svm_heart.pkl")
        scaler = joblib.load("modelos/scaler_heart.pkl")
        return rf, svm, scaler
    except Exception as e:
        return None, None, None

modelo_rf, modelo_svm, scaler = cargar_modelos()

# ========== PREDICCIÓN ==========
if predecir:
    if modelo_rf is None:
        st.error("❌ Error al cargar los modelos. Verifica que los archivos .pkl estén en la carpeta 'modelos'.")
    else:
        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal
        ]])
        
        features_scaled = scaler.transform(features)
        
        proba_rf = modelo_rf.predict_proba(features_scaled)[0]
        proba_svm = modelo_svm.predict_proba(features_scaled)[0]
        
        riesgo_rf = proba_rf[1] * 100
        riesgo_svm = proba_svm[1] * 100
        riesgo_promedio = (riesgo_rf + riesgo_svm) / 2
        
        # Determinar nivel
        if riesgo_promedio >= 60:
            nivel = "Alto"
            color = "risk-high"
            emoji = "🔴"
        elif riesgo_promedio >= 30:
            nivel = "Moderado"
            color = "risk-moderate"
            emoji = "🟡"
        else:
            nivel = "Bajo"
            color = "risk-low"
            emoji = "🟢"
        
        # Mostrar resultados
        st.markdown("## 📊 Resultados")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.markdown(f"""
            <div class="result-box">
                <p style="font-size:0.85rem; color:#6c757d;">RIESGO PROMEDIO</p>
                <div class="result-risk {color}">{riesgo_promedio:.1f}%</div>
                <p style="font-size:0.9rem;">{emoji} Nivel {nivel}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r2:
            st.markdown(f"""
            <div class="result-box">
                <p style="font-size:0.85rem; color:#6c757d;">RANDOM FOREST</p>
                <div class="result-risk">{riesgo_rf:.1f}%</div>
                <p style="font-size:0.9rem;">Predicción: {'⚠️ Positivo' if riesgo_rf >= 50 else '✅ Negativo'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r3:
            st.markdown(f"""
            <div class="result-box">
                <p style="font-size:0.85rem; color:#6c757d;">SVM</p>
                <div class="result-risk">{riesgo_svm:.1f}%</div>
                <p style="font-size:0.9rem;">Predicción: {'⚠️ Positivo' if riesgo_svm >= 50 else '✅ Negativo'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Barra de progreso
        st.progress(int(riesgo_promedio))
        
        # Recomendación
        if riesgo_promedio >= 60:
            st.warning("🔴 **Recomendación:** Se sugiere evaluación por cardiología en los próximos días.")
        elif riesgo_promedio >= 30:
            st.info("🟡 **Recomendación:** Programar control médico y adoptar hábitos saludables.")
        else:
            st.success("🟢 **Recomendación:** Mantener hábitos saludables y controles anuales.")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.75rem; padding: 1rem;">
    <p>🔗 <a href="https://colab.research.google.com/drive/1uo0Sb4xdyYNEVlsn71h-7_QYlIFdNIPP?usp=sharing" target="_blank">📓 Cuaderno en Google COLAB</a></p>
    <p><strong>Neil Pariona</strong> | Código ISIL: 6816</p>
    <p>Heart Disease UCI Dataset · Random Forest & SVM</p>
    <p>© 2026 - Herramienta de apoyo diagnóstico. No reemplaza criterio médico.</p>
</div>
""", unsafe_allow_html=True)
