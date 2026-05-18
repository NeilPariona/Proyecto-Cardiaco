import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="CardioRisk | Evaluación Cardiovascular",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== ESTILOS CSS PROFESIONALES ==========
st.markdown("""
<style>
    /* Fondo institucional */
    .stApp {
        background: #f4f6f9;
    }
    
    /* Tarjeta base */
    .card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Tarjeta de resultado */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    
    .card-title {
        font-size: 14px;
        font-weight: 600;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #1a202c;
    }
    
    .metric-sub {
        font-size: 13px;
        color: #718096;
        margin-top: 8px;
    }
    
    .risk-low {
        color: #2ecc71;
    }
    
    .risk-moderate {
        color: #f39c12;
    }
    
    .risk-high {
        color: #e74c3c;
    }
    
    /* Barra de probabilidad */
    .prob-bar-container {
        background: #edf2f7;
        border-radius: 8px;
        height: 6px;
        margin: 12px 0;
        overflow: hidden;
    }
    
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        width: 0%;
        transition: width 0.3s ease;
    }
    
    .prob-bar-fill.low { background: #2ecc71; }
    .prob-bar-fill.moderate { background: #f39c12; }
    .prob-bar-fill.high { background: #e74c3c; }
    
    /* Header principal */
    .main-header {
        background: white;
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        border: 1px solid #e2e8f0;
    }
    
    .main-header h1 {
        font-size: 24px;
        font-weight: 600;
        color: #1a202c;
        margin: 0;
    }
    
    .main-header p {
        color: #718096;
        margin: 4px 0 0;
        font-size: 14px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        padding: 20px;
    }
    
    /* Inputs */
    .stSlider > div > div > div {
        background: #3182ce;
    }
    
    .stSelectbox > div > div, .stNumberInput > div > div {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    label {
        color: #4a5568 !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }
    
    /* Botón principal */
    .stButton > button {
        background: #3182ce;
        color: white;
        font-size: 15px;
        font-weight: 500;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: background 0.2s;
    }
    
    .stButton > button:hover {
        background: #2c5282;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        background: white;
        border-radius: 12px;
        color: #718096;
        font-size: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .footer a {
        color: #3182ce;
        text-decoration: none;
    }
    
    /* Divider */
    hr {
        margin: 20px 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Badge de riesgo */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .badge-low { background: #e8f8f5; color: #2ecc71; }
    .badge-moderate { background: #fef5e7; color: #f39c12; }
    .badge-high { background: #fdedec; color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
<div class="main-header">
    <h1>🏥 CardioRisk | Evaluación de Riesgo Cardiovascular</h1>
    <p>Herramienta clínica de apoyo basada en Machine Learning · Heart Disease UCI Dataset</p>
</div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 📋 Datos del Paciente")
    st.markdown("---")
    
    age = st.slider("Edad (años)", 20, 100, 50)
    sex = st.radio("Sexo", ["Femenino", "Masculino"], horizontal=True)
    sex = 1 if sex == "Masculino" else 0
    
    st.markdown("### ❤️ Síntomas y Signos Vitales")
    
    cp = st.selectbox("Tipo de dolor torácico", [
        "Asintomático", "Angina atípica", "Angina no anginal", "Angina típica"
    ])
    cp_map = {"Asintomático": 0, "Angina atípica": 1, "Angina no anginal": 2, "Angina típica": 3}
    cp = cp_map[cp]
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        trestbps = st.number_input("Presión arterial (mmHg)", min_value=80, max_value=200, value=120)
    with col_sb2:
        chol = st.number_input("Colesterol (mg/dl)", min_value=100, max_value=400, value=200)
    
    st.markdown("### 📊 Estudios Complementarios")
    
    fbs = st.radio("Glucemia basal > 120 mg/dl", ["No", "Sí"], horizontal=True)
    fbs = 1 if fbs == "Sí" else 0
    
    restecg = st.selectbox("ECG en reposo", ["Normal", "Anomalía ST-T", "Hipertrofia ventricular"])
    restecg_map = {"Normal": 0, "Anomalía ST-T": 1, "Hipertrofia ventricular": 2}
    restecg = restecg_map[restecg]
    
    thalach = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=220, value=150)
    exang = st.radio("Angina inducida por ejercicio", ["No", "Sí"], horizontal=True)
    exang = 1 if exang == "Sí" else 0

# ========== COLUMNAS CAMPOS RESTANTES ==========
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔬 Parámetros ECG")
        oldpeak = st.slider("Depresión del segmento ST", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Pendiente del segmento ST", ["Ascendente", "Plana", "Descendente"])
        slope_map = {"Ascendente": 0, "Plana": 1, "Descendente": 2}
        slope = slope_map[slope]
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🫀 Hallazgos por Imagen")
        ca = st.slider("Vasos coloreados (fluoroscopia)", 0, 3, 0)
        thal = st.selectbox("Talasemia", ["Normal", "Defecto fijo", "Defecto reversible"])
        thal_map = {"Normal": 1, "Defecto fijo": 2, "Defecto reversible": 3}
        thal = thal_map[thal]
        st.markdown('</div>', unsafe_allow_html=True)

# ========== BOTÓN ==========
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predecir = st.button("📊 Generar Evaluación de Riesgo", use_container_width=True)

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
        st.error("❌ Error: No se pudieron cargar los modelos. Verifique la carpeta 'modelos'.")
    else:
        # Construir features
        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal
        ]])
        
        features_scaled = scaler.transform(features)
        
        # Predicciones
        pred_rf = modelo_rf.predict(features_scaled)[0]
        pred_svm = modelo_svm.predict(features_scaled)[0]
        
        proba_rf = modelo_rf.predict_proba(features_scaled)[0]
        proba_svm = modelo_svm.predict_proba(features_scaled)[0]
        
        riesgo_rf = proba_rf[1] * 100
        riesgo_svm = proba_svm[1] * 100
        riesgo_promedio = (riesgo_rf + riesgo_svm) / 2
        
        # Determinar nivel de riesgo
        if riesgo_promedio < 30:
            risk_level = "Bajo"
            risk_class = "low"
            risk_badge = "badge-low"
            risk_color = "risk-low"
        elif riesgo_promedio < 60:
            risk_level = "Moderado"
            risk_class = "moderate"
            risk_badge = "badge-moderate"
            risk_color = "risk-moderate"
        else:
            risk_level = "Alto"
            risk_class = "high"
            risk_badge = "badge-high"
            risk_color = "risk-high"
        
        st.markdown("---")
        st.markdown("## 📋 Resultados de la Evaluación")
        
        # Fila de resultados
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.markdown(f"""
            <div class="result-card">
                <div class="card-title">RIESGO CARDIOVASCULAR</div>
                <div class="metric-value {risk_color}">{riesgo_promedio:.1f}%</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill {risk_class}" style="width: {riesgo_promedio}%;"></div>
                </div>
                <div class="metric-sub">
                    <span class="badge {risk_badge}">{risk_level}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r2:
            st.markdown(f"""
            <div class="result-card">
                <div class="card-title">RANDOM FOREST</div>
                <div class="metric-value">{riesgo_rf:.1f}%</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill {risk_class}" style="width: {riesgo_rf}%;"></div>
                </div>
                <div class="metric-sub">Predicción: <strong>{'Positivo' if pred_rf == 1 else 'Negativo'}</strong></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r3:
            st.markdown(f"""
            <div class="result-card">
                <div class="card-title">SVM</div>
                <div class="metric-value">{riesgo_svm:.1f}%</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill {risk_class}" style="width: {riesgo_svm}%;"></div>
                </div>
                <div class="metric-sub">Predicción: <strong>{'Positivo' if pred_svm == 1 else 'Negativo'}</strong></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Recomendación clínica
        st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown("#### 📋 Recomendación Clínica")
        
        if riesgo_promedio >= 70:
            st.warning("""
            **⚠️ Riesgo Cardiovascular Alto**  
            - Se recomienda evaluación por cardiología en los próximos 7 días.  
            - Considerar estudios complementarios: ecocardiograma, prueba de esfuerzo.  
            - Evaluar factores de riesgo modificables.
            """)
        elif riesgo_promedio >= 40:
            st.info("""
            **🟡 Riesgo Cardiovascular Moderado**  
            - Programar control en 1-3 meses.  
            - Promover cambios en estilo de vida: dieta, ejercicio, control de estrés.  
            - Monitorear presión arterial y perfil lipídico.
            """)
        else:
            st.success("""
            **🟢 Riesgo Cardiovascular Bajo**  
            - Mantener controles periódicos anuales.  
            - Continuar con hábitos de vida saludables.  
            - No se requiere intervención inmediata.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown(f"""
<div class="footer">
    <p>🔗 <a href="https://colab.research.google.com/drive/1uo0Sb4xdyYNEVlsn71h-7_QYlIFdNIPP?usp=sharing" target="_blank">📓 Cuaderno de desarrollo en Google COLAB</a></p>
    <p><strong>Neil Pariona</strong> | Código ISIL: 6816</p>
    <p>Modelos entrenados con Heart Disease UCI Dataset · Random Forest & Support Vector Machine</p>
    <p>© 2026 - Herramienta de apoyo diagnóstico. No reemplaza criterio médico.</p>
</div>
""", unsafe_allow_html=True)
