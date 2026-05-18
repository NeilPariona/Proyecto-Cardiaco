import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="CardioRisk | Evaluación Cardiovascular",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== ESTILOS CSS PROFESIONAL - TIMES NEW ROMAN ==========
st.markdown("""
<style>
    /* Reset y fuentes */
    @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
    
    html, body, .stApp, .stMarkdown, div, p, span, label, h1, h2, h3, h4 {
        font-family: 'Times New Roman', Georgia, serif !important;
    }
    
    /* Fondo neutro profesional */
    .stApp {
        background: #f8f9fa;
    }
    
    /* Contenedor principal con márgenes */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Tarjeta base - estilo institucional */
    .card {
        background: #ffffff;
        border-radius: 0px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        border: 1px solid #e0e0e0;
    }
    
    /* Tarjeta de resultado */
    .result-card {
        background: #ffffff;
        border-radius: 0px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: none;
        border: 1px solid #e0e0e0;
        height: 100%;
        margin: 0.5rem 0;
    }
    
    .card-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
        font-family: 'Times New Roman', Georgia, serif !important;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        margin: 0.75rem 0;
        font-family: 'Times New Roman', Georgia, serif !important;
    }
    
    .metric-sub {
        font-size: 0.75rem;
        color: #6c757d;
        margin-top: 0.5rem;
        font-family: 'Times New Roman', Georgia, serif !important;
    }
    
    /* Colores de riesgo profesionales */
    .risk-low {
        color: #2c7a4d;
    }
    .risk-moderate {
        color: #b7652e;
    }
    .risk-high {
        color: #c0392b;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0px;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    .badge-low { background: #eaf4ed; color: #2c7a4d; border-left: 3px solid #2c7a4d; }
    .badge-moderate { background: #fef1e6; color: #b7652e; border-left: 3px solid #b7652e; }
    .badge-high { background: #fdedec; color: #c0392b; border-left: 3px solid #c0392b; }
    
    /* Barra de probabilidad */
    .prob-bar-container {
        background: #e9ecef;
        border-radius: 0px;
        height: 4px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        width: 0%;
        transition: width 0.3s ease;
    }
    .prob-bar-fill.low { background: #2c7a4d; }
    .prob-bar-fill.moderate { background: #b7652e; }
    .prob-bar-fill.high { background: #c0392b; }
    
    /* Header principal */
    .main-header {
        background: #ffffff;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        border-bottom: 2px solid #c0392b;
        text-align: left;
    }
    .main-header h1 {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0;
        font-family: 'Times New Roman', Georgia, serif !important;
        letter-spacing: -0.3px;
    }
    .main-header p {
        color: #6c757d;
        margin: 0.5rem 0 0;
        font-size: 0.85rem;
        font-family: 'Times New Roman', Georgia, serif !important;
    }
    
    /* Sidebar profesional */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] .sidebar-content {
        padding: 1.5rem;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #1a1a1a;
        font-weight: 500;
        font-size: 1rem;
        margin-bottom: 0.75rem;
        font-family: 'Times New Roman', Georgia, serif !important;
    }
    
    /* Inputs profesionales */
    .stSlider > div > div > div {
        background: #c0392b;
    }
    .stSelectbox > div > div, .stNumberInput > div > div {
        border-radius: 0px;
        border: 1px solid #ced4da;
        background: white;
    }
    label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        font-family: 'Times New Roman', Georgia, serif !important;
    }
    
    /* Botón profesional */
    .stButton > button {
        background: #1a1a1a;
        color: white;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.6rem 1.2rem;
        border-radius: 0px;
        border: none;
        width: 100%;
        transition: all 0.2s;
        font-family: 'Times New Roman', Georgia, serif !important;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: #c0392b;
        transform: none;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        background: #ffffff;
        border-top: 1px solid #e0e0e0;
        color: #6c757d;
        font-size: 0.7rem;
        font-family: 'Times New Roman', Georgia, serif !important;
    }
    .footer a {
        color: #1a1a1a;
        text-decoration: none;
        border-bottom: 1px solid #c0392b;
    }
    
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Divider sutil */
    .section-divider {
        height: 1px;
        background: #e0e0e0;
        margin: 1rem 0;
    }
    
    /* Ajuste de márgenes en columnas */
    .row-widget {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
<div class="main-header">
    <h1>CardioRisk</h1>
    <p>Evaluación de Riesgo Cardiovascular · Herramienta de apoyo clínico</p>
</div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### Datos del Paciente")
    st.markdown("---")
    
    age = st.slider("Edad (años)", 20, 100, 50)
    sex = st.radio("Sexo", ["Femenino", "Masculino"], horizontal=True)
    sex = 1 if sex == "Masculino" else 0
    
    st.markdown("### Síntomas y Signos Vitales")
    
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
    
    st.markdown("### Estudios Complementarios")
    
    fbs = st.radio("Glucemia basal > 120 mg/dl", ["No", "Sí"], horizontal=True)
    fbs = 1 if fbs == "Sí" else 0
    
    restecg = st.selectbox("ECG en reposo", ["Normal", "Anomalía ST-T", "Hipertrofia ventricular"])
    restecg_map = {"Normal": 0, "Anomalía ST-T": 1, "Hipertrofia ventricular": 2}
    restecg = restecg_map[restecg]
    
    thalach = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=220, value=150)
    exang = st.radio("Angina inducida por ejercicio", ["No", "Sí"], horizontal=True)
    exang = 1 if exang == "Sí" else 0

# ========== COLUMNAS - PARÁMETROS ==========
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Parámetros ECG")
        oldpeak = st.slider("Depresión del segmento ST", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Pendiente del segmento ST", ["Ascendente", "Plana", "Descendente"])
        slope_map = {"Ascendente": 0, "Plana": 1, "Descendente": 2}
        slope = slope_map[slope]
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Hallazgos por Imagen")
        ca = st.slider("Vasos coloreados (fluoroscopia)", 0, 3, 0)
        thal = st.selectbox("Talasemia", ["Normal", "Defecto fijo", "Defecto reversible"])
        thal_map = {"Normal": 1, "Defecto fijo": 2, "Defecto reversible": 3}
        thal = thal_map[thal]
        st.markdown('</div>', unsafe_allow_html=True)

# ========== BOTÓN ==========
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predecir = st.button("Generar Evaluación de Riesgo", use_container_width=True)

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
        st.error("Error: No se pudieron cargar los modelos. Verifique la carpeta 'modelos'.")
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
        st.markdown("## Resultados de la Evaluación")
        
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
        
        st.markdown('<div class="card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown("#### Recomendación Clínica")
        if riesgo_promedio >= 70:
            st.info("""
            **Riesgo Cardiovascular Alto**  
            - Se recomienda evaluación por cardiología en los próximos 7 días.  
            - Considerar estudios complementarios: ecocardiograma, prueba de esfuerzo.  
            - Evaluar factores de riesgo modificables.
            """)
        elif riesgo_promedio >= 40:
            st.info("""
            **Riesgo Cardiovascular Moderado**  
            - Programar control en 1 a 3 meses.  
            - Promover cambios en estilo de vida: dieta, ejercicio, control de estrés.  
            - Monitorear presión arterial y perfil lipídico.
            """)
        else:
            st.info("""
            **Riesgo Cardiovascular Bajo**  
            - Mantener controles periódicos anuales.  
            - Continuar con hábitos de vida saludables.  
            - No se requiere intervención inmediata.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown(f"""
<div class="footer">
    <p>🔗 <a href="https://colab.research.google.com/drive/1uo0Sb4xdyYNEVlsn71h-7_QYlIFdNIPP?usp=sharing" target="_blank">Cuaderno de desarrollo en Google COLAB</a></p>
    <p><strong>Neil Pariona</strong> | Código ISIL: 6816</p>
    <p>Heart Disease UCI Dataset · Random Forest & Support Vector Machine</p>
    <p>© 2026 - Herramienta de apoyo diagnóstico. No reemplaza el criterio médico.</p>
</div>
""", unsafe_allow_html=True)
