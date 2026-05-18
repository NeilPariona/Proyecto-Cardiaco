import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Configurar página
st.set_page_config(page_title="Predictor de Enfermedad Cardíaca", layout="wide")
st.title("❤️ Predictor de Enfermedad Cardíaca")
st.markdown("Ingresa los datos clínicos del paciente para obtener una predicción")

# Cargar modelos
@st.cache_resource
def cargar_modelos():
    rf = joblib.load("modelos/modelo_rf_heart.pkl")
    svm = joblib.load("modelos/modelo_svm_heart.pkl")
    scaler = joblib.load("modelos/scaler_heart.pkl")
    return rf, svm, scaler

modelo_rf, modelo_svm, scaler = cargar_modelos()

# Crear columnas para entrada de datos
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sexo", ["Femenino", "Masculino"])
    sex = 1 if sex == "Masculino" else 0
    
    cp = st.selectbox("Tipo de dolor de pecho", 
                      ["Asintomático", "Angina atípica", "Angina no anginal", "Angina típica"])
    cp_map = {"Asintomático": 0, "Angina atípica": 1, "Angina no anginal": 2, "Angina típica": 3}
    cp = cp_map[cp]
    
    trestbps = st.number_input("Presión arterial en reposo (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Colesterol sérico (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Azúcar en sangre en ayunas > 120 mg/dl", ["No", "Sí"])
    fbs = 1 if fbs == "Sí" else 0

with col2:
    restecg = st.selectbox("Electrocardiograma en reposo", 
                           ["Normal", "Anomalía ST-T", "Hipertrofia ventricular"])
    restecg_map = {"Normal": 0, "Anomalía ST-T": 1, "Hipertrofia ventricular": 2}
    restecg = restecg_map[restecg]
    
    thalach = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Angina inducida por ejercicio", ["No", "Sí"])
    exang = 1 if exang == "Sí" else 0
    
    oldpeak = st.number_input("Depresión ST inducida por ejercicio", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox("Pendiente del segmento ST", ["Plana", "Ascendente", "Descendente"])
    slope_map = {"Ascendente": 0, "Plana": 1, "Descendente": 2}
    slope = slope_map[slope]
    
    ca = st.slider("Número de vasos coloreados (0-3)", 0, 3, 0)
    thal = st.selectbox("Talasemia", ["Normal", "Defecto fijo", "Defecto reversible"])
    thal_map = {"Normal": 1, "Defecto fijo": 2, "Defecto reversible": 3}
    thal = thal_map[thal]

# Botón para predecir
if st.button("🔍 Predecir", type="primary"):
    # Crear array con las características
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                          thalach, exang, oldpeak, slope, ca, thal]])
    
    # Escalar características
    features_scaled = scaler.transform(features)
    
    # Predicciones
    pred_rf = modelo_rf.predict(features_scaled)[0]
    pred_svm = modelo_svm.predict(features_scaled)[0]
    
    # Probabilidades (SVM con probability=True)
    proba_rf = modelo_rf.predict_proba(features_scaled)[0]
    proba_svm = modelo_svm.predict_proba(features_scaled)[0]
    
    # Mostrar resultados
    st.divider()
    st.subheader("📊 Resultados")
    
    col_rf, col_svm = st.columns(2)
    
    with col_rf:
        st.markdown("### 🌲 Random Forest")
        if pred_rf == 1:
            st.error("⚠️ **Predicción: Enfermedad cardíaca**")
        else:
            st.success("✅ **Predicción: Sin enfermedad cardíaca**")
        st.write(f"**Probabilidad de enfermedad:** {proba_rf[1]:.2%}")
    
    with col_svm:
        st.markdown("### 🤖 SVM")
        if pred_svm == 1:
            st.error("⚠️ **Predicción: Enfermedad cardíaca**")
        else:
            st.success("✅ **Predicción: Sin enfermedad cardíaca**")
        st.write(f"**Probabilidad de enfermedad:** {proba_svm[1]:.2%}")
    
    st.divider()

# Enlace a COLAB y datos del estudiante
st.markdown("---")
st.markdown("🔗 [Ver cuaderno en Google COLAB](https://colab.research.google.com/drive/TU_ENLACE_AQUI)")
st.markdown("**Nombre:** Tu Nombre Completo | **Código ISIL:** 12345678")
