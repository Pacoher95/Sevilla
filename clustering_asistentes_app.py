#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 11:45:33 2025

@author: pacohermosilla
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

st.title("🏋️ Clustering de Clientes según su Perfil en el Gimnasio")
st.write("Responde a unas preguntas rápidas y descubre a qué grupo de cliente perteneces: ¿rendimiento, salud o estética? El modelo lo deduce a partir de tus hábitos, ¡sin preguntarlo directamente!")

# Inicializar almacenamiento persistente para respuestas
if 'registro' not in st.session_state:
    st.session_state.registro = pd.DataFrame()

# ------------------------
# 1. Formulario de entrada
# ------------------------
st.header("📋 Cuestionario anónimo")
st.write("*Tus respuestas se usarán para un análisis colectivo en tiempo real.*")

edad = st.slider("Edad", 16, 65, 30)
sexo = st.selectbox("Sexo", ["Mujer", "Hombre"])
sexo_val = 0 if sexo == "Mujer" else 1

horas_gym = st.slider("¿Cuántas horas entrenas en el gimnasio por semana?", 0, 20, 4)
usa_pesas = st.selectbox("¿Entrenas con pesas o máquinas de fuerza?", ["Nunca", "Ocasionalmente", "Frecuentemente"])
pesas_val = {"Nunca": 0, "Ocasionalmente": 1, "Frecuentemente": 2}[usa_pesas]
cardio_frecuencia = st.slider("¿Cuántos días haces cardio a la semana?", 0, 7, 2)

# Nueva pregunta indirecta
equilibrio = st.slider("¿Cuánto tiempo puedes mantener el equilibrio sobre una pierna? (segundos)", 0, 60, 15)

selfie_gym = st.selectbox("¿Te haces fotos o selfies en el gimnasio?", ["Nunca", "A veces", "A menudo"])
selfie_val = {"Nunca": 0, "A veces": 1, "A menudo": 2}[selfie_gym]

usuario = pd.DataFrame({
    'Edad': [edad],
    'Sexo': [sexo_val],
    'Horas_gym': [horas_gym],
    'Entrena_pesas': [pesas_val],
    'Cardio_dias': [cardio_frecuencia],
    'Equilibrio': [equilibrio],
    'Selfie_freq': [selfie_val]
})

# Guardar registro en sesión
if st.button("Enviar y ver resultados"):
    st.session_state.registro = pd.concat([st.session_state.registro, usuario], ignore_index=True)

# ------------------------
# 2. Clustering y análisis
# ------------------------
if len(st.session_state.registro) > 5:
    st.subheader("🔍 Análisis de los participantes")
    data = st.session_state.registro.copy()

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    data['Cluster'] = kmeans.labels_
    usuario_cluster = int(data.iloc[-1]['Cluster'])

    labels_map = {
        0: "💪 Rendimiento",
        1: "🧘 Salud",
        2: "🪞 Estética"
    }
    descripcion = labels_map.get(usuario_cluster, "Grupo sin etiqueta")
    st.success(f"Has sido clasificado como: {descripcion}")

    # Tabla comparativa de perfiles
    st.markdown("### 📊 Perfil promedio por grupo")
    resumen = data.groupby('Cluster').mean(numeric_only=True).round(1)
    resumen.index = [f"Grupo {i+1}" for i in resumen.index]
    st.dataframe(resumen)

    # PCA para visualización
    pca = PCA(n_components=2)
    pca_resultado = pca.fit_transform(data_scaled)
    data['PCA1'] = pca_resultado[:, 0]
    data['PCA2'] = pca_resultado[:, 1]

    fig, ax = plt.subplots()
    for i in range(3):
        grupo = data[data['Cluster'] == i]
        ax.scatter(grupo['PCA1'], grupo['PCA2'], label=f'Grupo {i+1}', alpha=0.6)
    ax.scatter(data.iloc[-1]['PCA1'], data.iloc[-1]['PCA2'], c='black', s=100, label='Tú', edgecolors='white')
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.set_title("Clustering usando todos los datos")
    ax.legend()
    st.pyplot(fig)

    st.caption("Clustering realizado con K-Means + PCA. Puedes ver tu posición respecto al resto.")
else:
    st.warning("Se necesitan al menos 6 respuestas para activar el análisis grupal. Comparte esta encuesta con más asistentes.")


