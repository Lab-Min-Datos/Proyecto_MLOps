# 📦 Proyecto MLOps — Predicción de Churn

Este repositorio implementa un pipeline **end-to-end de Machine Learning** siguiendo buenas prácticas de **MLOps**, integrando:

- **DVC** para versionado de datos y artefactos  
- **MLflow** (mediante DagsHub) para tracking de experimentos  
- **GitHub Actions** como CI/CD  
- **Evaluación, experimentación y selección de modelos**  
- **Explicación de despliegue mediante FastAPI**  

El objetivo es **predecir churn** (abandono de clientes) utilizando un modelo de clasificación, dentro de un workflow reproducible, automatizado y colaborativo.

---

## 🚀 Estructura del Proyecto
```
Proyecto_MLOps/
│
├── data/
│   ├── raw/                  # Dataset original
│   └── processed/            # Dataset preprocesado (DVC)
│
├── models/                   # Modelos versionados con DVC
│
├── reports/                  # Métricas finales y curva ROC
│
├── src/
│   ├── data_prep.py          # Preprocesamiento
│   ├── train.py              # Entrenamiento
│   └── evaluate.py           # Evaluación
│
├── dvc.yaml                  # Pipeline DVC
├── dvc.lock                  # Estado bloqueado del pipeline
├── params.yaml               # Hiperparámetros y rutas
├── iteracion.md              # Historial de experimentos y ramas
├── deploy.md                 # Guía de despliegue con FastAPI
└── README.md
```

🔧 Tecnologías Utilizadas

- Python 3.10+
- DVC
- MLflow + DagsHub
- GitHub Actions
- pandas
- json
- joblib
- yaml
- scikit-learn
- FastAPI (ver *deploy.md*)
- Matplotlib

## 📑 Flujo General del Pipeline

### 1️⃣ Etapa 1 — Setup inicial
- Creación del repositorio
- Integración con DagsHub
- Configuración del entorno
- Creación de estructura base
- Versionado del dataset inicial con DVC


### 2️⃣ Etapa 2 — Preprocesamiento
- **Script:** src/data_prep.py
- Eliminación de columnas irrelevantes
- Encoding de variables categóricas
- Escalado de variables numéricas
- Guardado del dataset limpio en data/processed/


### 3️⃣ Etapa 3 — Entrenamiento
- **Script:** src/train.py
- Carga del dataset procesado
- Lectura de hiperparámetros desde params.yaml
- Train/test split
- Entrenamiento de modelo
- Cálculo de métricas (accuracy, precision, recall, f1)
- Guardado de modelo + métricas con DVC
- Registro automático en MLflow (si está habilitado)


### 4️⃣ Etapa 4 — Pipeline con DVC
Pipeline definido en dvc.yaml:

- data_prep
- train
- evaluate


### 5️⃣ Etapa 5 — CI/CD con GitHub Actions
El workflow `.github/workflows/ci.yaml`:

- Instala dependencias
- Configura autenticación con DagsHub
- Ejecuta dvc pull
- Ejecuta dvc repro
- Muestra métricas del experimento


### 6️⃣ Etapa 6 — Iteración colaborativa
Incluye:
- Ramas feat-* para experimentación
- Pruebas con nuevos hiperparámetros
- Apertura de Pull Requests
- Validación vía CI
- Selección del mejor experimento

**Más detalles:** *iteracion.md*

### 7️⃣ Etapa 7 — Evaluación avanzada
- **Script:** src/evaluate.py

Genera:
- reports/metrics_final.json
- reports/roc_curve.png

Ambos artefactos están versionados con DVC.

### 8️⃣ Etapa 8 — Despliegue
Documentado en *deploy.md* — Despliegue con FastAPI

Incluye:
- Carga del modelo entrenado
- Implementación del endpoint /predict
- Validación con Pydantic
- Replicación del preprocesamiento
- Ejemplo JSON de request/response
- Sugerencias para uso productivo

## 📊 Resultados Principales
Mejor experimento seleccionado:

- **Modelo:** BernoulliNB
- **alpha:** 1
- **fit_prior:** False
- **train/test split:** 0.85 / 0.15

**Métricas finales:**

- **accuracy:** 0.6447
- **precision:** 0.5084
- **recall:** 0.6642
- **f1:** 0.5759

Más información disponible en **iteracion.md**.

## 🎥 Demo / Video explicativo
[Ir al video (requiere cuenta de ISTEA)](https://drive.google.com/file/d/1etxC7MfoVi-1yArLoftQga2544T18sLn/view?usp=sharing)