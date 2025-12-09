# Despliegue del modelo de Churn con FastAPI

## 1. Objetivo

Este documento describe cómo integrar el modelo de churn entrenado (`models/churn_model.pkl`) en un servicio web real usando **FastAPI**.  

La idea es exponer un endpoint HTTP que reciba los datos de un cliente y devuelva:

- la **predicción de churn** (0/1)  
- la **probabilidad** asociada  


---

## 2. Arquitectura propuesta

1. **Repositorio MLOps (este proyecto)**  
   - Contiene el código de preprocesamiento (`src/data_prep.py`), entrenamiento (`src/train.py`) y evaluación (`src/evaluate.py`); y el modelo entrenado en `models/churn_model.pkl`.
   - Los artefactos se versionan con **DVC**.

2. **Servicio de predicción**  
   - App de **FastAPI** (`src/app.py`) que:
     - carga el modelo entrenado al iniciar,
     - expone un endpoint `POST /predict`,
     - valida la entrada con Pydantic,
     - aplica el mismo preprocesamiento y devuelve la predicción.

3. **Cliente**  
   - Puede ser un front web, un script de Python o cualquier sistema que llame al endpoint vía HTTP.

---

## 3. Requisitos

Agregar a `requirements.txt`:

- fastapi
- uvicorn
- pydantic
---

## 4. Ejemplo de request/response

### 4.1. JSON de entrada (features del cliente)

```json
{
  "age": 35,
  "gender": "Male",
  "region": "South",
  "contract_type": "Two year",
  "tenure_months": 12,
  "monthly_charges": 70.5,
  "total_charges": 840.0,
  "internet_service": "Fiber optic",
  "phone_service": "Yes",
  "multiple_lines": "No",
  "payment_method": "Electronic check"
}
````

### 4.2. JSON de salida (respuesta del modelo)

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.78
}