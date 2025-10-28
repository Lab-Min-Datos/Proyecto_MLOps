# Lectura de datos y preparación para análisis
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

with open("params.yaml") as f:
    params = yaml.safe_load(f)

def load_data(file_path):
    """Carga datos desde un archivo CSV y devuelve un DataFrame de pandas."""
    try:
        data = pd.read_csv(file_path)
        print(f"Datos cargados exitosamente desde {file_path}")
        return data
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def preprocess_data(data):
    """Realiza la limpieza y preparación de los datos para el análisis."""
    if data is not None:
        # Eliminar ids y aplicar get dummies para variables categóricas
        data.drop(columns=['customer_id'], inplace=True, errors='ignore')
        data = pd.get_dummies(data, drop_first=True)
        
        # Escalar características numéricas
        scaler = MinMaxScaler()
        data[data.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(
            data.select_dtypes(include=['float64', 'int64']))
        
        print("Datos preprocesados exitosamente.")
    return data

data = load_data(params['paths']['raw_data'])
data = preprocess_data(data)
output_file = params['paths']['processed_data']

data.to_csv(output_file, index=False)