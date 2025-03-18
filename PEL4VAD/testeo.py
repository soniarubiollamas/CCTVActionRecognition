# import pandas as pd

# # Ajustar opciones de visualización en pandas
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)


# # Cargar el archivo Excel
# file_path = 'predictions/resultados_clasificacion_test.xlsx'  # Cambia esta ruta a la ubicación de tu archivo
# df = pd.read_excel(file_path)

# # Extraer el tipo de anomalía de cada video
# df['Anomaly_Type'] = df['Video'].str.extract(r'([A-Za-z_]+)')

# # Calcular los porcentajes de aciertos para cada tipo de anomalía
# accuracy_summary = df.groupby('Anomaly_Type').apply(lambda x: pd.Series({
#     'Percent_3_Agreement': (x['3_algorithm'].sum() / x['Total_frames'].sum()) * 100,
#     'Percent_2_Agreement': (x['2_algorithm'].sum() / x['Total_frames'].sum()) * 100,
#     'Percent_1_Agreement': (x['1_algorithm'].sum() / x['Total_frames'].sum()) * 100,
#     'Percent_None_Agreement': (x['none'].sum() / x['Total_frames'].sum()) * 100
# }))

# # Mostrar el resumen de aciertos por tipo de anomalía
# print(accuracy_summary.reset_index())


import numpy as np
import os

# Ruta donde están almacenados los archivos de predicciones de cada algoritmo
# Cambia estas rutas según la ubicación de tus archivos
algorithms_paths = [
    "C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/test", 
    "C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/frame_label/predictions/test", 
    "C:/Users/xoni/Documents/1_uni/1_insterships/BN-WVAD-main/frame_label/predictions/test_normalized"
]

# Ruta donde se guardarán las predicciones combinadas
output_path = "predictions/test/combined_3_algorithms"

# Asegúrate de que el directorio de salida existe
os.makedirs(output_path, exist_ok=True)


video_files = sorted([f for f in os.listdir(algorithms_paths[0]) if f.endswith('.npy')])


# Procesar cada video
for video_file in video_files:
    # Cargar las predicciones de cada algoritmo
    preds = []
    for path in algorithms_paths:
        pred = np.load(os.path.join(path, video_file))  # Cargar archivo de cada algoritmo
        preds.append(pred)
    
    # Asegurarse de que todas las predicciones tienen la misma dimensión X
    X = preds[0].shape[0]
    # check if all preds have the same shape
    if all(pred.shape[0] != X for pred in preds):
        raise ValueError(f"Las dimensiones no coinciden para el archivo {video_file}")
    
    # Crear el array de salida con formato [3, X, 1024]
    combined_pred = np.stack(preds, axis=0)

    # Guardar el archivo combinado
    np.save(os.path.join(output_path, video_file), combined_pred)

print("Predicciones combinadas guardadas en:", output_path)