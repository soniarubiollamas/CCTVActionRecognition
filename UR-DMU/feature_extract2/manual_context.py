import os
import numpy as np
import pandas as pd

def add_manual_context(features, start_new, end_new, fps):
    """
    Genera un nuevo conjunto de frames que agrega contexto al video original
    basado en los rangos seleccionados manualmente (start_new, end_new).

    Args:
        features (np.array): Características del video.
        start_new (float): Segundo de inicio del segmento de contexto.
        end_new (float): Segundo final del segmento de contexto.
        fps (int): Frames por segundo del video.

    Returns:
        np.array: Frames con el contexto añadido al inicio.
    """
    start_frame = int(start_new * fps/16)
    end_frame = int(end_new * fps/16)

    extended_frames = []

    for i in range(features.shape[0]):
        context_segment = features[i][start_frame:end_frame]
        repeated_context = np.tile(context_segment, (5, 1))
        
        extended_sequence = np.concatenate((repeated_context, features[i]), axis=0)
        extended_frames.append(extended_sequence)

   
    extended_frames = np.array(extended_frames)
    return extended_frames

def process_with_manual_context(directory, excel_path, output_path, fps=30):
    """
    Procesa todos los videos en un directorio y agrega contexto basado
    en rangos seleccionados manualmente desde un archivo Excel.

    Args:
        directory (str): Ruta del directorio con los archivos .npy.
        excel_path (str): Ruta al archivo Excel con los rangos manuales.
        output_path (str): Directorio donde se guardarán los archivos procesados.
        fps (int): Frames por segundo de los videos.

    Returns:
        None
    """
    # Cargar los datos desde el archivo Excel
    manual_selection = pd.read_excel(excel_path)

    # Iterar sobre cada archivo en el directorio
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            video_name = filename.split('_i3d')[0]  # Nombre base del video
            
            # Filtrar el Excel para obtener la fila correspondiente
            row = manual_selection[manual_selection['Video'] == video_name.split('_x264')[0]]
            if not row.empty:
                # Extraer los valores de inicio y fin en segundos
                start_new = row.iloc[0]['start_new']
                end_new = row.iloc[0]['end_new']
                
                print(f"Processing {video_name}: Adding context from {start_new}s to {end_new}s.")

                # Cargar las características del video
                features = np.load(os.path.join(directory, filename))

                # Generar el nuevo video con el contexto añadido
                extended_features = add_manual_context(features, start_new, end_new, fps)

                # Guardar las características procesadas
                output_filename = f"{video_name}_extended.npy"
                np.save(os.path.join(output_path, output_filename), extended_features)

    print("Processing completed.")

# Rutas y configuración
directory = "UCF_ten/minitest_2_abnormal"  # Directorio con los .npy
excel_path = "C:/Users/xoni/Documents/1_uni/TFM/documents/tiempos_contexto_minitest.xlsx"  # Ruta al archivo Excel con rangos manuales
output_path = "UCF_ten/30fps/test/context_minitest_5"  # Directorio de salida
fps = 30  # Frames por segundo

process_with_manual_context(directory, excel_path, output_path, fps)
