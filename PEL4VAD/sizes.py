import numpy as np
import os

import os
from collections import defaultdict

import numpy as np
import os

def compare_first_two_files(directory):
    # Diccionario para almacenar los nombres de los archivos por prefix
    file_groups = {}

    # Leer todos los archivos en el directorio especificado
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            # Obtener el prefix del archivo (sin la extensión)
            prefix = filename.split('_x264')[0]  # Usar la parte antes de cualquier '__'
            filepath = os.path.join(directory, filename)
            # Agregar el archivo al grupo correspondiente
            if prefix not in file_groups:
                file_groups[prefix] = []
            file_groups[prefix].append(filepath)

    # Comparar los tamaños de los archivos
    for prefix, files in file_groups.items():
        if len(files) >= 2:  # Asegurarse de que hay al menos dos archivos
            # Cargar el primer y segundo archivo
            size1 = np.load(files[0]).shape
            size2 = np.load(files[1]).shape
            # Imprimir los tamaños si son diferentes
            if size1 != size2:
                print(f'{os.path.basename(files[0])}')

            # Agregar condición para imprimir el primer archivo en caso de que no se compare
            if len(files) == 1:
                print(f'El archivo "{os.path.basename(files[0])}" es único en su grupo y tiene tamaño: {size1}')


# Uso
directory = 'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/train/'
compare_first_two_files(directory)


