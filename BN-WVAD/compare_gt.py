import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Función para graficar los archivos .npy para PEL4VAD y GT
def plot_pel4vad_and_gt(folder_pel4vad, folder_gt, filename, label_pel4vad, label_gt, ax):
    # Cargar los datos
    data_pel4vad = np.load(folder_pel4vad)
    data_gt = np.load(folder_gt)

    # Plot en el subgráfico actual (ax)
    ax.plot(data_pel4vad, label=label_pel4vad)
    ax.plot(data_gt, label=label_gt)

    ax.set_xlabel("Frames")
    ax.set_ylabel("Value")
    ax.set_title(f"Comparison for {filename}")

    # Ajustar los límites del gráfico
    max_ylim = max(max(data_pel4vad), max(data_gt))
    ax.set_ylim(0, 1.01)
    max_len = len(data_gt)
    ax.set_xlim(0, max_len)

    ax.legend()

# Función para seleccionar archivos aleatorios de un archivo .list, asegurando que no tengan el mismo prefijo
def seleccionar_aleatorios(archivo_list, num_selecciones=4):
    # Leer el archivo .list y obtener todos los archivos como una lista
    with open(archivo_list, 'r') as f:
        archivos = f.readlines()

    # Limpiar los nombres de archivo (quitar saltos de línea)
    archivos = [archivo.strip() for archivo in archivos]

    seleccionados = []
    prefijos = set()  # Usar un conjunto para rastrear prefijos únicos

    while len(seleccionados) < num_selecciones and len(archivos) > 0:
        archivo = random.choice(archivos)  # Seleccionar aleatoriamente un archivo
        prefijo = archivo.split('/')[-1].split('_')[0]  # Extraer el prefijo

        if prefijo not in prefijos:  # Verificar si el prefijo ya ha sido seleccionado
            seleccionados.append(archivo)  # Agregar a la lista de seleccionados
            prefijos.add(prefijo)  # Agregar el prefijo al conjunto

    return seleccionados

# Crear un grid 2x2
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# Aplanar el arreglo de subgráficos (ax) para iterar más fácilmente
ax = ax.ravel()

mode = "test"

if mode == "test":
    path = 'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/list/ucf/test.list'
else:
    path = 'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/list/ucf/test_train.list'

archivos_seleccionados = seleccionar_aleatorios(path)

# Graficar cada uno de los 4 archivos seleccionados
for i, archivo in enumerate(archivos_seleccionados):
    name = archivo.split('/')[-1].split('_x264')[0]
    name = "Abuse028"
    filename = f"{name}_pred.npy"

    folder_pel4vad = f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}/windows/{filename}'
    folder_gt = f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/gt/windows/{filename}.npy'

    label_pel4vad = "PEL4VAD"
    label_gt = "GT"

    plot_pel4vad_and_gt(folder_pel4vad, folder_gt, filename, label_pel4vad, label_gt, ax=ax[i])

# Ajustar la distribución de los subgráficos
plt.tight_layout()
plt.show()
