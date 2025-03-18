import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Función para graficar los archivos .npy en un grid 2x2
def plot_npy_files(folder1, folder2, folder3, filename, label1, label2, label3, ax):
    # Cargar los datos
    data1 = np.load(folder1)
    # data1 = np.repeat(data1, 16)
    data2 = np.load(folder2)
    data3 = np.load(folder3)

    # Plot en el subgráfico actual (ax)
    ax.plot(data1, label=label1)
    ax.plot(data2, label=label2)
    ax.plot(data3, label=label3)

    ax.set_xlabel("Frames")
    ax.set_ylabel("Value")
    ax.set_title(f"Comparison for {filename}")
    
    # Ajustar los límites del gráfico
    max_ylim = max(max(data1), max(data2), max(data3))
    ax.set_ylim(0, 1.01)
    max_len = max(len(data1), len(data2), len(data3))
    ax.set_xlim(0, max_len)

    ax.legend()




def plot_npy_files_gt(folder1, folder2, folder3, folder4, filename, label1, label2, label3, label4, ax):
    # Cargar los datos
    data1 = np.load(folder1)
    data2 = np.load(folder2)
    data3 = np.load(folder3)
    gt = np.load(folder4)

    # Plot en el subgráfico actual (ax)
    ax.plot(data1, label=label1)
    ax.plot(data2, label=label2)
    ax.plot(data3, label=label3)
    ax.plot(gt, label=label4)

    ax.set_xlabel("Frames")
    ax.set_ylabel("Value")
    ax.set_title(f"Comparison for {filename}")
    
    # Ajustar los límites del gráfico
    max_ylim = max(max(data1), max(data2), max(data3), max(gt))
    ax.set_ylim(0, 1.01)
    max_len = len(gt)
    ax.set_xlim(0, max_len)

    ax.legend()


def plot_npy_files_gt2(folder2, folder3, folder4, filename, label2, label3, label4, ax):
    # Cargar los datos

    data2 = np.load(folder2)
    data3 = np.load(folder3)
    
    gt = np.load(folder4)

    # Plot en el subgráfico actual (ax)

    ax.plot(data2, label=label2)
    ax.plot(data3, label=label3)
    ax.plot(gt, label=label4)

    ax.set_xlabel("Frames")
    ax.set_ylabel("Value")
    ax.set_title(f"Comparison for {filename}")
    
    # Ajustar los límites del gráfico
    max_ylim = max(max(data2), max(data3), max(gt))
    ax.set_ylim(0, 1.01)
    max_len = len(gt)
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
mode_BN = "test_normalized"
compare = True
compare_train = False

if mode == "test":
   path = 'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/list/ucf/test.list'
else: 
   path = 'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/list/ucf/test_train.list'

archivos_seleccionados = seleccionar_aleatorios(path)

# Graficar cada uno de los 4 archivos seleccionados
for i, archivo in enumerate(archivos_seleccionados):
    name = archivo.split('/')[-1].split('_x264')[0]
    name = "Shooting047"
    filename = f"{name}_pred.npy"

    gt = f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/gt/{name}_x264_pred.npy'

    
    
    

    if compare == True:
        PEL4VAd = f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}/{filename}'
        URDMU = f'C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/frame_label/predictions/{mode}/30fps/{filename}'
        BNWVAD = f'frame_label/predictions/{mode_BN}/30fps/{filename}'

        label1 = "PEL4VAD"
        label2 = "UR-DMU"
        label3 = "BN-WVAD normalized"


        name = filename.split('_pred.npy')[0]
        
        if mode == "test":

            plot_npy_files_gt(PEL4VAd, URDMU, BNWVAD, gt, filename, label1, label2, label3, label4="GT", ax=ax[i])
        else:
            plot_npy_files(PEL4VAd, URDMU, BNWVAD, filename, label1, label2, label3, ax=ax[i])
    
    if compare == False and mode == "test":
        BNWVAD = f'frame_label/predictions/test_normalized/{filename}'
        BNWVAD_quince = f'frame_label/predictions/test_normalscores/{filename}'
        label2 = "BN-WVAD normalized"
        label3 = "BN-WVAD normalscores"
        plot_npy_files_gt2(BNWVAD, BNWVAD_quince, gt, filename, label2, label3, label4="GT", ax=ax[i])

    if compare_train == True:
        PEL4VAd = f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/train/{filename}'
        URDMU = f'C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/frame_label/predictions/train/{filename}'
        BNWVAD = f'frame_label/predictions/train_normalized/{filename}'

        gt = f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/gt/train/{name}_x264_pred.npy'

        label1 = "PEL4VAD"
        label2 = "UR-DMU"
        label3 = "BN-WVAD normalized"

        plot_npy_files_gt(PEL4VAd, URDMU, BNWVAD, gt, filename, label1, label2, label3, label4="GT", ax=ax[i])


        

    

# Ajustar la distribución de los subgráficos
plt.tight_layout()
plt.show()
