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
    ax.set_ylim(0, max_ylim)
    max_len = max(len(data1), len(data2), len(data3))
    ax.set_xlim(0, max_len)

    ax.legend()


def plot_npy_files_gt(folder1, folder2, folder4, filename, label1, label2, label4, ax):
    # Cargar los datos
    data1 = np.load(folder1)
    data2 = np.load(folder2)
    gt = np.load(folder4)
    data2 = data2[len(data2)-len(gt):]

    # Plot en el subgráfico actual (ax)
    ax.plot(data1, label=label1)
    ax.plot(data2, label=label2)

    ax.plot(gt, label=label4)

    ax.set_xlabel("Frames")
    ax.set_ylabel("Value")
    ax.set_title(f"Comparison for {filename}")
    
    # Ajustar los límites del gráfico
    max_ylim = max(max(data1), max(data2), max(gt))
    ax.set_ylim(0, 1.1)
    max_len = len(gt)
    ax.set_xlim(0, max_len)

    ax.legend()


# Función para seleccionar archivos aleatorios de un archivo .list
def seleccionar_aleatorios(archivo_list, num_selecciones=4):
    # Leer el archivo .list y obtener todos los archivos como una lista
    names = {}
    seleccionados = []
    with open(archivo_list, 'r') as f:
        archivos = f.readlines()
        # Limpiar los nombres de archivo (quitar saltos de línea)
        archivos = [archivo.strip() for archivo in archivos]

        while len(names) < num_selecciones:
            
            # Seleccionar aleatoriamente 'num_selecciones' archivos de la lista
            temp = random.sample(archivos, 1)[0]
            name = temp.split('/')[-1].split('_')[0]
            if name not in names:
                names[name] = 1
                seleccionados.append(temp)
            


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
    # name = "Vandalism028"
    filename = f"{name}_pred.npy"
    # filename = "Vandalism028_pred.npy"

    PEL4VAD = f'frame_label/predictions/{mode}/30fps/{filename}'
    URDMU = f'frame_label/predictions/{mode}/30fps/extended/{filename}'
    
    
    label1 = "w/o context"
    label2 = "w/ context"

    if mode == "test":
      name = filename.split('_pred.npy')[0]
      gt = f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/gt/{name}_x264_pred.npy'
      plot_npy_files_gt(PEL4VAD, URDMU, gt, filename, label1, label2, label4="GT", ax=ax[i])
    

# Ajustar la distribución de los subgráficos
plt.tight_layout()
plt.show()
