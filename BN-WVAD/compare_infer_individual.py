import os
import numpy as np
import matplotlib.pyplot as plt

# Función para graficar un archivo .npy con el ground truth y un solo algoritmo
def plot_npy_files_with_gt(gt_path, algo_path, filename, algo_label, algo_color, ax):
    # Cargar los datos
    gt = np.load(gt_path)
    algo_data = np.load(algo_path)

    # Graficar
    ax.plot(gt, label="Ground Truth", linestyle="--", color="red")
    ax.plot(algo_data, label=algo_label, linestyle="-", color=algo_color)

    ax.set_xlabel("Frames")
    ax.set_ylabel("Value")
    filename = filename.split("_")[0]
    ax.set_title(f"{filename} prediction")

    # Ajustar los límites del gráfico
    ax.set_ylim(0, 1.01)
    max_len = len(gt)
    ax.set_xlim(0, max_len)

    ax.legend()

# Parámetros
mode = "test"  # Puede ser "test" o "train"
algorithm = "PEL4VAD"  # Selecciona el algoritmo: "PEL4VAD", "UR-DMU", o "BN-WVAD"
name = "Shooting047"  # Nombre del archivo a procesar
filename = f"{name}_pred.npy"

# Paths
base_path = "C:/Users/xoni/Documents/1_uni/1_insterships/"
gt_path = f"{base_path}cuda_version/PEL4VAD-master/predictions/gt/{name}_x264_pred.npy"

if algorithm == "PEL4VAD":
    algo_path = f"{base_path}cuda_version/PEL4VAD-master/predictions/{mode}/{filename}"
    algo_label = "PEL4VAD"
    algo_color = "blue"
elif algorithm == "UR-DMU":
    algo_path = f"{base_path}UR-DMU/frame_label/predictions/{mode}/30fps/{filename}"
    algo_label = "UR-DMU"
    algo_color = "green"
elif algorithm == "BN-WVAD":
    algo_path = f"frame_label/predictions/{mode}_normalized/30fps/{filename}"
    algo_label = "BN-WVAD"
    algo_color = "yellow"
else:
    raise ValueError("Algoritmo no reconocido. Selecciona 'PEL4VAD', 'UR-DMU' o 'BN-WVAD'.")

# Crear el gráfico
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plot_npy_files_with_gt(gt_path, algo_path, filename, algo_label, algo_color, ax)

# Mostrar el gráfico
plt.tight_layout()
plt.show()
