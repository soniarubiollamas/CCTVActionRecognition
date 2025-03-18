from video_player import attach_video_player_to_figure
import matplotlib.pyplot as plt
import numpy as np

filename = "pistola_balines"
video_path = f"C:/Users/xoni/Documents/1_uni/TFM/video/{filename}.mp4"

mode = "test"
PEL4VAD_path = f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/xoni/{filename}_pred.npy'

predictions_PEL4VAD = np.load(PEL4VAD_path)
predictions_PEL4VAD = np.repeat(predictions_PEL4VAD, 16)


PEL4VAD = []
URDMU = []

for frame_num, pred_value in enumerate(predictions_PEL4VAD):
    timestamp = frame_num / 30  # Calculate timestamp based on frame number and FPS
    PEL4VAD.append((timestamp, pred_value))



def on_frame(video_timestamp):
    # Asegúrate de que 'line' no sea None
    if line is None:
        return

    timestamps_PEL4VAD, y_PEL4VAD = zip(*PEL4VAD)
    x_PEL4VAD = [timestamp - video_timestamp for timestamp in timestamps_PEL4VAD]

    # Limpiar el gráfico antes de actualizar
    ax.cla()
    
    # Redibujar el eje de referencia
    plt.axvline(x=0, color='k', linestyle='--')
    
    # Establecer límites del gráfico
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, max(y_PEL4VAD) + 1)

    # Dibujar los puntos de PEL4VAD y URDMU
    ax.scatter(x_PEL4VAD, y_PEL4VAD, color='blue', marker='o', s=10, label="PEL4VAD")

    # Actualizar el gráfico
    line.set_data([], [])  # Reiniciar los datos de la línea
    ax.relim()
    ax.autoscale_view()
    ax.figure.canvas.draw()

    
fig, ax = plt.subplots()
plt.xlim(-5, 5)
plt.axvline(x=0, color='k', linestyle='--')

line, = ax.plot([], [], color='blue')  # Inicializa correctamente 'line'
attach_video_player_to_figure(fig, video_path, on_frame)

plt.show()
