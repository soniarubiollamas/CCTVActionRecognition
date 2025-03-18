import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Crear los datos manualmente en un DataFrame
data = {
    'Tasa (fps)': [30, 15, 10, 6, 3, 2, 1],
    'PEL4VAD ROC AUC': [86.76, 87.66, 87.75, 87.41, 85.39, 84.07, 82.33],
    'PEL4VAD AP': [33.74, 36.42, 37.16, 36.83, 34.30, 31.62, 23.70],
    'URDMU ROC AUC': [86.97, 87.43, 87.30, 86.56, 84.74, 84.69, 81.73],
    'URDMU AP': [34.68, 37.45, 23.70, 40.65, 38.73, 41.20, 37.46],
    'BN-WVAD ROC AUC': [83.86, 83.37, 83.58, 83.51, 82.03, 82.40, 81.33],
    'BN-WVAD AP': [25.79, 27.58, 28.28, 29.80, 29.28, 28.65, 25.10]  # "XX" tratado como valor faltante
}

df = pd.DataFrame(data)

# Gráfica para ROC AUC
plt.figure(figsize=(10, 5))
plt.plot(df['Tasa (fps)'], df['PEL4VAD ROC AUC'], marker='o', label='PEL4VAD ROC AUC')
plt.plot(df['Tasa (fps)'], df['URDMU ROC AUC'], marker='o', label='URDMU ROC AUC')
plt.plot(df['Tasa (fps)'], df['BN-WVAD ROC AUC'], marker='o', label='BN-WVAD ROC AUC')
plt.xlabel('FPS')
plt.ylabel('ROC AUC')
plt.title('Evolución de ROC AUC según FPS')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()  # Invertir el eje X para ver los FPS en orden descendente
plt.show()

# Gráfica para AP
plt.figure(figsize=(10, 5))
plt.plot(df['Tasa (fps)'], df['PEL4VAD AP'], marker='o', label='PEL4VAD AP')
plt.plot(df['Tasa (fps)'], df['URDMU AP'], marker='o', label='URDMU AP')
plt.plot(df['Tasa (fps)'], df['BN-WVAD AP'], marker='o', label='BN-WVAD AP')
plt.xlabel('FPS')
plt.ylabel('Average Precision (AP)')
plt.title('Evolución de AP según FPS')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()  # Invertir el eje X para ver los FPS en orden descendente
plt.show()
