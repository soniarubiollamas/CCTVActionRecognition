import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import pandas as pd

mode = "train"
mode_BN = "train_normalized"

folders = {
    'PEL4VAD': f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}',
    'URDMU': f'C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/frame_label/predictions/{mode}',
    'BNWVAD': f'C:/Users/xoni/Documents/1_uni/1_insterships/BN-WVAD-main/frame_label/predictions/{mode_BN}'  # Asegúrate de ajustar la ruta
}


def load(folders):
    predictions = {}

    for algorithm, folder in folders.items():
        files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        
        for file in files:
            path = os.path.join(folder, file)
            name = file.split('_pred.npy')[0]
            
            if name not in predictions:
                predictions[name] = {}
                
            predictions[name][algorithm] = np.load(path)

    return predictions

def sumar_aciertos_por_frame(matches):
    agree = {}

    three_agree = 0
    two_agree = 0  


    new_gt = {}
    anomaly = {}

    for video, data in matches.items():
        pel4vad_preds = data["PEL4VAD"]
        urdmu_preds = data["URDMU"]
        bnwvad_preds = data["BNWVAD"]


        total_frames = len(pel4vad_preds)
        if video not in agree:
            agree[video] = {}
            new_gt[video] = {}
            anomaly[video] = {}

        for i in range(total_frames):
            if pel4vad_preds[i] == urdmu_preds[i] == bnwvad_preds[i]:
                three_agree += 1
                new_gt[video][i] = pel4vad_preds[i]
            elif (
                (pel4vad_preds[i] == urdmu_preds[i]) or
                (pel4vad_preds[i] == bnwvad_preds[i]) or
                (urdmu_preds[i] == bnwvad_preds[i])
            ):
                two_agree += 1
                if (pel4vad_preds[i] == urdmu_preds[i]):
                    new_gt[video][i] = pel4vad_preds[i]
                elif (pel4vad_preds[i] == bnwvad_preds[i]):
                    new_gt[video][i] = pel4vad_preds[i]
                elif (urdmu_preds[i] == bnwvad_preds[i]):
                    new_gt[video][i] = urdmu_preds[i]

        agree[video]["three_agree"] = three_agree
        agree[video]["two_agree"] = two_agree
            
        # Convertir el diccionario new_gt[video] a una lista de valores ordenados por frame
        gt_values = [new_gt[video][i] for i in sorted(new_gt[video])]
        np.save(f"predictions/gt/train/{video}_x264_pred.npy", np.array(gt_values))

        num_ones = sum(gt_values)
        num_zeros = total_frames - num_ones
        anomaly[video] = 1 if num_ones > num_zeros else 0

        three_agree = 0
        two_agree = 0

        print(f"Video: {video}, Anomaly: {anomaly[video]}")

    return agree, anomaly

def clasification(predictions, threshold=0.5):
    prediction = {}

    for video, preds in predictions.items():
        if video not in prediction:
                prediction[video] = {}

        for algorithm, pred in preds.items():


            binary_prediction = (pred >= threshold).astype(int)
            prediction[video][algorithm] = binary_prediction
            
            
        # hacer el conteo de matches
    resultados_agree, anomaly = sumar_aciertos_por_frame(prediction)

    
    # Crear un DataFrame para guardar en Excel
    excel_data = {
        "Video": [],
        "Total_frames": [],
        "3_agree": [],
        "2_agree": [],
        "anomaly" : []
    }

    for video in resultados_agree.keys():
        excel_data["Video"].append(video)
        excel_data["Total_frames"].append(len(predictions[video]["PEL4VAD"]))
        excel_data["3_agree"].append(resultados_agree[video]["three_agree"])
        excel_data["2_agree"].append(resultados_agree[video]["two_agree"])
        excel_data["anomaly"].append(anomaly[video])
        
    # Crear el DataFrame de pandas
    df = pd.DataFrame(excel_data)

    # Guardar el DataFrame en un archivo Excel
    df.to_excel('predictions/resultados_clasificacion_train.xlsx', index=False)

    # Mostrar el DataFrame
    print(df)
    

# Cargar las predicciones y ground truth
predictions = load(folders)

# Aplicar votación de consenso a los modelos
clasification(predictions, threshold=0.5)