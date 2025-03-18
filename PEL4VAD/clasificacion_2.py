import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import pandas as pd

mode = "test"
mode_BN = "test_og"

folders = {
    'PEL4VAD': f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}',
    'URDMU': f'C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/frame_label/predictions/{mode}',
}

gt_path = f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/gt/'  # Ruta al archivo de ground truth

def load(folders, gt_path):
    predictions = {}
    gt = {}

    for algorithm, folder in folders.items():
        files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        
        for file in files:
            path = os.path.join(folder, file)
            name = file.split('_pred.npy')[0]
            
            if name not in predictions:
                predictions[name] = {}
                
            predictions[name][algorithm] = np.load(path)
            
            if name not in gt:
                ruta_gt = os.path.join(gt_path, f"{name}_x264_pred.npy")
                if os.path.exists(ruta_gt):
                    gt[name] = np.load(ruta_gt)
                else:
                    print(f"Ground truth para {name} no encontrado.")

    return predictions, gt

def sumar_aciertos_por_frame(matches):
    resultados = {}

    all_two_correct = 0  
    one_correct = 0
    none_correct = 0

    solution = {}
    correct = {}

    for video, data in matches.items():
        pel4vad_preds = data["PEL4VAD"]
        urdmu_preds = data["URDMU"]

        total_frames = len(pel4vad_preds)
        if video not in solution:
            solution[video] = {}
            resultados[video] = {}

        for i in range(total_frames):
            if pel4vad_preds[i] == urdmu_preds[i] == 1:
                all_two_correct += 1
            elif pel4vad_preds[i] != urdmu_preds[i] and (pel4vad_preds[i] == 1 or urdmu_preds[i] == 1):
                one_correct += 1
            elif pel4vad_preds[i] == urdmu_preds[i] == 0:
                none_correct += 1

        solution[video]["all_two_correct"] = all_two_correct
        solution[video]["one_correct"] = one_correct
        solution[video]["none_correct"] = none_correct
            
        resultados[video]["PEL4VAD"] = sum(pel4vad_preds)
        resultados[video]["URDMU"] = sum(urdmu_preds)

        all_two_correct = 0
        one_correct = 0
        none_correct = 0

    return resultados, solution

def clasification(predictions, gt, threshold=0.5):
    matches = {}
    roc_auc_scores = {}

    for video, preds in predictions.items():
        if video not in matches:
                matches[video] = {}

        if video not in gt:
            print(f"Ground truth no disponible para el video {video}.")
            continue
        
        if video not in roc_auc_scores:
            roc_auc_scores[video] = {}

        for algorithm, pred in preds.items():
            if len(pred) < len(gt[video]):
                # rellenar prediccion con el ultimo valor
                last_value = pred[-1]
                prediction_resized = np.zeros(len(gt[video]))
                last_frame = len(pred)
                prediction_resized[:last_frame] = pred
                prediction_resized[last_frame:] = last_value
                pred = prediction_resized
            if len(pred) > len(gt[video]):
                pred = pred[:len(gt[video])]
            
            if "GT" not in matches:
                matches[video]["GT"] = gt[video]

            binary_prediction = (pred >= threshold).astype(int)
            matches[video][algorithm] = (binary_prediction == gt[video]).astype(int)
            
            # ROC AUC score
            fpr, tpr, thresholds = roc_curve(gt[video], pred)
            roc_auc = auc(fpr, tpr)
            roc_auc_scores[video][algorithm] = roc_auc
            
        # hacer el conteo de matches
    resultados_aciertos, solucion = sumar_aciertos_por_frame(matches)

    
    # Crear un DataFrame para guardar en Excel
    excel_data = {
        
        "Video": [],
        "Total_frames": [],
        "2_algorithm": [],
        "1_algorithm": [],
        "none": [],
        "PEL4VAD": [],
        "UR-DMU": [],
        "ROC AUCPEL4VAD": [],
        "ROC AUC UR-DMU": []
    }

    for video in solucion.keys():
        
        excel_data["Video"].append(video)
        excel_data["Total_frames"].append(len(matches[video]["GT"]))
        excel_data["2_algorithm"].append(solucion[video]["all_two_correct"])
        excel_data["1_algorithm"].append(solucion[video]["one_correct"])
        excel_data["none"].append(solucion[video]["none_correct"])
        excel_data["PEL4VAD"].append(resultados_aciertos[video]["PEL4VAD"])
        excel_data["UR-DMU"].append(resultados_aciertos[video]["URDMU"])
        excel_data["ROC AUC PEL4VAD"].append(roc_auc_scores[video]["PEL4VAD"])
        excel_data["ROC AUC UR-DMU"].append(roc_auc_scores[video]["URDMU"])

    # Crear el DataFrame de pandas
    df = pd.DataFrame(excel_data)

    # Guardar el DataFrame en un archivo Excel
    df.to_excel('predictions/resultados_clasificacion.xlsx', index=False)

    # Mostrar el DataFrame
    print(df)








    # for roc_auc in roc_auc_scores:
    #     print(f"Video: {roc_auc}")
    #     for algorithm, score in roc_auc_scores[roc_auc].items():
    #         print(f"{algorithm}: {score}")
    #     print()
        
    # for video, data in solucion.items():
    #     print(f"Video: {video}")
    #     print(f"Total de frames: {len(matches[video]['GT'])}")
    #     print(f"Frames con ambas correctas: {data['all_two_correct']}")
    #     print(f"Frames con una correcta: {data['one_correct']}")
    #     print(f"Frames con ninguna correcta: {data['none_correct']}")
       
    
    # for video, data in resultados_aciertos.items():
    #     print(f"Video: {video}")
    #     print(f"PEL4VAD: {data['PEL4VAD']}")
    #     print(f"URDMU: {data['URDMU']}")
    #     print(f"GT: {sum(matches[video]['GT'])}")
    #     print()

    

# Cargar las predicciones y ground truth
predictions, gt = load(folders, gt_path)

# Aplicar votación de consenso a los modelos
clasification(predictions, gt, threshold=0.5)
