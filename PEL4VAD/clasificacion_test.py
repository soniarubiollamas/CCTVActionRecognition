import numpy as np
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd

mode = "test"
mode_BN = "test_normalized"

folders = {
    'PEL4VAD': f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}',
    'URDMU': f'C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/frame_label/predictions/{mode}',
    'BNWVAD': f'C:/Users/xoni/Documents/1_uni/1_insterships/BN-WVAD-main/frame_label/predictions/{mode_BN}'  # Asegúrate de ajustar la ruta
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

def sumar_aciertos_por_frame(matches, gt):
    resultados = {}

    all_three_correct = 0
    correct_abnormal_frame = {}
    correct_normal_frame =  {}
    total_abnormal_frames = {}
    two_correct = 0  
    one_correct = 0
    none_correct = 0
    abnormal = 0
    normal = 0

    solution = {}

    for video, data in matches.items():
        pel4vad_preds = data["PEL4VAD"]
        urdmu_preds = data["URDMU"]
        bnwvad_preds = data["BNWVAD"]


        total_frames = len(pel4vad_preds)
        if video not in solution:
            solution[video] = {}
            resultados[video] = {}
            correct_abnormal_frame[video] = []
            correct_normal_frame[video] =[]
            total_abnormal_frames[video] = []


        total_abnormal_frames[video] = sum(gt[video])
        for i in range(total_frames):
            if pel4vad_preds[i] == urdmu_preds[i] == bnwvad_preds[i] == 1:
                all_three_correct += 1
                if gt[video][i] == 1:
                    abnormal += 1
                else:
                    normal += 1
            elif (
                (pel4vad_preds[i] == 1 and urdmu_preds[i] == 1) or
                (pel4vad_preds[i] == 1 and bnwvad_preds[i] == 1) or
                (urdmu_preds[i] == 1 and bnwvad_preds[i] == 1)
            ):
                two_correct += 1
            elif (
                pel4vad_preds[i] == 1 or
                urdmu_preds[i] == 1 or
                bnwvad_preds[i] == 1
            ):
                one_correct += 1
            else:
                none_correct += 1

        solution[video]["all_three_correct"] = all_three_correct
        solution[video]["two_correct"] = two_correct
        solution[video]["one_correct"] = one_correct
        solution[video]["none_correct"] = none_correct

        correct_abnormal_frame[video] = abnormal
        correct_normal_frame[video] = normal
            
        resultados[video]["PEL4VAD"] = sum(pel4vad_preds)
        resultados[video]["URDMU"] = sum(urdmu_preds)
        resultados[video]["BNWVAD"] = sum(bnwvad_preds)

        all_three_correct = 0
        two_correct = 0
        one_correct = 0
        none_correct = 0
        abnormal = 0
        normal = 0

    return resultados, solution, correct_abnormal_frame, correct_normal_frame, total_abnormal_frames

def clasification(predictions, gt, threshold=0.5):
    matches = {}
    roc_auc_scores = {}
    pr_auc_scores = {}
    correct_abnormal_count = {
        "PEL4VAD": {},
        "URDMU": {},
        "BNWVAD": {}
    }
    false_alarms = {
        "PEL4VAD": {},
        "URDMU": {},
        "BNWVAD": {}
    }

    for video, preds in predictions.items():
        if video not in matches:
                matches[video] = {}

        if video not in gt:
            print(f"Ground truth no disponible para el video {video}.")
            continue
        
        if video not in roc_auc_scores:
            roc_auc_scores[video] = {}
            pr_auc_scores[video] = {}

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

            # calculate correct abnormal count for each algorithm
            abnormal_indices = np.where(gt[video] == 1)[0]
            correct_abnormal_count[algorithm][video] = np.sum((binary_prediction[abnormal_indices] == gt[video][abnormal_indices]))

            #calculate number of false alarms
            false_alarm_indices = np.where(gt[video] == 0)[0]
            false_alarms[algorithm][video] = np.sum((binary_prediction[false_alarm_indices] != gt[video][false_alarm_indices]))

            
            # ROC AUC score
            fpr, tpr, thresholds = roc_curve(gt[video], pred)
            roc_auc = auc(fpr, tpr)
            roc_auc_scores[video][algorithm] = roc_auc

            precision, recall, _ = precision_recall_curve(gt[video], pred)
            pr_auc_scores[video][algorithm] = auc(recall, precision)
            
        # hacer el conteo de matches
    resultados_aciertos, solucion, correct_abnormal_frame, correct_normal_frame, total_abnormal_frames = sumar_aciertos_por_frame(matches, gt)

    
    # Crear un DataFrame para guardar en Excel
    excel_data = {
        "Video": [],
        "Total_frames": [],
        "3_algorithm": [],
        "2_algorithm": [],
        "1_algorithm": [],
        "none": [],
        "total_abnormal_frames": [],
        "correct_abnormal_frame": [],
        "correct_normal_frame": [],
        "PEL4VAD": [],
        "PEL4VAD_correct_abnormal": [],
        "PEL4VAD false_alarms": [],
        "UR-DMU": [],
        "UR-DMU_correct_abnormal": [],
        "UR-DMU false_alarms": [],
        "BNWVAD": [],
        "BNWVAD_correct_abnormal": [],
        "BNWVAD false_alarms": [],
        "ROC PEL4VAD": [],
        "PR PEL4VAD": [],
        "ROC UR-DMU": [],
        "PR UR-DMU": [],
        "ROC BNWVAD": [],
        "PR BNWVAD": []
    }

    for video in solucion.keys():
        excel_data["Video"].append(video)
        excel_data["Total_frames"].append(len(matches[video]["GT"]))
        excel_data["3_algorithm"].append(solucion[video]["all_three_correct"])
        excel_data["2_algorithm"].append(solucion[video]["two_correct"])
        excel_data["1_algorithm"].append(solucion[video]["one_correct"])
        excel_data["none"].append(solucion[video]["none_correct"])
        excel_data["total_abnormal_frames"].append(total_abnormal_frames[video])
        excel_data["correct_abnormal_frame"].append(correct_abnormal_frame[video])
        excel_data["correct_normal_frame"].append(correct_normal_frame[video])
        excel_data["PEL4VAD"].append(resultados_aciertos[video]["PEL4VAD"])
        excel_data["PEL4VAD_correct_abnormal"].append(correct_abnormal_count["PEL4VAD"][video])
        excel_data["PEL4VAD false_alarms"].append(false_alarms["PEL4VAD"][video])
        excel_data["UR-DMU"].append(resultados_aciertos[video]["URDMU"])
        excel_data["UR-DMU_correct_abnormal"].append(correct_abnormal_count["URDMU"][video])
        excel_data["UR-DMU false_alarms"].append(false_alarms["URDMU"][video])
        excel_data["BNWVAD"].append(resultados_aciertos[video]["BNWVAD"])
        excel_data["BNWVAD_correct_abnormal"].append(correct_abnormal_count["BNWVAD"][video])
        excel_data["BNWVAD false_alarms"].append(false_alarms["BNWVAD"][video])
        excel_data["ROC PEL4VAD"].append(roc_auc_scores[video]["PEL4VAD"])
        excel_data["PR PEL4VAD"].append(pr_auc_scores[video]["PEL4VAD"])
        excel_data["ROC UR-DMU"].append(roc_auc_scores[video]["URDMU"])
        excel_data["PR UR-DMU"].append(pr_auc_scores[video]["URDMU"])
        excel_data["ROC BNWVAD"].append(roc_auc_scores[video]["BNWVAD"])
        excel_data["PR BNWVAD"].append(pr_auc_scores[video]["BNWVAD"])

    # Crear el DataFrame de pandas
    df = pd.DataFrame(excel_data)

    # Guardar el DataFrame en un archivo Excel
    df.to_excel('predictions/resultados_clasificacion_test_26dic.xlsx', index=False)

    # Mostrar el DataFrame
    print(df)
    

# Cargar las predicciones y ground truth
predictions, gt = load(folders, gt_path)

# Aplicar votación de consenso a los modelos
clasification(predictions, gt, threshold=0.5)