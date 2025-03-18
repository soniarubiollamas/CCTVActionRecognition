import pandas as pd
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    auc,
    precision_score,
)
import os
import numpy as np

mode = "test"
mode_BN = "test_normalized"

carpetas_algoritmos = {
    'PEL4VAD': f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}',
    'URDMU': f'C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/frame_label/predictions/{mode}',
    'BNWVAD': f'C:/Users/xoni/Documents/1_uni/1_insterships/BN-WVAD-main/frame_label/predictions/{mode_BN}'
}

gt_path = f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/gt/test'  # Ruta al archivo de ground truth

# Función para cargar predicciones y ground truth de cada video
def load_and_concatenate_files(pred_folder, gt_folder, mode="test"):
    preds_list = []
    gt_list = []

    for pred_file in os.listdir(gt_folder):
        pred_file_pred = pred_file.replace("x264_", "")
        name = pred_file_pred.split("_pred")[0]

        ###### si no quiero hacer el infer con los videos normales ######
        if name == "Normal" and mode == "train":
            continue
        ###############################################################

        pred_path = os.path.join(pred_folder, pred_file_pred)
        gt_path = os.path.join(gt_folder, pred_file)

        if os.path.isfile(pred_path) and os.path.isfile(gt_path):
            preds = np.load(pred_path)
            gt = np.load(gt_path)

            if len(preds) < len(gt):
                # truncar gt
                gt = gt[:len(preds)]

            if len(preds) > len(gt):
                gt = np.append(gt, np.repeat(gt[-1], len(preds) - len(gt)))

            preds_list.append(preds)
            gt_list.append(gt)
        else:
            print(f"Pred file or GT file not found for {pred_file}")

    concatenated_preds = np.concatenate(preds_list)
    concatenated_gt = np.concatenate(gt_list)

    return concatenated_preds, concatenated_gt

predictions_PEL4VAD, ground_truth_PEL4VAD = load_and_concatenate_files(carpetas_algoritmos['PEL4VAD'], gt_path)
predictions_URDMU, ground_truth_URDMU = load_and_concatenate_files(carpetas_algoritmos['URDMU'], gt_path)
predictions_BNWVAD, ground_truth_BNWVAD = load_and_concatenate_files(carpetas_algoritmos['BNWVAD'], gt_path)
algorithm_names = ["PEL4VAD", "URDMU", "BN-WVAD"]

predictions = [predictions_PEL4VAD, predictions_URDMU, predictions_BNWVAD]
ground_truth = [ground_truth_PEL4VAD, ground_truth_URDMU, ground_truth_BNWVAD]

# Almacenar los resultados
results = []

# Calcular métricas para cada algoritmo
for i, preds in enumerate(predictions):
    # Convertir probabilidades a etiquetas binarias con umbral 0.5
    preds_binary = (preds >= 0.5).astype(int)
    
    # Sensibilidad (Recall)
    recall = recall_score(ground_truth[i], preds_binary)
    
    # Precisión
    precision = precision_score(ground_truth[i], preds_binary)

    # F1 Score
    f1 = f1_score(ground_truth[i], preds_binary)
    
    # AUC-ROC
    fpr, tpr, _ = roc_curve(ground_truth[i], preds)
    roc_auc = auc(fpr, tpr)
    
    # Curva Precision-Recall y AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(ground_truth[i], preds)
    auc_pr = auc(recall_curve, precision_curve)
    
    # Tasa de Falsos Negativos (FNR)
    fn = sum((ground_truth[i] == 1) & (preds_binary == 0))
    tp_fn = sum(ground_truth[i] == 1)
    fnr = fn / tp_fn if tp_fn > 0 else 0
    
    # Tasa de Falsos Positivos (FPR)
    fpr = sum((ground_truth[i] == 0) & (preds_binary == 1)) / sum(ground_truth[i] == 0) if sum(ground_truth[i] == 0) > 0 else 0

    # Tasa de Verdaderos Positivos (TPR)
    tpr = recall  # TPR es lo mismo que la sensibilidad
    
    # Valor de Precisión Promedio (AP)
    average_precision = auc(recall_curve, precision_curve)

    # Guardar resultados en el diccionario
    results.append({
        "Algoritmo": algorithm_names[i],
        "Recall": recall,
        "Precision": precision,
        "F1 Score": f1,
        "AUC-ROC": roc_auc,
        "AUC-PR": auc_pr,
        "FNR": fnr,
        "FPR": fpr,
        "TPR": tpr,
        "AP": average_precision
    })

# Convertir a DataFrame
results_df = pd.DataFrame(results)

# Guardar en un archivo Excel
results_df.to_excel("predictions/nuevas_metricas_algoritmos.xlsx", index=False)

print("Resultados guardados en nuevas_metricas_algoritmos.xlsx")
