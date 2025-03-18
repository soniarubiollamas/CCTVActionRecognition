import numpy as np
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
import re

# Función para calcular la tasa de falsos positivos
def cal_false_alarm(labels, preds, threshold=0.5):
    binary_preds = (preds >= threshold).astype(int)
    false_alarms = np.sum((labels == 0) & (binary_preds == 1))
    total_negatives = np.sum(labels == 0)
    return false_alarms / total_negatives if total_negatives > 0 else 0

# Función para cargar predicciones y ground truth por tipo de anomalía
def load_and_concatenate_files_by_type(pred_folder, gt_folder):
    dict_preds = {}
    dict_gt = {}

    for pred_file in os.listdir(gt_folder):
        pred_file_pred = pred_file.replace("x264_", "")
        name = pred_file_pred.split("_pred")[0]

        anomaly = re.sub(r'\d+$', '', name)

        if anomaly not in dict_preds:
            dict_preds[anomaly] = {}
            dict_gt[anomaly] = {}

        dict_gt[anomaly][name] = []
        dict_preds[anomaly][name] = []

        pred_path = os.path.join(pred_folder, pred_file_pred)
        gt_path = os.path.join(gt_folder, pred_file)

        if os.path.isfile(pred_path) and os.path.isfile(gt_path):
            preds = np.load(pred_path)
            gt = np.load(gt_path)

            # Ajustar la longitud de las predicciones
            if len(preds) < len(gt):
                gt = gt[:len(preds)]
            if len(preds) > len(gt):
                gt = np.append(gt, np.repeat(gt[-1], len(preds) - len(gt)))

            dict_preds[anomaly][name] = preds
            dict_gt[anomaly][name] = gt

    return dict_preds, dict_gt

def calculate_metrics_by_anomaly_type(pred_folder, gt_folder):
    preds_dict, gt_dict = load_and_concatenate_files_by_type(pred_folder, gt_folder)

    results = []
    for anomaly_type, preds_by_name in preds_dict.items():
        gt_by_name = gt_dict[anomaly_type]
        
        all_preds = []
        all_gt = []

        for name, preds in preds_by_name.items():
            gt = gt_by_name.get(name)
            if gt is not None:
                all_preds.extend(preds)
                all_gt.extend(gt)

        all_preds = np.array(all_preds)
        all_gt = np.array(all_gt)
        
        # Calcular métricas si hay datos
        if len(all_preds) > 0 and len(all_gt) > 0:
            # Convertir probabilidades a etiquetas binarias con umbral 0.5
            preds_binary = (all_preds >= 0.5).astype(int)
            
            # Cálculo de FN, FP, TP, TN
            TP = np.sum((all_gt == 1) & (preds_binary == 1))
            TN = np.sum((all_gt == 0) & (preds_binary == 0))
            FP = np.sum((all_gt == 0) & (preds_binary == 1))
            FN = np.sum((all_gt == 1) & (preds_binary == 0))

            # Sensibilidad (Recall)
            recall = recall_score(all_gt, preds_binary)
            
            # Precisión
            precision = precision_score(all_gt, preds_binary)

            # F1 Score
            f1 = f1_score(all_gt, preds_binary)
            
            # AUC-ROC
            fpr, tpr, _ = roc_curve(all_gt, all_preds)
            roc_auc = auc(fpr, tpr)
            
            # Curva Precision-Recall y AUC-PR
            precision_curve, recall_curve, _ = precision_recall_curve(all_gt, all_preds)
            auc_pr = auc(recall_curve, precision_curve)
            
            # Tasa de Falsos Negativos (FNR)
            tp_fn = np.sum(all_gt == 1)
            fnr = FN / tp_fn if tp_fn > 0 else 0
            
            # Tasa de Falsos Positivos (FPR)
            total_negatives = np.sum(all_gt == 0)
            fpr_rate = FP / total_negatives if total_negatives > 0 else 0
            
            # Guardar resultados en la lista
            results.append({
                'Anomaly_Type': anomaly_type,
                'TP': TP,
                'TN': TN,
                'FP': FP,
                'FN': FN,
                'Recall': recall,
                'Precision': precision,
                'F1 Score': f1,
                'AUC-ROC': roc_auc,
                'AUC-PR': auc_pr,
                'FNR': fnr,
                'FPR': fpr_rate
            })

    # Crear DataFrame de resultados
    results_df = pd.DataFrame(results)

    # Guardar el DataFrame en un archivo Excel
    results_df.to_excel("predictions/new_metrics_by_anomaly_type.xlsx", index=False)

    return results_df

# Definir carpetas
gt_folder = "C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/gt"
mode = "test"
pred_folder = f"C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}"

# Calcular métricas por tipo de anomalía y guardar en Excel
metrics_by_type_df = calculate_metrics_by_anomaly_type(pred_folder, gt_folder)
print("Métricas por tipo de anomalía guardadas en 'metrics_by_anomaly_type.xlsx'")
