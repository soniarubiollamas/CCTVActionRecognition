import numpy as np
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

import pandas as pd

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

def calculate_window_metrics(predictions, gt, window_size, threshold=0.5, anomaly_percentage=0.5):
    excel_data = {
        "Video": [],
        "Total_frames": [],
        "Total_windows": [],
        "Windows_anomaly_gt": [],
        "Windows_3_correct": [],
        "Windows_abnormal_3_correct": [],
        "PEL4VAD_correct_abnormal_windows": [],
        "UR-DMU_correct_abnormal_windows": [],
        "BNWVAD_correct_abnormal_windows": [],
        "ROC AUC PEL4VAD": [],
        "ROC AUC UR-DMU": [],
        "ROC AUC BNWVAD": [],
        "PR_AUC PEL4VAD": [],
        "PR_AUC URDMU": [],
        "PR_AUC BNWVAD": []
    }

    for video, preds in predictions.items():
        if video not in gt:
            print(f"Ground truth no disponible para el video {video}.")
            continue

        video_length = len(gt[video])
        num_windows = int(np.ceil(video_length / window_size))
        windows_gt = []
        windows_by_algo = {algo: [] for algo in preds.keys()}

        # Compute ground truth windows
        for i in range(num_windows):
            start = i * window_size
            end = min(start + window_size, video_length)
            gt_window = (np.sum(gt[video][start:end]) / (end - start)) >= anomaly_percentage
            windows_gt.append(gt_window)
        num_anomaly_gt = sum(windows_gt)
        gt_save = np.repeat(windows_gt, window_size)
        np.save(f'predictions/test/windows/{video}_gt.npy', gt_save)

        for algorithm, pred in preds.items():
            algo_windows = []

            for i in range(num_windows):
                start = i * window_size
                end = min(start + window_size, video_length)

                binary_pred = (pred[start:end] >= threshold).astype(int)
                consensus_pred = (np.sum(binary_pred) / (end - start)) >= anomaly_percentage
                algo_windows.append(consensus_pred)
            algo_windows_save = np.repeat(algo_windows, window_size)
            np.save(f'predictions/test/windows/{algorithm}/{video}_pred.npy', algo_windows_save)
            windows_by_algo[algorithm] = algo_windows

        # Compute correct abnormal windows per algorithm
        correct_windows = {algo: 0 for algo in preds.keys()}
        for i in range(num_windows):
            for algo in preds.keys():
                if windows_gt[i] and windows_by_algo[algo][i]:
                    correct_windows[algo] += 1

       # Compute windows where all 3 algorithms agree and are correct
        agreement = sum(
            all(windows_by_algo[algo][i] == windows_gt[i] for algo in preds.keys())
            for i in range(num_windows)
        )

        # Compute abnormal windows where all 3 algorithms agree, are correct, and the ground truth indicates anomaly
        abnormal_agreement = sum(
            all(windows_by_algo[algo][i] == windows_gt[i] for algo in preds.keys()) and windows_gt[i]
            for i in range(num_windows)
        )


        # Calculate AUC metrics for each algorithm
        roc_aucs = {}
        pr_aucs = {}
        windows_gt = np.repeat(windows_gt, window_size)

        for algo, algo_preds in windows_by_algo.items():
            try:
                if algo not in roc_aucs:
                    roc_aucs[algo] = {}
                    
                algo_preds = np.repeat(algo_preds, window_size)
                fpr, tpr, thresholds = roc_curve(windows_gt,algo_preds)
                roc_aucs[algo] = auc(fpr, tpr)
                precision, recall, _ = precision_recall_curve(windows_gt, algo_preds)
                pr_aucs[algo] = auc(recall, precision)
            except ValueError:
                roc_aucs[algo] = None
                pr_aucs[algo] = None

        # Store results
        excel_data["Video"].append(video)
        excel_data["Total_frames"].append(video_length)
        excel_data["Total_windows"].append(num_windows)
        excel_data["Windows_anomaly_gt"].append(num_anomaly_gt)
        excel_data["Windows_3_correct"].append(agreement)
        excel_data["Windows_abnormal_3_correct"].append(abnormal_agreement)
        excel_data["PEL4VAD_correct_abnormal_windows"].append(correct_windows.get("PEL4VAD", 0))
        excel_data["UR-DMU_correct_abnormal_windows"].append(correct_windows.get("URDMU", 0))
        excel_data["BNWVAD_correct_abnormal_windows"].append(correct_windows.get("BNWVAD", 0))
        excel_data["ROC AUC PEL4VAD"].append(roc_aucs.get("PEL4VAD"))
        excel_data["ROC AUC UR-DMU"].append(roc_aucs.get("URDMU"))
        excel_data["ROC AUC BNWVAD"].append(roc_aucs.get("BNWVAD"))
        excel_data["PR_AUC PEL4VAD"].append(pr_aucs.get("PEL4VAD"))
        excel_data["PR_AUC URDMU"].append(pr_aucs.get("URDMU"))
        excel_data["PR_AUC BNWVAD"].append(pr_aucs.get("BNWVAD"))

    return pd.DataFrame(excel_data)

# Ejemplo de uso
folders = {
    'PEL4VAD': f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/test',
    'URDMU': f'C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/frame_label/predictions/test',
    'BNWVAD': f'C:/Users/xoni/Documents/1_uni/1_insterships/BN-WVAD-main/frame_label/predictions/test_normalized'
}

gt_path = f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/gt/'

predictions, gt = load(folders, gt_path)

# Definir tamaño de ventana y umbral 
window_size = 30  # Tamaño de la ventana en frames
threshold = 0.5
anomaly_percentage = 0.2  # Porcentaje de anomalías para considerar una ventana como anómala

# Calcular métricas a nivel de ventana
window_metrics_df = calculate_window_metrics(predictions, gt, window_size, threshold, anomaly_percentage)

# Guardar resultados
window_metrics_df.to_excel('predictions/window_metrics_results.xlsx', index=False)
print(window_metrics_df)
