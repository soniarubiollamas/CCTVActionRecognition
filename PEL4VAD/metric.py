import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from matplotlib import pyplot as plt
import os

def cal_false_alarm(labels, preds, threshold=0.5):
    """
    Función para calcular la tasa de falsas alarmas.
    """
    # Convertir predicciones a binario según el umbral
    binary_preds = (preds >= threshold).astype(int)
    false_alarms = np.sum((labels == 0) & (binary_preds == 1))
    total_negatives = np.sum(labels == 0)

    if total_negatives == 0:
        return 0
    return false_alarms / total_negatives

def cal_false_alarm2(gt, preds, threshold=0.5):
    # preds = list(preds)
    # gt = list(gt)

    # preds = np.repeat(preds, 16)
    preds_temp = preds
    gt_temp = gt
    preds_temp[preds_temp < threshold] = 0
    preds_temp[preds_temp >= threshold] = 1
    tn, fp, fn, tp = confusion_matrix(gt_temp, preds_temp, labels=[0, 1]).ravel()

    far = fp / (fp + tn)

    return far


def load_and_concatenate_files(pred_folder, gt_folder, mode):
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

            # np.save(f'predictions/gt/test/{name}_pred.npy', gt)
            preds_list.append(preds)
            gt_list.append(gt)
        else:
            print(f"Pred file or GT file not found for {pred_file}")

    concatenated_preds = np.concatenate(preds_list)
    concatenated_gt = np.concatenate(gt_list)

    # np.save(f"list/ucf/ucf-gt-quince.npy", concatenated_gt)

    return concatenated_preds, concatenated_gt

# Define folders
gt_folder = "C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/gt"
mode = "test"
pred_folder = f"predictions/{mode}/1fps"

# Load and concatenate files
preds, gt = load_and_concatenate_files(pred_folder, gt_folder, mode)



fpr, tpr, thresholds = roc_curve(gt, preds)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(gt, preds)
pr_auc = auc(recall, precision)

# Calculate metrics
false_alarm_rate = cal_false_alarm(gt, preds)
false_alarm_rate2 = cal_false_alarm2(gt, preds)

print(f"False Alarm Rate (FAR): {false_alarm_rate}")
print(f"False Alarm Rate (FAR) 2: {false_alarm_rate2}")
print(f"ROC AUC: {roc_auc}")
print(f"AP AUC: {pr_auc}")

# # If you want to save the ROC and PR curves to files, you can do it with np.save
# np.save("fpr.npy", fpr)
# np.save("tpr.npy", tpr)
# np.save("precision.npy", precision)
# np.save("recall.npy", recall)

# Optional: plot the ROC and PR curves
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# plt.figure()
# plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall curve')
# plt.legend(loc="lower left")
# plt.show()
