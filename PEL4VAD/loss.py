import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import gen_label


def CLAS2(logits, label, seq_len, criterion):
    logits = logits.squeeze()
    ins_logits = torch.zeros(0).cuda()  # tensor([])
    # label y seq_len son de tamaño 128 porque es el tamaño del batch size
    for i in range(logits.shape[0]):
        if label[i] == 0:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=1, largest=True)
        else:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
        tmp = torch.mean(tmp).view(1)
        ins_logits = torch.cat((ins_logits, tmp))

    clsloss = criterion(ins_logits, label)
    return clsloss

    
def CLAS2_weighted(logits, label, seq_len, criterion):
    logits = logits.squeeze()

    total_loss = 0
    w1 = 10
    w2 = 1

    for i in range(logits.shape[0]):
        if label[i] == 0:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=1, largest=True)
            weight = w2
        else:
            # se guardan los k valores más altos
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
            weight = w1

        tmp = torch.mean(tmp).view(1) # valor de la predicción haciendo la media de los k valores más altos
        # ins_logits = torch.cat((ins_logits, tmp)) # se concatenan los valores para luego pasarlos a la función de pérdida
        # para ponderar, no tengo que acumularlo a los anteriores, porque se pondera por frame, por tanto tengo que
        # calcular la perdida por frame, y luego ponderar. 

        # se calcula la perdida para cada frame, no para el conjunto entero, por eso luego hay que ponderar
        label_i = label[i].view(1) # valor, sino no lo pilla
        frame_loss = criterion(tmp,label_i)*weight
        total_loss += frame_loss

    clsloss = total_loss / logits.shapeyt[0]
    return clsloss

def CLAS2_with_focal_tversky(logits, label, seq_len, alpha=0.7, beta=0.3, gamma=4/3):

    logits = logits.squeeze()
    ins_logits = torch.zeros(0).cuda()  # tensor([]) on GPU
    
    # Process each sequence in the batch
    for i in range(logits.shape[0]):
        if label[i] == 0:
            # For normal sequences, take the top 1
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=1, largest=True)
        else:
            # For anomaly sequences, take the top k (based on length of sequence)
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
        
        # Average over the top-k selected values
        tmp = torch.mean(tmp).view(1)
        ins_logits = torch.cat((ins_logits, tmp))

    

    # Apply sigmoid to logits to get probabilities
    # ins_logits = torch.sigmoid(ins_logits)
    clsloss = focal_tversky_loss(ins_logits, label.float(), alpha=alpha, beta=beta, gamma=gamma)
    
    return clsloss

def focal_tversky_loss(y_pred, y_true, alpha=0.7, beta=0.3, gamma=4/3):

    # Smooth term to avoid division by zero
    smooth = 1e-6

    # Flatten the tensors to calculate TP, FP, FN
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    # Calculate true positives, false positives, and false negatives
    TP = (y_pred * y_true).sum()
    FP = ((1 - y_true) * y_pred).sum()
    FN = (y_true * (1 - y_pred)).sum()

    # Calculate the Tversky Index
    TI = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)

    # Calculate the Focal Tversky Loss
    FTL = torch.pow((1 - TI), gamma)

    return FTL


def KLV_loss(preds, label, criterion):
    preds = F.softmax(preds, dim=1)
    preds = torch.log(preds)
    if torch.isnan(preds).any():
        loss = 0
    else:
        # preds = F.log_softmax(preds, dim=1)
        target = F.softmax(label * 10, dim=1)
        loss = criterion(preds, target)

    return loss


def temporal_smooth(arr):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2-arr)**2)
    return loss


def temporal_sparsity(arr):
    loss = torch.sum(arr)
    # loss = torch.mean(torch.norm(arr, dim=0))
    return loss


def Smooth(logits, seq_len, lamda=8e-5):
    smooth_mse = []
    for i in range(logits.shape[0]):
        tmp_logits = logits[i][:seq_len[i]-1]
        sm_mse = temporal_smooth(tmp_logits)
        smooth_mse.append(sm_mse)
    smooth_mse = sum(smooth_mse) / len(smooth_mse)

    return smooth_mse * lamda


def Sparsity(logits, seq_len, lamda=8e-5):
    spar_mse = []
    for i in range(logits.shape[0]):
        tmp_logits = logits[i][:seq_len[i]]
        sp_mse = temporal_sparsity(tmp_logits)
        spar_mse.append(sp_mse)
    spar_mse = sum(spar_mse) / len(spar_mse)

    return spar_mse * lamda


def Smooth_Sparsity(logits, seq_len, lamda=8e-5):
    smooth_mse = []
    spar_mse = []
    for i in range(logits.shape[0]):
        tmp_logits = logits[i][:seq_len[i]]
        sm_mse = temporal_smooth(tmp_logits)
        sp_mse = temporal_sparsity(tmp_logits)
        smooth_mse.append(sm_mse)
        spar_mse.append(sp_mse)
    smooth_mse = sum(smooth_mse) / len(smooth_mse)
    spar_mse = sum(spar_mse) / len(spar_mse)

    return (smooth_mse + spar_mse) * lamda