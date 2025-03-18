import torch
from options import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def get_predicts(test_loader, net):
    load_iter = iter(test_loader)
    frame_predict = []
    
    real_res = []
    total_frame_count = []
    total_frame_count.append(0)
    names = []
        
    for i in range(len(test_loader.dataset)//5):
        _data, _label, _vid_name = next(load_iter)
        
        _data = _data.cuda()
        _label = _label.cuda()
        res = net(_data)   #shape 5,88

        res = res.cpu().numpy()
        real_res.append(res)

        if i % 2 == 1:
            real_res = np.concatenate(real_res, axis=0)
            a_predict = real_res.mean(0)   
            # a_predict_normalized = (a_predict - np.min(a_predict)) / (np.max(a_predict) - np.min(a_predict))
            a_predict = np.repeat(a_predict, 15)
            fpre_ = np.repeat(a_predict, 16) 
            frame_count = len(fpre_)
            total_frame_count.append(total_frame_count[-1]+frame_count)
            names.append(_vid_name[0].split("/")[-1].split("_x264")[0])

            
            # np.save(f"frame_label/predictions/test_normalized/{_vid_name[0].split('/')[-1].split('_x264')[0]}_pred.npy", fpre_)
            print(f'video {_vid_name[0].split("/")[-1].split("_x264")[0]} predicted')

            frame_predict.append(fpre_)
            real_res = []
            
    frame_predict = np.concatenate(frame_predict, axis=0)
    # print(f'max: {np.max(frame_predict)}, min: {np.min(frame_predict)}')
    frame_predict_normalized = (frame_predict - np.min(frame_predict)) / (np.max(frame_predict) - np.min(frame_predict))

    
    return frame_predict_normalized

def get_metrics(frame_predict, frame_gt):
    metrics = {}
    fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
    metrics['AUC'] = auc(fpr, tpr)
    
    precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
    metrics['AP'] = auc(recall, precision)
    
    return metrics

def test(net, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/gt-ucf-quince.npy")
        
        frame_predicts = get_predicts(test_loader, net)

        metrics = get_metrics(frame_predicts, frame_gt)
        
        test_info['step'].append(step)
        for score_name, score in metrics.items():
            metrics[score_name] = score * 100
            test_info[score_name].append(metrics[score_name])

        return metrics
