import torch
import numpy as np
from dataset_loader import XDVideo
from options import parse_args
import pdb
import utils
import os
from models import WSAD
from tqdm import tqdm
from dataset_loader import data
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import prettytable
import time

def get_predict(test_loader, net):
    time1 = time.time()
    load_iter = iter(test_loader)
    frame_predict = []
    real_res = []
    total_frame_count = []
    total_frame_count.append(0)
    names = []
    start_time_all = time.time()

# check i
        
    for i in range(len(test_loader.dataset)//5):
        
        print(f"time to load data: {time.time()-start_time_all}")
        _data, _label, _vid_name = next(load_iter)
        
        _data = _data #.cuda()
        _label = _label #.cuda()
        res = net(_data)   #shape 5,88

        res = res.cpu().numpy()
        real_res.append(res)
        start_time_video = time.time()

        if i % 2 == 1:
            real_res = np.concatenate(real_res, axis=0)
            a_predict = real_res.mean(0)   
            # a_predict_normalized = (a_predict - np.min(a_predict)) / (np.max(a_predict) - np.min(a_predict))
            rate = 1
            fpre_ = np.repeat(a_predict, 16*rate) 
            frame_count = len(fpre_)
            total_frame_count.append(total_frame_count[-1]+frame_count)

            print(f"time to predict video: {(time.time()-start_time_video)*2}")
            names.append(_vid_name[0].split("/")[-1].split("_x264")[0])

            
            # np.save(f"frame_label/predictions/test_normalized/{_vid_name[0].split('/')[-1].split('_x264')[0]}_pred.npy", fpre_)
            print(f'video {_vid_name[0].split("/")[-1].split("_x264")[0]} predicted')

            frame_predict.append(fpre_)
            real_res = []
            
    frame_predict = np.concatenate(frame_predict, axis=0)
    # print(f'max: {np.max(frame_predict)}, min: {np.min(frame_predict)}')
    # frame_predict = np.exp(frame_predict)
    frame_predict_normalized = (frame_predict - np.min(frame_predict)) / (np.max(frame_predict) - np.min(frame_predict))
    print(f"time to predict: {time.time()-time1}")

    for i in range(len(total_frame_count)-1):
        init_frame = total_frame_count[i]
        final_frame = total_frame_count[i+1]
        archivo = frame_predict_normalized[init_frame:final_frame]
        name = names[i]
        # np.save(f"frame_label/predictions/test_normalized/30fps/{name}_pred.npy", archivo)
    return frame_predict_normalized

# def get_sub_metrics(frame_predict, frame_gt):
#     anomaly_mask = np.load('frame_label/xd_anomaly_mask.npy')
#     sub_predict = frame_predict[anomaly_mask]
#     sub_gt = frame_gt[anomaly_mask]
    
#     fpr,tpr,_ = roc_curve(sub_gt, sub_predict)
#     auc_sub = auc(fpr, tpr)

#     precision, recall, th = precision_recall_curve(sub_gt, sub_predict)
#     ap_sub = auc(recall, precision)
#     return auc_sub, ap_sub

def get_metrics(frame_predict, frame_gt):
    metrics = {}
    
    fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
    metrics['AUC'] = auc(fpr, tpr)
    
    precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
    metrics['AP'] = auc(recall, precision)

    # auc_sub, ap_sub = get_sub_metrics(frame_predict, frame_gt)
    # metrics['AUC_sub'] = auc_sub
    # metrics['AP_sub'] = ap_sub

    return metrics

def test(net, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        frame_gt = np.load("frame_label/ucf_gt.npy")
        
        x = time.time()
        frame_predict = get_predict(test_loader, net)
        print("total predicion time is: ", time.time()-x)
        metrics = get_metrics(frame_predict, frame_gt)
        
        test_info['step'].append(step)
        for score_name, score in metrics.items():
            metrics[score_name] = score * 100
            test_info[score_name].append(metrics[score_name])

        return metrics


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()
    worker_init_fn = None
    if args.seed >= 0:
        utils.set_seed(args.seed)
        worker_init_fn = np.random.seed(args.seed)
    net = WSAD(args.len_feature, flag = "Test", args = args)
    #net = net.cuda()
    test_loader = data.DataLoader(
        XDVideo(root_dir = args.root_dir, mode = 'Test', num_segments = args.num_segments, len_feature = args.len_feature),
            batch_size = 5,
            shuffle = False, num_workers = args.num_workers,
            worker_init_fn = worker_init_fn)
    
    test_info = {'step': [], 'AUC': [], 'AP': []}

    res = test(net, test_loader, test_info, 1, model_file = args.model_path)

    pt = prettytable.PrettyTable()
    pt.field_names = ['AUC', 'AP']
    for k, v in res.items():
        res[k] = round(v, 2)
    pt.add_row([res['AUC'], res['AP']])
    print(pt)