
import time
from utils import fixed_smooth, slide_smooth
from test import *
import pdb


def adjust_size(v_input, target_size):
    # Ajusta el tamaño de v_input para que coincida con target_size
    if v_input.shape[1] > target_size:
        print(f"El tamaño de v_input es mayor que el tamaño objetivo: {v_input.shape[1]} > {target_size}")
        v_input = v_input[:, :target_size, :]
    return v_input


def infer_func(model, dataloader, gt, logger, cfg):
    st = time.time()
    with torch.no_grad():
        model.eval()
        
        normal_preds = torch.zeros(0) #.cuda()
        normal_labels = torch.zeros(0) #.cuda()
        gt_tmp = torch.tensor(gt.copy()) #.cuda()

        start_time_load_dataset = time.time()

        for i, (v_input, name) in enumerate(dataloader):

            end_time_load_dataset = time.time() - start_time_load_dataset
            print(f"Dataset loading time: {end_time_load_dataset:.4f} seconds")
            pred = torch.zeros(0) #.cuda()

            name = name[0].split('_x264')[0].split('quince/')[-1]
            
            print(f"Lote {i}, {name}: {v_input.shape}")

            v_input = v_input.float() #.cuda(non_blocking=True)
            v_input = v_input.squeeze(0)

            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)

            time_start = time.time()
            logits, _ = model(v_input, seq_len)
            time_end = time.time() - time_start
            
            logits = torch.mean(logits, 0)
            logits = logits.squeeze(dim=-1)

            seq = len(logits)
            if cfg.smooth == 'fixed':
                logits = fixed_smooth(logits, cfg.kappa)
            elif cfg.smooth == 'slide':
                logits = slide_smooth(logits, cfg.kappa)
            else:
                pass
            logits = logits[:seq]

            pred = torch.cat((pred, logits))
    
            # pred_save = list(pred.cpu().detach().numpy())
            rate = 2
            # pred_save = np.repeat(pred_save, 16*rate)

            # name = name[0].split('/')[1].split('_x264')[0]
            # name = name[0].split('/')[-1].split('_x264')[0]
            name = name.split('/')[-1].split('_i3d')[0]

            # save predictions in .npy file
            # np.save(f"predictions/train/quince/{name}_pred.npy", pred_save)
            print(f'file {name}_pred.npy saved')
            # pred_save = torch.zeros(0).cuda()
            labels = gt_tmp[: seq_len[0]*16*rate]
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            gt_tmp = gt_tmp[seq_len[0]*16*rate:]

        # pred = list(pred.cpu().detach().numpy())
        far = cal_false_alarm(normal_labels, normal_preds)
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16*rate))
        roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16*rate))
        pr_auc = auc(rec, pre)

    time_elapsed = time.time() - st
    logger.info('offline AUC:{:.4f} AP:{:.4f} FAR:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
        roc_auc, pr_auc, far, time_elapsed // 60, time_elapsed % 60))
    logger.info(f"Prediction time for is : {time_elapsed:.4f} seconds")

    return time_elapsed, time_end
