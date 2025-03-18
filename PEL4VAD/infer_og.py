
import time
from utils import fixed_smooth, slide_smooth
from test import *


def infer_func(model, dataloader, gt, logger, cfg):
    st = time.time()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy()).cuda()

        for i, (v_input, name) in enumerate(dataloader):
            print(f"Lote {i}, {name[0].split('/')[1].split('_')[0]}: {v_input.shape}")
            # if i >= 34:
            #     breakpoint()
            v_input = v_input.float().cuda(non_blocking=True)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            
            time_start = time.time()
            logits, _ = model(v_input, seq_len)
            time_end = time.time() - time_start
            if (name[0].split('/')[1].split('_')[0]=="Shoplifting049"):
                logger.info(f"Prediction time for {name[0].split('/')[1].split('_')[0]} is : {time_end:.4f} seconds")

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
            labels = gt_tmp[: seq_len[0]*16]
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            gt_tmp = gt_tmp[seq_len[0]*16:]

        pred = list(pred.cpu().detach().numpy())
        far = cal_false_alarm(normal_labels, normal_preds)
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
        roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(rec, pre)

    time_elapsed = time.time() - st
    logger.info('offline AUC:{:.4f} AP:{:.4f} FAR:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
        roc_auc, pr_auc, far, time_elapsed // 60, time_elapsed % 60))
    logger.info(f"Prediction time for is : {time_elapsed:.4f} seconds")

    return time_elapsed, time_end
