
import time
from utils import fixed_smooth, slide_smooth
from test import *


def infer_func(model, dataloader, gt, logger, cfg):
    st = time.time()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0) #.cuda()
        normal_preds = torch.zeros(0) #.cuda()
        normal_labels = torch.zeros(0) #.cuda()
        gt_tmp = torch.tensor(gt.copy()) #.cuda()

        for i, (v_input, name) in enumerate(dataloader):

            time_start = time.time()

            # name = name[0].split('/')[1].split('_x264')[0]
            name = name[0].split('_x264')[0].split('/')[-1]
            # breakpoint()
            # name = name[0].split('_i3d')[0].split('/')[-1]

            print(f"Lote {i}, {name}: {v_input.shape}")
            v_input = v_input.float() #.cuda(non_blocking=True)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            logits, _ = model(v_input, seq_len)
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
            pred_save = list(logits.cpu().detach().numpy())
            # pred_save = np.repeat(pred_save, 15)
            rate = 1
            pred_save = np.repeat(pred_save, 16*rate)
            # np.save(f"predictions/test/{name}_pred.npy", pred_save)
            print(f'file {name}_pred.npy saved')
            pred_save = torch.zeros(0) #.cuda()
            print(f"Time elapsed: {time.time() - time_start}")
            


            labels = gt_tmp[: seq_len[0]*16*rate]
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            gt_tmp = gt_tmp[seq_len[0]*16*rate:]

            

        pred = list(pred.cpu().detach().numpy())
        # save predictions in .npy file
        far = cal_false_alarm(normal_labels, normal_preds)
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16*rate))
        roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16*rate))
        pr_auc = auc(rec, pre)

    time_elapsed = time.time() - st
    logger.info('offline AUC:{:.4f} AP:{:.4f} FAR:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
        roc_auc, pr_auc, far, time_elapsed // 60, time_elapsed % 60))