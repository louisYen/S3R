import torch
import numpy as np
import os.path as osp

from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

def inference(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)

        gt = dataloader.dataset.ground_truths
        dataset = args.dataset.lower()

        if args.inference:
            video_list = dataloader.dataset.video_list
            result_dict = dict()

        for i, (video, macro) in enumerate(dataloader):
            video = video.to(device)
            video = video.permute(0, 2, 1, 3)

            macro = macro.to(device)

            outputs = model(video, macro)

            # >> parse outputs
            logits = outputs['video_scores']

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            if args.inference:
                video_id = video_list[i]
                result_dict[video_id] = logits.cpu().detach().numpy()

            sig = logits
            pred = torch.cat((pred, sig))

        if args.inference:
            out_dir = f'output/{dataset}'
            import pickle
            with open(osp.join(out_dir, f'{dataset}_taskaware_results.pickle'), 'wb') as fout:
                pickle.dump(result_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        score = rec_auc

        return score

