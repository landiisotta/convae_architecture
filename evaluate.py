"""
Evaluate representation model
"""
import torch
import numpy as np


def evaluate(model, loss_fn, data_iter_ts, metrics, best_eval=False):
    model.eval()
    summ = []
    encoded_list = []
    encoded_list_avg = []
    mrn_list = []

    with torch.no_grad():
        for idx, (list_mrn, list_batch) in enumerate(data_iter_ts):
            for batch, mrn in zip(list_batch, list_mrn):
                batch = batch.cuda()
                out, encoded = model(batch)
                loss = loss_fn(out, batch)
                out.cpu()
                encoded.cpu()
                summary_batch = {metric: metrics[metric](out, batch).item()
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)
                if best_eval:
                    encoded_list_avg.append(
                        np.mean(encoded.tolist(), axis=0).tolist())
                    encoded_list.append(encoded.tolist())
                    mrn_list.append(mrn)
        metrics_mean = {metric: np.mean(
            [x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = " -- ".join("{}: {:05.3f}".format(k.capitalize(), v)
                                     for k, v in sorted(metrics_mean.items()))
        print(metrics_string)

        return mrn_list, encoded_list, encoded_list_avg, metrics_mean
