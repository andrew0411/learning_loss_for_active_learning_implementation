'''
Loss prediction module's learning loss

    - LPL (loss prediction loss) : proposed from the paper

    - MRR (mean reciprocal rank)

    - KTL (Kendall tau loss)
'''

import torch

def LPL(predicted, y, margin=1):
    B = y.shape[0]

    loss = 0

    for i in range(0, n, 2):
        pair_loss = y[i] - y[i+1]
        predicted_pair_loss = predicted[i] - predicted[i+1]

        if pair_loss > 0:
            loss += max(0, margin - predicted_pair_loss)
        else:
            loss += max(0, margin + predicted_pair_loss)

    return loss / (B / 2)

def MRR_loss(predicted, y):
    B = y.shape[0]
    MRR = 0
    for i in range(n):
        true_rank = (y[:i] > y[i]).sum() + 1
        pred_rank = (predicted[:i] > predicted[i]).sum() + 1
        MRR += 1 / pred_rank
    return MRR / B


def KTL(predicted, y):
    B = y.shape[0]
    C = torch.sum(torch.eq(y, predicted)) - B
    return 1 - (2 * C) / (B * (B - 1))
