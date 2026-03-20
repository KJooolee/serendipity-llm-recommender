from config import *

import json
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim as optim





def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                         for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()
##
# def absolute_recall_mrr_ndcg_for_ks(scores, labels, ks):
#     metrics = {}
#     labels = F.one_hot(labels, num_classes=scores.size(1))
#     answer_count = labels.sum(1)

#     labels_float = labels.float()
#     ks = sorted(ks, reverse=True)
#     max_k = ks[0]

#     # top-k 인덱스만 추출
#     _, topk_indices = torch.topk(scores, k=max_k, dim=1)

#     # topk 후보에 대해 one-hot 기준으로 hit 계산
#     hits = labels_float.gather(1, topk_indices)

#     for k in ks:
#         hits_k = hits[:, :k]
#         metrics['Recall@%d' % k] = (
#             hits_k.sum(1) / torch.min(torch.tensor(k).to(labels.device), answer_count.float())
#         ).mean().cpu().item()

#         metrics['MRR@%d' % k] = (
#             hits_k / torch.arange(1, k+1, device=hits.device).float().unsqueeze(0)
#         ).sum(1).mean().cpu().item()

#         weights = 1 / torch.log2(torch.arange(2, k+2).float().to(hits.device))
#         dcg = (hits_k * weights).sum(1)
#         idcg = torch.tensor([
#             weights[:min(int(n), k)].sum() for n in answer_count
#         ]).to(dcg.device)
#         ndcg = (dcg / idcg).mean()
#         metrics['NDCG@%d' % k] = ndcg.cpu().item()

#     return metrics

def absolute_recall_mrr_ndcg_for_ks(scores, labels, ks):
    metrics = {}
    ##print(f"Labels before one_hot: shape={labels.shape}, dtype={labels.dtype}, min={labels.min()}, max={labels.max()}")
    labels = F.one_hot(labels, num_classes=scores.size(1))
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)

    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(
                labels.device), labels.sum(1).float())).mean().cpu().item()
        
        metrics['MRR@%d' % k] = \
            (hits / torch.arange(1, k+1).unsqueeze(0).to(
                labels.device)).sum(1).mean().cpu().item()

        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                             for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics

def recall_mrr_ndcg_topk(pred_indices, labels, ks=[1, 5, 10]):
    metrics = {}
    pred_indices = torch.tensor(pred_indices)
    labels = torch.tensor(labels)
    for k in ks:
        topk_preds = pred_indices[:, :k]
        hits = (topk_preds == labels.unsqueeze(1)).float()
        recalls = hits.sum(dim=1)
        mrrs = hits / (torch.arange(1, k+1).float().to(hits.device))  # 1/rank
        mrrs = mrrs.max(dim=1)[0]  # only first hit
        ndcgs = hits / torch.log2(torch.arange(2, k+2).float().to(hits.device))
        ndcgs = ndcgs.sum(dim=1)

        metrics[f'Recall@{k}'] = recalls.mean().item()
        metrics[f'MRR@{k}'] = mrrs.mean().item()
        metrics[f'NDCG@{k}'] = ndcgs.mean().item()
    return metrics


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
