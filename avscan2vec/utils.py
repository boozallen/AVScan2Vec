import re
import math
import torch
import numpy as np
import torch.nn as nn

from .globalvars import *


def read_supported_avs(av_path):
    """Read list of supported AV products from av_path."""

    supported_avs = set()
    with open(av_path, "r") as f:
        for line in f:
            av = line.strip()
            supported_avs.add(av)
    return supported_avs


def tokenize_label(label):
    """Split an AV label into a list of tokens."""

    tokens = re.split(r"[^0-9a-zA-Z]", label)
    tokens = [tok.lower().strip() for tok in tokens]
    tokens = [tok for tok in tokens if len(tok)]
    return tokens


def accuracy_score(Y, Y_hat, ignore_idxs, device="cpu"):
    """Compute accuracy, ignoring any special tokens in ignore_idxs."""

    mask = torch.ones_like(Y, dtype=torch.bool, device=device)
    for idx in ignore_idxs:
        mask = mask & (Y != idx)
    correct_tokens = Y_hat.eq(Y).masked_select(mask).sum().item()
    total_tokens = mask.sum().item()
    if total_tokens == 0:
        return 0.0
    return correct_tokens / total_tokens, mask


def collate_fn(batch):
    """Collate function for AVScanDataset class."""

    X_scan = torch.stack([i[0] for i in batch])
    X_av = torch.stack([i[1] for i in batch])
    md5s = [i[2] for i in batch]
    sha1s = [i[3] for i in batch]
    sha256s = [i[4] for i in batch]
    scan_dates = [i[5] for i in batch]
    return (X_scan, X_av, md5s, sha1s, sha256s, scan_dates)


def pretrain_collate_fn(batch):
    """Collate function for PretrainDataset class."""

    X_scan = torch.stack([i[0] for i in batch])
    X_av = torch.stack([i[1] for i in batch])
    Y_scan = torch.as_tensor(np.concatenate([i[2] for i in batch]))
    Y_idxs = torch.stack([i[3] for i in batch])
    Y_label = torch.stack([i[4] for i in batch])
    Y_av = torch.stack([i[5] for i in batch])
    md5s = [i[6] for i in batch]
    scan_dates = [i[7] for i in batch]
    return (X_scan, X_av, Y_scan, Y_idxs, Y_label, Y_av, md5s, scan_dates)


def finetune_collate_fn(batch):
    """Collate function for FinetuneDataset class."""

    X_scan_anc = torch.stack([i[0] for i in batch])
    X_av_anc = torch.stack([i[1] for i in batch])
    X_scan_pos = torch.stack([i[2] for i in batch])
    X_av_pos = torch.stack([i[3] for i in batch])
    md5s = [i[4] for i in batch]
    scan_dates = [i[5] for i in batch]
    return (X_scan_anc, X_av_anc, X_scan_pos, X_av_pos, md5s, scan_dates)
