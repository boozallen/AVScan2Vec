import os
import copy
import json
import mmap
import torch
import pickle
import random
import numpy as np
from datetime import datetime as dt
from torch.utils.data import Dataset

from .globalvars import *
from .utils import tokenize_label, read_supported_avs


class AVScanDataset(Dataset):

    def __init__(self, data_dir, max_tokens=7, max_chars=20, max_vocab=10000000):
        """Base dataset class for AVScan2Vec.

        Arguments:
        data_dir -- Path to dataset directory
        max_tokens -- Maximum number of tokens per label
        max_chars -- Maximum number of chars per token
        max_vocab -- Maximum number of tokens to track (for masked token / label prediction)
        """

        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.max_vocab = max_vocab

        # Read supported AVs
        av_path = os.path.join(data_dir, "avs.txt")
        self.supported_avs = read_supported_avs(av_path)
        self.avs = sorted(list(self.supported_avs))
        self.av_vocab_rev = [NO_AV] + self.avs
        self.num_avs = len(self.avs)

        # Map each AV to a unique index
        self.av_vocab = {av: idx for idx, av in enumerate(self.av_vocab_rev)}

        # Construct character alphabet
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        self.alphabet = [char for char in self.alphabet]
        self.SOS_toks = ["<SOS_{}>".format(av) for av in self.avs]
        self.special_tokens = [PAD, CLS, EOS, ABS, BEN] + self.SOS_toks + [SOW, EOW, MASK, UNK]
        self.special_tokens_set = set(self.special_tokens)
        self.alphabet = self.special_tokens + self.alphabet
        self.alphabet_rev = {char: i for i, char in enumerate(self.alphabet)}

        # Load token vocabulary
        vocab_path = os.path.join(data_dir, "vocab.txt")
        self.token_vocab_rev = []
        with open(vocab_path, "r") as f:
            for line in f:
                if len(self.token_vocab_rev) >= self.max_vocab:
                    break
                tok = line.strip()
                self.token_vocab_rev.append(tok)

        # Map each token to a unique index
        self.token_vocab = {tok: idx for idx, tok in enumerate(self.token_vocab_rev)}

        # Zipf distribution for sampling tokens
        self.vocab_size = len(self.token_vocab_rev)
        self.zipf_vals = np.arange(5, self.vocab_size)
        self.zipf_p = 1.0 / np.power(self.zipf_vals, 2.0)
        self.zipf_p /= np.sum(self.zipf_p)

        # Load line offsets
        line_path = os.path.join(data_dir, "line_offsets.pkl")
        with open(line_path, "rb") as f:
            self.line_offsets = pickle.load(f)
        self.line_paths = sorted(list(self.line_offsets.keys()))

        # Get total number of scan reports
        self.num_reports = sum([len(v) for v in self.line_offsets.values()])


    def parse_scan_report(self, idx):

        # Find the file path that contains the target scan report
        line_path = None
        for file_path in self.line_paths:
            if idx - len(self.line_offsets[file_path]) < 0:
                line_path = file_path
                break
            idx -= len(self.line_offsets[file_path])

        # Seek to first byte of that scan report in file path
        start_byte = self.line_offsets[line_path][idx]

        # Read report from file
        with open(file_path, "r") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as f_mmap:
                f_mmap.seek(start_byte)
                line = f_mmap.readline()
                report = json.loads(line)["data"]["attributes"]
        md5 = report["md5"]
        sha1 = report["sha1"]
        sha256 = report["sha256"]
        scan_date = report["last_analysis_date"]
        scan_date = dt.fromtimestamp(scan_date).strftime("%Y-%m-%d")

        # Parse AVs and tokens from scan report
        av_tokens = {}
        
        for av in report["last_analysis_results"].keys():

            # Normalize name of AV
            scan_info = report["last_analysis_results"][av]
            av = AV_NORM.sub("", av).lower().strip()

            # Skip AVs that aren't supported
            if av not in self.supported_avs:
                continue

            # Use <BEN> special token for AVs that detected file as benign
            if scan_info.get("result") is None:
                tokens = [BEN]
            else:
                label = scan_info["result"]
                tokens = tokenize_label(label)[:self.max_tokens-2]
            av_tokens[av] = tokens

        return av_tokens, md5, sha1, sha256, scan_date


    def tok_to_tensor(self, tok):
        """Return a tensor representing each char in a token"""
        if tok in self.special_tokens_set:
            tok = [SOW, tok, EOW]
        else:
            tok = tok[:self.max_chars-2]
            tok = [SOW] + [char for char in tok] + [EOW]
        tok += [PAD]*(self.max_chars-len(tok))
        X_tok = torch.LongTensor([self.alphabet_rev[char] for char in tok])
        return X_tok


    def __getitem__(self, idx):

        # Parse scan report
        av_tokens, md5, sha1, sha256, scan_date = self.parse_scan_report(idx)

        # Construct X_scan from scan report
        X_scan = []
        for av in self.avs:
            if av_tokens.get(av) is None:
                Xi = ["<SOS_{}>".format(av), ABS, EOS]
            else:
                Xi = ["<SOS_{}>".format(av)] + av_tokens[av] + [EOS]
            Xi += [PAD]*(self.max_tokens-len(Xi))
            X_scan += Xi

        # Convert X_scan to tensor of characters
        X_scan_char = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
        for tok in X_scan:
            X_scan_char = np.concatenate((X_scan_char, self.tok_to_tensor(tok).reshape(1, -1)))

        # Construct X_av from list of AVs in report
        X_scan = torch.as_tensor(X_scan_char)
        X_av = [av if av_tokens.get(av) is not None else NO_AV for av in self.avs]
        X_av = torch.LongTensor([self.av_vocab[av] for av in X_av])
        return X_scan, X_av, md5, sha1, sha256, scan_date


    def __len__(self):
        return self.num_reports


class PretrainDataset(AVScanDataset):

    def __init__(self, data_dir, max_tokens):
        """AVScan2Vec dataset class for pre-training."""
        super().__init__(data_dir, max_tokens=max_tokens)


    def __getitem__(self, idx):

        # Parse scan report
        av_tokens, md5, _, _, scan_date = self.parse_scan_report(idx)

        # Randomly select one AV to hold out (train only)
        # Construct Y_label from held-out AV's label
        Y_label = []
        Y_av = random.choice(list(av_tokens.keys()))
        Y_label = ["<SOS_{}>".format(Y_av)] + av_tokens[Y_av] + [EOS]
        Y_label += [PAD]*(self.max_tokens-len(Y_label))
        av_tokens[Y_av] = None

        # Randomly select 5% of tokens to be replaced with MASK
        Y_idxs = [0] * self.num_avs
        rand_nums = np.random.randint(0, 100, size=self.num_avs)
        pred_tokens = set()
        for i, (av, tokens) in enumerate(av_tokens.items()):
            if tokens is None:
                continue
            if rand_nums[i] < 5:
                token_idxs = [i+1 for i, tok in enumerate(tokens) if not
                              tok.startswith("<") and not tok.endswith(">")]
                if not len(token_idxs):
                    continue
                Y_idx = random.choice(token_idxs)
                Y_idxs[self.av_vocab[av]-1] = Y_idx
                pred_tokens.add(tokens[Y_idx-1])

        # Construct X_scan from scan report
        X_scan = []
        for av in self.avs:
            if av_tokens.get(av) is None:
                Xi = ["<SOS_{}>".format(av), ABS, EOS]
            else:
                Xi = ["<SOS_{}>".format(av)] + av_tokens[av] + [EOS]
            Xi += [PAD]*(self.max_tokens-len(Xi))
            X_scan += Xi

        # Construct X_av from list of AVs in report
        X_av = [av if av_tokens.get(av) is not None else NO_AV for av in self.avs]

        # Construct Y_scan from 5% of held-out tokens
        Y_scan = []
        for i, av in enumerate(self.avs):
            Y_idx = Y_idxs[i]
            if Y_idx > 0:
                Y_scan.append(X_scan[i*self.max_tokens+Y_idx])

        # MASK any tokens in pred_tokens 80% of the time
        # 10% of the time, replace with a random token
        # 10% of the time, leave the token alone
        rand_nums = np.random.randint(0, 100, size=self.num_avs*self.max_tokens)
        for i, tok in enumerate(X_scan):
            if tok in pred_tokens:
                if rand_nums[i] < 80:
                    X_scan[i] = MASK
                elif rand_nums[i] < 90:
                    X_scan[i] = self.token_vocab_rev[random.randint(5, self.vocab_size-1)]

        # Convert X_scan to tensor of characters
        X_scan_char = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
        for tok in X_scan:
            X_scan_char = np.concatenate((X_scan_char, self.tok_to_tensor(tok).reshape(1, -1)))

        # Convert to LongTensor
        X_scan = torch.as_tensor(X_scan_char)
        X_av = torch.LongTensor([self.av_vocab[av] for av in X_av])
        Y_scan = torch.LongTensor([self.token_vocab[tok] if self.token_vocab.get(tok) is not None else self.token_vocab[UNK] for tok in Y_scan])
        Y_idxs = torch.LongTensor(Y_idxs)
        Y_label = torch.LongTensor([self.token_vocab[tok] if self.token_vocab.get(tok) is not None else self.token_vocab[UNK] for tok in Y_label])
        Y_av = torch.LongTensor([self.av_vocab[Y_av]-1])

        return X_scan, X_av, Y_scan, Y_idxs, Y_label, Y_av, md5, scan_date


class FinetuneDataset(AVScanDataset):

    def __init__(self, data_dir, max_tokens):
        """AVScan2Vec dataset class for fine-tuning."""
        super().__init__(data_dir, max_tokens=max_tokens)

        # Load idxs of similar files
        similar_idx_path = os.path.join(data_dir, "similar_ids.pkl")
        with open(similar_idx_path, "rb") as f:
            similar_idxs = pickle.load(f)
        self.similar_idxs = {idx1: idx2 for idx1, idx2 in similar_idxs}
        self.num_reports = len(self.similar_idxs.keys())

    def __getitem__(self, idx):

        # Parse scan reports
        av_tokens_anc, md5, _, _, scan_date = self.parse_scan_report(idx)
        idx_pos = self.similar_idxs[idx]
        av_tokens_pos, md5_pos, _, _, _ = self.parse_scan_report(idx_pos)

        # Construct X_scan_anc and X_scan_pos
        X_scan_anc = []
        X_scan_pos = []
        for av in self.avs:
            if av_tokens_anc.get(av) is None:
                Xi_anc = ["<SOS_{}>".format(av), ABS, EOS]
            else:
                Xi_anc = ["<SOS_{}>".format(av)] + av_tokens_anc[av] + [EOS]
            if av_tokens_pos.get(av) is None:
                Xi_pos = ["<SOS_{}>".format(av), ABS, EOS]
            else:
                Xi_pos = ["<SOS_{}>".format(av)] + av_tokens_pos[av] + [EOS]
            Xi_anc += [PAD]*(self.max_tokens-len(Xi_anc))
            X_scan_anc += Xi_anc
            Xi_pos += [PAD]*(self.max_tokens-len(Xi_pos))
            X_scan_pos += Xi_pos

        # Convert X_scan_anc and X_scan_pos to tensors of characters
        X_scan_char_anc = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
        X_scan_char_pos = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
        for tok in X_scan_anc:
            X_scan_char_anc = np.concatenate((X_scan_char_anc, self.tok_to_tensor(tok).reshape(1, -1)))
        for tok in X_scan_pos:
            X_scan_char_pos = np.concatenate((X_scan_char_pos, self.tok_to_tensor(tok).reshape(1, -1)))

        # Construct X_av_anc, X_av_pos from lists of AVs in reports
        X_av_anc = [av if av_tokens_anc.get(av) is not None else NO_AV for av in self.avs]
        X_av_pos = [av if av_tokens_pos.get(av) is not None else NO_AV for av in self.avs]

        X_scan_anc = torch.as_tensor(X_scan_char_anc)
        X_av_anc = torch.LongTensor([self.av_vocab[av] for av in X_av_anc])
        X_scan_pos = torch.as_tensor(X_scan_char_pos)
        X_av_pos = torch.LongTensor([self.av_vocab[av] for av in X_av_pos])
        return X_scan_anc, X_av_anc, X_scan_pos, X_av_pos, md5, md5_pos, scan_date


    def __len__(self):
        return self.num_reports
