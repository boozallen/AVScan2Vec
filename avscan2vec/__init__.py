import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AdaptiveLogSoftmaxWithLoss

from .globalvars import *
from .char_embd import CharCNN


class PositionalEmbedding(nn.Module):

    def __init__(self, A, L, D, n_chars, max_chars, PAD_idx):
        """CharacterBERT-style embeddings

        Arguments:
        A -- The number of AV products
        L -- Number of tokens per label
        D -- Embedding dimension
        n_chars -- Size of character dataset
        max_chars -- Max number of characters in token
        PAD_idx -- Index of <PAD> in token vocabulary
        """

        super(PositionalEmbedding, self).__init__()
        self.token_embd = CharCNN(D, n_chars, max_chars, PAD_idx)
        self.av_embd = nn.Embedding(A+1, D)
        self.pos_embd = nn.Embedding(A*L+1, D)
        self.layer_norm = nn.LayerNorm(D)

        positions = torch.arange(A*L+1, dtype=torch.long).reshape(1, -1) # (1, A*L+1)
        avs = torch.arange(A+1, dtype=torch.long) # (A+1)
        avs = avs.repeat(L)[L-1:].reshape(1, -1) # (1, A*L+1)
        self.register_buffer("positions", positions)
        self.register_buffer("avs", avs)


    def forward(self, X_scan):

        # Get batch size
        B = X_scan.shape[0]

        # Repeat positions and avs B times
        pos = self.positions.repeat(B, 1)
        avs = self.avs.repeat(B, 1)

        # Embed token

        token_embd = self.token_embd(X_scan) + self.av_embd(avs) + self.pos_embd(pos)
        token_embd = self.layer_norm(token_embd)
        return token_embd


class PretrainEncoder(nn.Module):

    def __init__(self, A, L, D, H, tok_layers, PAD_idx, token_embd):
        """Implements AVScan2Vec's forward pass during pre-training

        Arguments:
        A -- The number of AV products
        L -- Number of tokens per label
        D -- Embedding dimension
        H -- Hidden layer dimension
        tok_layers -- Number of layers in the token encoder
        PAD_idx -- Index of <PAD> in token vocabulary
        token_embd -- PositionalEmbedding object
        """

        super(PretrainEncoder, self).__init__()
        self.A = A
        self.L = L
        self.D = D
        self.H = H
        self.tok_layers = tok_layers
        self.PAD_idx = PAD_idx
        self.token_embd = token_embd

        # Define token transformer encoder
        encoderlayer_tok_args = {
            "d_model": D,
            "nhead": 8,
            "dim_feedforward": H,
            "batch_first": True
        }
        encoderlayer_tok = nn.TransformerEncoderLayer(**encoderlayer_tok_args)
        self.encoder_tok = nn.TransformerEncoder(encoderlayer_tok, num_layers=tok_layers)


    def forward(self, X_scan, X_av):
        """Forward pass through AVScan2Vec pretrain model.

        Arguments:
        X_scan -- Batch of scan reports (B, A*L+1, max_chars)
        X_av -- AVs with labels in batch (B, A)
        """

        # Get mask indicating <PAD> tokens in X_scan
        with torch.no_grad():
            token_mask = (X_scan[:, :, 1] == self.PAD_idx) # (B, A*L+1)

        # Apply positional and segment embeddings
        X_scan_embd = self.token_embd(X_scan) # (B, A*L+1, D)

        # Encode X_scan using token encoder
        X_tok_enc = self.encoder_tok(X_scan_embd, src_key_padding_mask=token_mask) # (B, A*L+1, D)
        return X_tok_enc


class PretrainLoss(nn.Module):
    def __init__(self, A, L, D, H, tok_layers, encoder, dataset):
        """Compute loss for Masked Token Prediction and Masked Label Prediction.

        Arguments:
        A -- The number of AV products
        L -- Number of tokens per label
        D -- Embedding dimension
        H -- Hidden layer dimension
        tok_layers -- Number of layers in the token encoder
        encoder -- AVScan2Vec PretrainEncoder object
        dataset -- AVScanDataset object
        """

        super(PretrainLoss, self).__init__()
        self.A = A
        self.L = L
        self.D = D
        self.H = H
        self.tok_layers = tok_layers
        self.encoder = encoder
        self.dataset = dataset
        self.vocab_size = dataset.vocab_size

        # Define network for aggregating outputs of transformer encoder for LSTM hidden state
        self.h_init = nn.Sequential(
            nn.Linear(D, D*tok_layers//2),
            nn.LeakyReLU(),
            nn.LayerNorm(D*tok_layers//2),
            nn.Linear(D*tok_layers//2, D*tok_layers),
            nn.LeakyReLU(),
            nn.LayerNorm(D*tok_layers),
        )

        # Initial cell state of decoder
        _c_init = torch.zeros(tok_layers, 1, D)
        self.register_buffer("c_init", _c_init) # (tok_layers, 1, D)

        # <SOS> token to use as initial decoder input
        _X_SOS = torch.cat([self.dataset.tok_to_tensor(tok) for tok in dataset.SOS_toks]).reshape(A, -1) # (A, max_chars)        
        self.register_buffer("X_SOS", _X_SOS) # (A, max_chars)

        # Define the LSTM decoder
        decoder_args = {
            "input_size": D,
            "hidden_size": D,
            "num_layers": tok_layers,
            "batch_first": True
        }
        self.decoder = nn.LSTM(**decoder_args)

        # Define token prediction network
        self.predict_token = nn.Sequential(
            nn.Linear(D, H),
            nn.LeakyReLU(),
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.LeakyReLU(),
            nn.LayerNorm(H),
        )

        # Define label prediction network
        self.predict_label = nn.Sequential(
            nn.Linear(D, H),
            nn.LeakyReLU(),
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.LeakyReLU(),
            nn.LayerNorm(H),
        )

        # Adaptive softmax with loss
        self.alswl = AdaptiveLogSoftmaxWithLoss(H, self.vocab_size, cutoffs=[750, 5000, 20000])


    def forward(self, X_scan, X_av, Y_scan, Y_idxs, Y_label, Y_av):
        """Compute masked token prediction and masked label prection loss.
        If not being trained, also returns predicted tokens and labels

        Arguments:
        X_scan -- Batch of scan reports (B, A*L+1)
        X_av -- AVs with labels in batch (B, A)
        Y_scan -- Randomly selected tokens to predict (?)
        Y_idxs -- Indices of tokens in to predict (B, A)
        Y_label -- Batch of randomly selected labels to predict (B, L)
        Y_av -- Batch of AVs which produced the labels to predict (B)
        """

        # Get batch size
        B = X_scan.shape[0]

        # Encode X_scan
        X_tok_enc = self.encoder(X_scan, X_av)

        # Get encoding of CLS token
        X_vec = X_tok_enc[:, 0, :] # (B, D)

        # Reshape X_tok_enc to (B*A, L, D) (Drop CLS token)
        X_tok_sel = X_tok_enc[:, 1:, :]
        X_tok_sel = X_tok_sel.reshape(B*self.A, self.L, self.D) # (B*A, L, D)

        # Select elements from X_scan_enc for prediction
        Y_idxs = Y_idxs.reshape(B*self.A) # (B*A)
        X_tok_pred = X_tok_sel[Y_idxs > 0] # (?, L, D)
        Y_idxs = Y_idxs[Y_idxs > 0] # (?)
        X_tok_pred = X_tok_pred[torch.arange(Y_idxs.shape[0]), Y_idxs] # (?, D)

        # Get masked token prediction loss
        pred_token_logits = self.predict_token(X_tok_pred) # (?, D)
        _, token_loss = self.alswl(pred_token_logits, Y_scan)

        # Repeat X_scan_vec n_layers times to get decoder hidden state
        h_decoder = self.h_init(X_vec) # (B, D*tok_layers)
        h_decoder = h_decoder.reshape(B, self.D, self.tok_layers)
        h_decoder = h_decoder.permute(2, 0, 1).contiguous() # (tok_layers, B, D)

        # Repeat c_init B times to get decoder cell state
        with torch.no_grad():
            c_decoder = self.c_init.repeat(1, B, 1) # (tok_layers, B, D)

        # Embed the <SOS> token as the first input to the decoder
        with torch.no_grad():
            Y_pos = Y_av * self.L
            X_SOS_tok = self.X_SOS[Y_av] # (B, max_chars)
            X_decoder = self.encoder.token_embd.token_embd(X_SOS_tok) # (B, D)
            X_decoder = X_decoder + self.encoder.token_embd.av_embd(Y_av+1)
            X_decoder = X_decoder + self.encoder.token_embd.pos_embd(Y_pos+1)
            X_decoder = self.encoder.token_embd.layer_norm(X_decoder)
            X_decoder = X_decoder.reshape(B, 1, self.D) # (B, 1, D)

        # Determine whether to use teacher forcing (50% when training)
        teacher_forcing = False
        if self.training:
            teacher_forcing = np.random.choice((True, False))

        # Iterate over each timestep
        pred_tokens = None
        pred_labels = []
        label_loss = 0.0
        for t in range(self.L):

            # Get output for decoder at current timestep
            decoder_out, (h_decoder, c_decoder) = self.decoder(X_decoder, (h_decoder, c_decoder))
            decoder_out = decoder_out.reshape(B, self.D)

            # Predict token to use as decoder's next input
            pred_label_logits = self.predict_label(decoder_out)
            with torch.no_grad():
                if teacher_forcing:
                    X_next = Y_label[:, t]
                else:
                    X_next = self.alswl.predict(pred_label_logits)
                if not self.training:
                    pred_labels.append(X_next)

            # Embed predicted token
            if t+1 < self.L:
                X_next_tok = torch.stack([self.dataset.tok_to_tensor(self.dataset.token_vocab_rev[tok.item()]) for tok in X_next])
                X_next_tok = X_next_tok.to(next(self.parameters()).device)
                X_decoder = self.encoder.token_embd.token_embd(X_next_tok).reshape(B, -1, self.D) # (B, 1, D)

            # Get timestep loss
            _, timestep_loss = self.alswl(pred_label_logits, Y_label[:, t])
            label_loss = label_loss + timestep_loss

        # Predict tokens/labels if evaluating model
        if not self.training:
            with torch.no_grad():
                pred_tokens = self.alswl.predict(pred_token_logits)
                pred_labels = torch.stack(pred_labels, dim=1)
            return token_loss, label_loss, pred_tokens, pred_labels

        return token_loss, label_loss, None, None


class FinetuneEncoder(nn.Module):

    def __init__(self, pretrain_encoder):
        """Class for fine-tuning AVScan2Vec

        Arguments:
        pretrain_encoder -- Trained AVScan2Vec PretrainEncoder object
        """

        super(FinetuneEncoder, self).__init__()
        self.pretrain_encoder = pretrain_encoder

        # Freeze token embedding, all but last two layers of token encoder
        for param in pretrain_encoder.token_embd.parameters():
            param.requires_grad = False
        for layer in pretrain_encoder.encoder_tok.layers[:2]:
            for param in layer.parameters():
                param.requires_grad = False
            layer.dropout = nn.Dropout(p=0.0, inplace=True)


    def forward(self, X_scan, X_av):
        """Perform forward pass of AVScan2Vvec fine-tune model.

        Arguments:
        X_scan -- Batch of scan reports (B, A*L+1, max_chars)
        X_av -- AVs with labels in batch (B, A)
        """

        # Encode scan report using pretrained encoder
        X_tok_enc = self.pretrain_encoder(X_scan, X_av) # (B, A*L+1, D)

        # Return batch of vectors correspoding to <CLS> token
        X_vec = X_tok_enc[:, 0, :]
        return X_vec


class FinetuneLoss(nn.Module):

    def __init__(self, finetune_encoder):
        """Computes Multiple Negatives Ranking Loss for AVScan2Vec finetune model.

        Arguments:
        finetune_encoder -- AVScan2Vec FinetuneEncoder object
        """

        super(FinetuneLoss, self).__init__()
        self.finetune_encoder = finetune_encoder


    def forward(self, X_scan_anc, X_av_anc, X_scan_pos, X_av_pos):
        """Loss based on https://github.com/UKPLab/sentence-transformers/issues/373

        Arguments
        X_scan_anc -- Batch of anchor scan reports (B, A*L+1)
        X_scan_pos -- Batch of positive scan reports (B, A*L+1)
        X_av_anc -- AVs with labels in X_scan_anc (B, A)
        X_av_pos -- AVs with labels in X_scan_pos (B, A)
        """

        X_anc = self.finetune_encoder(X_scan_anc, X_av_anc)
        X_pos = self.finetune_encoder(X_scan_pos, X_av_pos)

        # Compute MNR loss
        scores = torch.matmul(X_pos, X_anc.t())
        diagonal_mean = torch.mean(torch.diag(scores))
        mean_log_row_sum_exp = torch.mean(torch.logsumexp(scores, dim=1))
        return -diagonal_mean + mean_log_row_sum_exp
