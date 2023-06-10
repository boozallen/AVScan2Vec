import re
import sys
import time
import torch
import pickle
import random
import argparse
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from avscan2vec.globalvars import *
from avscan2vec.utils import finetune_collate_fn
from avscan2vec.dataset import FinetuneDataset
from avscan2vec import PositionalEmbedding, PretrainEncoder, PretrainLoss
from avscan2vec import FinetuneEncoder, FinetuneLoss

def train_network(model, optimizer, scheduler, train_loader, val_loader, epochs, checkpoint_file, device):

    # Put model into train mode
    model = model.train()

    # Iterate over each epoch
    for epoch in range(epochs):
        model = model.train()
        print_loss = 0.0
        batches = 0

        # Iterate over each batch
        for X_scan_anc, X_av_anc, X_scan_pos, X_av_pos, _, _ in train_loader:
            X_scan_anc = X_scan_anc.to(device)
            X_av_anc = X_av_anc.to(device)
            X_scan_pos = X_scan_pos.to(device)
            X_av_pos = X_av_pos.to(device)

            # Train model on batch
            optimizer.zero_grad()
            mnr_loss = model(X_scan_anc, X_av_anc, X_scan_pos, X_av_pos)

            # Get batch_size
            B = X_scan_anc.shape[0]

            # Backprop and update weights
            mnr_loss.backward()
            optimizer.step()

            # Update loss totals
            batches += 1
            batch_loss = mnr_loss.item() * B
            print_loss += batch_loss

            # Print training info every 100 batches
            if batches % 100 == 0:
                fmt_str = "Batches: {}  MNR Loss: {}"
                print(fmt_str.format(batches, print_loss))
                sys.stdout.flush()
                print_loss = 0.0

            # Validate model and save statistics every 2,500 batches
            if batches % 2500 == 0:

                # Put model into eval mode
                print("Validating model...")
                sys.stdout.flush()
                model = model.eval()

                # Iterate over validation batch
                val_loss = 0.0
                for X_scan_anc, X_av_anc, X_scan_pos, X_av_pos, _, _ in val_loader:
                    X_scan_anc = X_scan_anc.to(device)
                    X_av_anc = X_av_anc.to(device)
                    X_scan_pos = X_scan_pos.to(device)
                    X_av_pos = X_av_pos.to(device)

                    # Get validation loss
                    with torch.no_grad():
                        mnr_loss = model(X_scan_anc, X_av_anc, X_scan_pos, X_av_pos)
                    B = X_scan_anc.shape[0]
                    val_loss += mnr_loss.item() * B

                # Step scheduler
                scheduler.step(val_loss)

                # Put model back into train mode
                model = model.train()

                # Save partial model
                checkpoint_file_part = checkpoint_file + "part"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, checkpoint_file_part)
                print("Saved model to {}".format(checkpoint_file_part))

        # Save model at end of epoch
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, checkpoint_file)
        print("Saved model to {}".format(checkpoint_file))

    return


if __name__ == "__main__":

    # Parse commnand line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("pretrain_file",
                        help="Path to the pretrain checkpoint file")
    parser.add_argument("--batch-size", default=100, type=int,
                        help="Batch size")
    parser.add_argument("--num-epochs", default=1, type=int,
                        help="Number of epochs")
    parser.add_argument("--checkpoint-file", default="checkpoint_finetune.sav",
                        help="Path to the checkpoint file")
    parser.add_argument("--num-validation", default=10000, type=int,
                        help="Size of validation set")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--num-workers", default=16, type=int,
                        help="Number of subprocesses per DataLoader")
    parser.add_argument("-L", default=7, type=int,
                        help="The maximum number of tokens in an AV label")
    parser.add_argument("-D", default=768, type=int,
                        help="AVScan2Vec vector dimension")
    parser.add_argument("-H", default=768, type=int,
                        help="Hidden layer dimension")
    parser.add_argument("--tok-layers", default=4, type=int,
                        help="Number of layers in the token encoder")
    args = parser.parse_args()
    print("Fine-tuning AVScan2Vec with args: {}".format(args))
    sys.stdout.flush()

    # Initialize dataset
    dataset = FinetuneDataset(args.data_dir, args.L)

    # Get sizes of train / validation datasets
    ids = list(dataset.similar_idxs.keys())
    n_train = len(ids) - args.num_validation
    n_train = n_train // args.batch_size * args.batch_size
    random.seed(42)
    random.shuffle(ids)

    # Get Subsets for train, validation, and test dataset
    train_dataset = Subset(dataset, ids[:n_train])
    val_dataset = Subset(dataset, ids[n_train:n_train+args.num_validation])
    print("Size of training set: {}".format(n_train))
    print("Size of validation set: {}".format(args.num_validation))
    sys.stdout.flush()

    # Get train, validation, and test loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True,
                              num_workers=args.num_workers,
                              collate_fn=finetune_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, pin_memory=True,
                            num_workers=args.num_workers,
                            collate_fn=finetune_collate_fn)

    # Define pre-train model
    A = dataset.num_avs
    n_chars = len(dataset.alphabet)
    max_chars = dataset.max_chars
    PAD_idx = dataset.alphabet_rev[PAD]
    token_embd = PositionalEmbedding(A, args.L, args.D, n_chars, max_chars,
                                     PAD_idx)
    encoder = PretrainEncoder(A, args.L, args.D, args.H, args.tok_layers,
                              PAD_idx, token_embd)
    pretrain_model = PretrainLoss(A, args.L, args.D, args.H, args.tok_layers,
                                  encoder, dataset)

    # Load pre-trained AVScan2Vec model from checkpoint
    save_info = torch.load(args.pretrain_file, map_location="cpu")
    state_dict = OrderedDict()
    for k, v in save_info["model_state_dict"].items():
        new_k = re.sub(r"module.", "", k)
        state_dict[new_k] = v
    pretrain_model.load_state_dict(state_dict)

    # Define fine-tune model
    finetune_encoder = FinetuneEncoder(pretrain_model.encoder)
    finetune_model = FinetuneLoss(finetune_encoder)
    finetune_model = finetune_model.to(args.device)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(finetune_model.parameters(), lr=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5,
                                  threshold=0.001, verbose=True)

    # Train network
    train_args = {
        "model": finetune_model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "epochs": args.num_epochs,
        "checkpoint_file": args.checkpoint_file,
        "device": args.device
    }
    train_network(**train_args)
