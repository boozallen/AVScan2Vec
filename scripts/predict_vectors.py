import os
import torch
import argparse
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from avscan2vec.globalvars import *
from avscan2vec.utils import collate_fn
from avscan2vec.dataset import AVScanDataset
from avscan2vec import PositionalEmbedding, PretrainEncoder, PretrainLoss
from avscan2vec import FinetuneEncoder, FinetuneLoss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("pretrain_file",
                        help="Path to the pretrain checkpoint file")
    parser.add_argument("checkpoint_file",
                        help="Path to the finetune checkpoint file")
    parser.add_argument("--qdrant-path", default=None,
                        help="Path to Qdrant database")
    parser.add_argument("--qdrant-collection", default="AVScan2Vec",
                        help="Name of Qdrant collection")
    parser.add_argument("--vec-file", default=None,
                        help="Path to write predicted vectors to")
    parser.add_argument("--hash-file", default="hashes.txt",
                        help="Path to write file hashes for each vector to")
    parser.add_argument("--device", default="cuda",
                        help="Device to use")
    parser.add_argument("--num-workers", default=16, type=int,
                        help="Number of subprocesses for DataLoader")
    parser.add_argument("--batch-size", default=100, type=int,
                        help="Batch size")
    parser.add_argument("-L", default=7, type=int,
                        help="The maximum number of tokens in an AV label")
    parser.add_argument("-D", default=768, type=int,
                        help="AVScan2Vec vector dimension")
    parser.add_argument("-H", default=768, type=int,
                        help="Hidden layer dimension")
    parser.add_argument("--tok-layers", default=4, type=int,
                        help="Number of layers in the token encoder")
    args = parser.parse_args()

    # Must provide either --qdrant-path or --vec-file
    if args.qdrant_path is None and args.vec_file is None:
        raise ValueError("Must provide either --qdrant-path or --vec-file")

    # Initialize dataset and dataloader
    dataset = AVScanDataset(args.data_dir, args.L)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, pin_memory=False, num_workers=16,
                             collate_fn=collate_fn)

    # Define pre-trained model
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

    # Define fine-tuned model
    finetune_encoder = FinetuneEncoder(pretrain_model.encoder)
    finetune_model = FinetuneLoss(finetune_encoder)

    # Load fine-tuned AVScan2vec model from checkpoint
    save_info = torch.load(args.checkpoint_file, map_location="cpu")
    finetune_model.load_state_dict(save_info["model_state_dict"])
    finetune_encoder = finetune_model.finetune_encoder.to(args.device)
    finetune_encoder = finetune_encoder.eval()

    # If using Qdrant, connect to Qdrant database on default port
    if args.qdrant_path is not None:
        client = QdrantClient(host="localhost", port=6333)
        client.recreate_collection(
            collection_name=args.qdrant_collection,
            vectors_config=VectorParams(size=args.D,
                                        distance=Distance.EUCLID),
        )

    # Otherwise, store vectors in --vec-file and store the file hashes for each
    # vector separately in --hash-file
    else:
        X = np.memmap(args.vec_file, dtype=np.float32, mode="w+",
                      shape=(len(dataset), args.D))
        f = open(args.hash_file, "w")

    # Get AVScan2Vec vector representations of scan reports
    B_total = 0
    for i, (X_scan, X_av, md5s, sha1s, sha256s, _) in enumerate(data_loader):
        B = X_scan.shape[0]
        X_scan = X_scan.to(args.device, non_blocking=True)
        X_av = X_av.to(args.device, non_blocking=True)
        with torch.no_grad():
            X_vec = finetune_encoder(X_scan, X_av).cpu().numpy()

        # Write vectors to Qdrant db
        if args.qdrant_path is not None:
            points = [
                PointStruct(
                    id=B_total+j,
                    vector=X_vec[j].tolist(),
                    payload = {
                        "md5": md5s[j],
                        "sha1": sha1s[j],
                        "sha256": sha256s[j]
                    }
                )
                for j in range(B)
            ]
            client.upsert(collection_name=args.qdrant_collection,
                          points=points)

        # Write vectors and hashes to file
        else:
            for j in range(B):
                X[B_total+j] = X_vec[j]
                f.write("{},{},{}\n".format(md5s[j], sha1s[j], sha256s[j]))

        # Update total number of points inserted
        B_total += B

    # Close files
    if not args.qdrant_path is not None:
        f.close()
    print("[+] Predicted {} vectors".format(B_total))
