# AVScan2Vec

AVScan2Vec is a sequence-to-sequence autoencoder that can embed antivirus results for a malicious file into a vector. These vectors can then be used for downstream ML tasks such as classification, clustering, and nearest-neighbor lookup. More details about AVScan2Vec are provided in our paper.

TODO


If you use AVScan2Vec in your own research, please use this citation:

TODO

## Installation

AVScan2Vec can be installed using the following commands:
```
export GIT_LFS_SKIP_SMUDGE=1
pip install git+https://github.com/boozallen/AVScan2Vec
```

This repository includes checkpoints for pre-trained and fine-tuned AVScan2vec models. Downloading these checkpoints requires [Git LFS](https://github.com/git-lfs/git-lfs).

On Debian-based systems, the Git LFS package can be installed using:

```
sudo apt-get install git-lfs
```

Once Git LFS is installed, you can clone this repository using:

```
git lfs clone https://github.com/boozallen/AVScan2Vec
```

If you want to clone this repository without Git LFS and without the checkpoint files, you can run:

```
export GIT_LFS_SKIP_SMUDGE=1
git clone https://github.com/boozallen/AVScan2Vec
```

AVScan2Vec can optionally store vectors in a Qdrant database. To use this option, install [Qdrant](https://qdrant.tech/documentation/guides/installation/).

If you wish to pre-train AVScan2Vec using multiple GPUs, install the [Apex extension](https://github.com/NVIDIA/apex#installation).


## AV Scan data

AVScan2Vec learns to embed AV scan reports into vectors. Each AV scan report contains analysis about a malware sample, including labels from different AV products, file hashes, and the date of the scan. Each report should be a JSON object with the following format.

```
{
    "scans": {
        av_product_name: {
            "result": label
        }
        av_product_name: {
            "result": label
        }
        ...
    }
    "md5": md5_hash,
    "sha1": sha1_hash,
    "sha256": sha256_hash,
    "scan_date: scan_date
}
```

AV scan reports in this format can be retrieved using Virustotal's [files/ API endpoint](https://developers.virustotal.com/reference/file-info). Reports should be stored in one or more .jsonl files, with one JSON object per line. A script (query_vt_reports.py) for querying malware from VirusTotal by file hash has been provided inside of the scripts/ directory.

```
python query_vt_reports.py /path/to/hashes_file /path/to/api_file /path/to/output_file

positional arguments:
  hash_file   Path to file with hashes to query
  api_file    Path to VirusTotal API key file
  out_file    Path to file to write scan reports

optional arguments:
  -h, --help  show this help message and exit
```


## Training AVScan2Vec

Checkpoints of the pre-trained and fine-tuned AVScan2Vec are provided in the checkpoints/ directory. If you want to train AVScan2Vec yourself, we have provided scripts for doing so inside of the scripts/ directory. AVScan2Vec can be trained from scratch or from a checkpoint.

### Parsing AV scan reports

Before pre-training AVScan2Vec, you will need to run generate_data.py, provided in the scripts/ directory. This script will parse the AV scan reports to create the token vocabulary, list of supported AV products, and other information needed for training.

```
python generate_data.py /path/to/scan_dir/ /path/to/data_dir/

positional arguments:
  scan_dir           Path to directory containing AV scan reports
  data_dir           Directory to write data to

optional arguments:
  -h, --help         show this help message and exit
  --av-path AV_PATH  Path of text file containing supported AVs
```

To use a preset list of AV products, use the --av-path argument to pass a text file with the name of one AV product per line. AV names should be in all lowercase, with non-alphanumeric characters removed (including spaces). If --av-path is not provided, the script will automatically identify a list of AV products which appear in at least 10% of the scan reports.


### Pre-training

The pre-training phase allows AVScan2Vec to learn AV label semantics. AVScan2Vec performs two self-supervised learning tasks during pre-training. First, it performs masked token prediction, in which tokens are randomly held out of each report, and AVScan2Vec uses surrounding context to predict them. Additionally, AVScan2Vec performs masked label prediction, in which an entire AV label is held out of each scan report. AVScan2Vec is prompted with the name of the AV product and must auto-regressively predict the missing label.

Use pretrain_avscan2vec.py to pre-train the model. 

```
python pretrain_avscan2vec.py /path/to/data_dir/

positional arguments:
  data_dir              Path to the data directory

optional arguments:
  -h, --help            show this help message and exit
  --temporal-split      Split dataset by date, rather than randomly
  --checkpoint-file CHECKPOINT_FILE
                        Path to the checkpoint file
  --batch-size BATCH_SIZE
                        Batch size
  --num-epochs NUM_EPOCHS
                        Number of epochs
  --devices DEVICES     Devices to use
  --num-workers NUM_WORKERS
                        Number of subprocesses per DataLoader
  -L L                  The maximum number of tokens in an AV label
  -D D                  AVScan2Vec vector dimension
  -H H                  Hidden layer dimension
  --tok-layers TOK_LAYERS
                        Number of layers in the token encoder
```

If --checkpoint-file is provided, the model will begin training from a warm state using the provided model checkpoint. By default, AVScan2Vec expects to be pretrained in parallel by distributing each batch across multiple GPUs.


### Preparing for Fine-tuning

AVScan2Vec is fine-tuned on pairs of similar malicious files. It learns that the scan reports of similar files should be embedded into nearby vectors. Similar files are identified using the [Trend Locality Sentitive Hash](https://github.com/trendmicro/tlsh). Files with a TLSH distance less than 30 are considered to be similar. The TLSH authors evaluate this distance threshold to have a false-positive rate of just 0.00181%.

Run tlsh_pairs.py to identify pairs of similar files using TLSH. It should be given a path to a directory to the malicious files and a second path to the data directory created by generate_data.py.

```
python tlsh_pairs.py /path/to/malware_dir /path/to/data_dir

positional arguments:
  malware_dir           Path to the malware directory
  data_dir              Path to the data directory

optional arguments:
  -h, --help            show this help message and exit
  --sketch-size SKETCH_SIZE
                        TLSH sketch size
  --tlsh-threshold TLSH_THRESHOLD
                        Max TLSH distance to be considered similar
  --num-procs NUM_PROCS
                        Number of processes
```

Lower values of --sketch size result in more TLSH comparisons, increasing runtime but ensuring that more related files are found. Higher values of --sketch-size allow the program to run faster but may fail to identify some related TLSH files. Larger values of --tlsh-threshold may identify more related files but also can produce more false positives.

If you use a different dataset for fine-tuning AVScan2Vec than the pre-training one, you will need to run generate_data.py again for the new dataset. Then, copy vocab.txt and avs.txt from your pre-training data directory into your fine-tuning data directory (overwriting the vocab.txt and avs.txt created by generate_data.py)


### Fine-tuning

After running tlsh_pairs.py (and generate_data.py if you have a different fine-tuning dataset), AVScan2Vec is ready to be fine-tuned with finetune_avscan2vec.py.

```
python finetune_avscan2vec.py /path/to/data_dir /path/to/pretrain_checkpoint_file.sav

positional arguments:
  data_dir              Path to the data directory
  pretrain_file         Path to the pretrain checkpoint file

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size
  --num-epochs NUM_EPOCHS
                        Number of epochs
  --checkpoint-file CHECKPOINT_FILE
                        Path to the checkpoint file
  --num-validation NUM_VALIDATION
                        Size of validation set
  --device DEVICE       Device to use
  --num-workers NUM_WORKERS
                        Number of subprocesses per DataLoader
  -L L                  The maximum number of tokens in an AV label
  -D D                  AVScan2Vec vector dimension
  -H H                  Hidden layer dimension
  --tok-layers TOK_LAYERS
                        Number of layers in the token encoder
```


## Predicting Vectors

To predict vectors with AVScan2Vec, you will first need to run generate_data.py on the dataset of scan reports you want to vectorize. Then, copy vocab.txt and avs.txt from your pre-training data directory into the new data directory (overwriting the vocab.txt and avs.txt created by generate_data.py). If you are predicting vectors using the provided pre-trained and fine-tuned AVScan2Vec checkpoints, you should instead copy the vocab.txt and avs.txt from the checkpoints/ directory into your data directory.

```
python predict_vectors.py /path/to/data_dir/ /path/to/pretrain_checkpoint_file.sav /path/to/finetune_checkpoint_file.sav

positional arguments:
  data_dir              Path to the data directory
  pretrain_file         Path to the pretrain checkpoint file
  checkpoint_file       Path to the finetune checkpoint file

optional arguments:
  -h, --help            show this help message and exit
  --qdrant-path QDRANT_PATH
                        Path to Qdrant database
  --qdrant-collection QDRANT_COLLECTION
                        Name of Qdrant collection
  --vec-file VEC_FILE   Path to write predicted vectors to
  --hash-file HASH_FILE
                        Path to write file hashes for each vector to
  --device DEVICE       Device to use
  --num-workers NUM_WORKERS
                        Number of subprocesses for DataLoader
  --batch-size BATCH_SIZE
                        Batch size
  -L L                  The maximum number of tokens in an AV label
  -D D                  AVScan2Vec vector dimension
  -H H                  Hidden layer dimension
  --tok-layers TOK_LAYERS
                        Number of layers in the token encoder
```

predict_vectors.py has two options for vector output: A Qdrant database or raw vectors. For Qdrant, use the --qdrant-path flag, with a path to the directory where the Qdrant database's files should be stored. You may also optionally use the --qdrant-collection flag to choose the name of the collection to store the vectors in.

To output raw vectors, use the --vec-file flag, with the path where the vectors should be written to. The --hash-file flag may also optionally be used to write the MD5, SHA-1, and SHA-256 file hashes corresponding to each vector to a file.
