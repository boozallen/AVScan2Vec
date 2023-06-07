# AVScan2Vec

AVScan2Vec is a sequence-to-sequence autoencoder that can embed antivirus results for a malicious file into a vector. These vectors can then be used for downstream ML tasks such as classification, clustering, and nearest-neighbor lookup. More details about AVScan2Vec are provided in our paper.

TODO


If you use AVScan2Vec in your own research, please use this citation:

TODO


## Installation

AVScan2Vec can be installed using the following command:
```
pip install pip@git+https://github.com/boozallen/AVScan2Vec
```

PyTorch's native implementation of DistributedDataParallel is not compatible with AdaptiveLogSoftmaxWithLoss. If you wish to pre-train AVScan2Vec using multiple GPUs, install the Apex extension:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```


## Training Data

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

AV scan reports in this format can be retrieved using the [Virustotal files/ API endpoint](https://developers.virustotal.com/reference/file-info). Reports should be stored in one or more .jsonl files, with one JSON object per line. A script (query_vt_reports.py) for querying malware from VirusTotal by file hash has been provided inside of scripts/.

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

scripts/ contains Python programs which can be used to train AVScan2Vec.

### Generating training data

First, run generate_data.py to parse the AV scan reports and create the token vocabulary, list of supported AV products, and other information needed for training.

```
python generate_data.py /path/to/scan_dir/ /path/to/data_dir/

positional arguments:
  scan_dir           Path to directory containing AV scan reports
  data_dir           Directory to write data to

optional arguments:
  -h, --help         show this help message and exit
  --av-path AV_PATH  Path of text file containing supported AVs
```

To use a preset list of AV products, use the --av-path argument to pass a text file with the name of one AV product per line. If --av-path is not provided, the script will automatically identify a list of AV products which appear in at least 1% of the scan reports.


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
  -L L                  The maximum number of tokens in an AV label
  -D D                  AVScan2Vec vector dimension
  -H H                  Hidden layer dimension
  --tok-layers TOK_LAYERS
                        Number of layers in the token encoder
```

If --checkpoint-file is provided, the model will begin training from a warm state using the provided model checkpoint. By default, AVScan2Vec expects to be pretrained in parallel by distributing each batch across multiple GPUs.


### Identifying related files

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


### Fine-tuning AVScan2Vec

After running tlsh_pairs.py, AVScan2Vec is ready to be fine-tuned with finetune_avscan2vec.py.


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
  -L L                  The maximum number of tokens in an AV label
  -D D                  AVScan2Vec vector dimension
  -H H                  Hidden layer dimension
  --tok-layers TOK_LAYERS
                        Number of layers in the token encoder
```


## Predicting Vectors with AVScan2Vec

TODO
