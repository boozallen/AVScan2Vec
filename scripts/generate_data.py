import os
import re
import sys
import json
import pickle
import argparse
from datetime import datetime as dt
from collections import Counter

from avscan2vec.globalvars import *
from avscan2vec.utils import read_supported_avs, tokenize_label


def get_supported_avs(scan_dir):
    """Get set of supported AVs from scan report files.
    Files must be in the .jsonl format
    Track statistics for how often AVs appear in reports

    Arguments:
    scan_dir -- Full path to file containing scan reports
    """

    av_counts = Counter()
    total_reports = 0
    for file_name in os.listdir(scan_dir):
        file_path = os.path.join(scan_dir, file_name)
        with open(file_path, "rb") as f:
            for line in f:
                try:
                    report = json.loads(line)
                except (json.decoder.JSONDecodeError, UnicodeDecodeError):
                    print("Supported AVs unable to parse: ", line)
                    sys.stdout.flush()
                    continue

                # Check that contents of scan report is valid
                if not len(report):
                    continue
                if not isinstance(report, dict):
                    continue
                if report.get("data") is None:
                    continue
                report = report["data"]
                if report.get("attributes") is None:
                    continue
                report = report["attributes"]
                if report.get("last_analysis_results") is None:
                    continue

                # Normalize AV names
                avs = [re.sub(r"\W+", "", av).lower().strip()
                       for av in report["last_analysis_results"].keys()]

                # Update AV counts with normalized AV names
                av_counts.update(avs)
                total_reports += 1

    # Support AVs which appear in at least 10% of scan reports
    supported_avs = set()
    threshold = total_reports // 10
    for av, count in av_counts.items():
        if count >= threshold:
            supported_avs.add(av)

    return supported_avs


def get_data(scan_dir, supported_avs):
    """Get token vocabulary, other information needed for training AVSCan2Vec.

    Arguments:
    scan_dir -- Full path to file containing scan reports
    supported_avs -- List of AVs that are supported
    """

    # Determine vocabulary from most common tokens
    token_counts = Counter()
    line_offsets = {}
    id_dates = []
    _id = 0
    for file_name in sorted(os.listdir(scan_dir)):
        file_path = os.path.join(scan_dir, file_name)
        line_offsets[file_path] = []
        with open(file_path, "rb") as f:
            while True:

                # Attempt to read line
                offset = f.tell()
                line = f.readline()
                if not line:
                    break

                try:
                    report = json.loads(line)
                except (json.decoder.JSONDecodeError, UnicodeDecodeError):
                    print("Vocab unable to decode: ", line)
                    continue

                # Check that contents of scan report is valid
                if not len(report):
                    continue
                if not isinstance(report, dict):
                    continue
                if report.get("data") is None:
                    continue
                report = report["data"]
                if report.get("attributes") is None:
                    continue
                report = report["attributes"]
                if report.get("last_analysis_results") is None:
                    continue
                if report.get("md5") is None:
                    continue
                if report.get("last_analysis_date") is None:
                    continue

                # Read scan information
                # Reports with no scan date or first seen have None values
                av_results = report["last_analysis_results"]
                md5 = report["md5"]
                scan_date = report["last_analysis_date"]
                scan_date = dt.fromtimestamp(scan_date).strftime("%Y-%m-%d")

                # Process labels from supported AVs
                num_detections = 0
                for av, scan_data in av_results.items():

                    # Verify that the AV is supported and detected the file
                    av = re.sub(r"\W+", "", av).lower().strip()
                    if av not in supported_avs:
                        continue
                    if scan_data.get("result") is None:
                        continue

                    # Split label on non-alphanumeric characters
                    label = scan_data["result"]
                    tokens = tokenize_label(label)

                    # Add tokens to counter
                    token_counts.update(tokens)
                    num_detections += 1

                # Only include scans with 2+ detections
                if num_detections < 2:
                    continue

                # Note line offset scan date of current report
                line_offsets[file_path].append(offset)
                id_dates.append((_id, scan_date))
                _id += 1

    # Add special tokens
    token_vocab = [PAD, UNK, EOS, ABS, BEN]
    for av in sorted(supported_avs):
        token_vocab.append("<SOS_{}>".format(av))

    # Add most popular tokens in scan reports to vocab
    token_vocab += [tok for tok, _ in token_counts.most_common()]

    # Sort id_dates chronologically
    id_dates = sorted(id_dates, key=lambda l:l[1])

    return token_vocab, line_offsets, id_dates


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir",
                        help="Path to directory containing AV scan reports")
    parser.add_argument("data_dir", help="Directory to write data to")
    parser.add_argument("--av-path", default=None,
                        help="Path of text file containing supported AVs")
    args = parser.parse_args()

    # Get set of supported AVs
    supported_avs = set()
    if args.av_path is None:
        print("[-] Determing set of supported AVs")
        supported_avs = get_supported_avs(args.scan_dir)
    else:
        print("[-] Reading supported AVs from {}".format(args.av_path))
        supported_avs = read_supported_avs(args.av_path)
    print("[-] Identified {} supported AVs".format(len(supported_avs)))

    # Parse scan reports
    data = get_data(args.scan_dir, supported_avs)
    token_vocab, line_offsets, id_dates = data

    # Create data directory if it does not exist already
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)

    # Write supported AVs to file
    av_path = os.path.join(args.data_dir, "avs.txt")
    with open(av_path, "w") as f:
        for av in supported_avs:
            f.write("{}\n".format(av))
    print("[-] Wrote supported AVs to {}".format(av_path))

    # Write token vocab to file
    vocab_path = os.path.join(args.data_dir, "vocab.txt")
    with open(vocab_path, "w") as f:
        for tok in token_vocab:
            f.write("{}\n".format(tok))
    print("[-] Wrote token vocabulary to {}".format(vocab_path))

    # Write line offsets to file
    line_path = os.path.join(args.data_dir, "line_offsets.pkl")
    with open(line_path, "wb") as f:
        pickle.dump(line_offsets, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("[-] Wrote line offsets to {}".format(line_path))

    # Wrie id dates to file
    id_dates_path = os.path.join(args.data_dir, "id_dates.pkl")
    with open(id_dates_path, "wb") as f:
        pickle.dump(id_dates, f)
    print("[-] Wrote id dates to {}".format(id_dates_path))
