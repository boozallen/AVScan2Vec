import os
import sys
import json
import tlsh
import pickle
import random
import argparse
import functools
import multiprocessing
import numpy as np


def get_malware_paths(malware_dir, md5s):
    """Helper function that returns a list of file paths for malware with
    valid AV scan reports. This script assumes that malicious files are named
    by their MD5 hash.

    Arguments:
    malware_dir -- The root directory storing all malicious files.
    md5s -- A set of md5s which have valid AV scan reports.
    """

    malware_paths = []
    for root, _, file_names in os.walk(malware_dir):
        for md5 in file_names:
            if md5 not in md5s:
                continue
            malware_paths.append(os.path.join(root, md5))

    sys.stdout.flush()
    return malware_paths


def get_tlsh(file_path):
    """Helper function that returns the TLSH digest of a file path."""
    return tlsh.hash(open(file_path, "rb").read())


def get_sketches(md5_tlsh, sketch_size):
    """Helper function that returns the sketches for a TLSH digest.

    Arguments:
    md5_tlsh: Tuple of the form (md5, tlsh)
    """

    md5, tlsh_digest = md5_tlsh
    sketches = []
    for i in range(0, len(tlsh_digest) - sketch_size):
        sketches.append("{}_{}".format(i, tlsh_digest[i:i+sketch_size]))
    return sketches


if __name__ == "__main__":

    # Parse commnand line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("malware_dir", help="Path to the malware directory")
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("--sketch-size", default=6, type=int,
                        help="TLSH sketch size")
    parser.add_argument("--tlsh-threshold", default=30, type=int,
                        help="Max TLSH distance to be considered similar")
    parser.add_argument("--num-procs", default=None, type=int,
                        help="Number of processes")
    args = parser.parse_args()

    # Load line offsets
    with open(os.path.join(args.data_dir, "line_offsets.pkl"), "rb") as f:
        line_offsets = pickle.load(f)
    line_paths = sorted(list(line_offsets.keys()))

    # Get md5 hash for each malware sample from its report
    _id = 0
    md5s = []
    md5_ids = {}
    for line_path in line_paths:
        offsets = set(line_offsets[line_path])
        with open(line_path, "r") as f:
            while True:

                # Attempt to read line
                offset = f.tell()
                line = f.readline()
                if not line:
                    break

                # Check if this is the offset for a valid scan report
                if offset not in offsets:
                    continue

                # Parse md5 hash from report
                report = json.loads(line)["data"]["attributes"]
                md5 = report["md5"]
                md5s.append(md5)
                md5_ids[md5] = _id
                _id += 1

    # Compute TLSH hashes of malware in parallel
    print("[-] Computing TLSH hashes")
    sys.stdout.flush()
    with multiprocessing.Pool(args.num_procs) as pool:
        malware_paths = get_malware_paths(args.malware_dir, md5s)
        sys.stdout.flush()
        malware_md5s = [os.path.basename(file_path)
                        for file_path in malware_paths]
        tlshs = pool.map(get_tlsh, malware_paths)
        md5_tlshs = {md5: tlsh for md5, tlsh in zip(malware_md5s, tlshs)}

    # Store "sketches" of each TLSH for fast lookup
    print("[-] Computing sketches")
    sys.stdout.flush()
    with multiprocessing.Pool(args.num_procs, maxtasksperchild=10) as pool:
        map_func = functools.partial(get_sketches, sketch_size=args.sketch_size)
        all_sketches = pool.map(map_func, (zip(md5s, tlshs)))
    sketch_md5s = {}
    for i in range(len(md5s)):
        sketches = all_sketches[i]
        for sketch in sketches:
            if sketch_md5s.get(sketch) is None:
                sketch_md5s[sketch] = set()
            sketch_md5s[sketch].add(md5s[i])

    # Use sketches to search for md5s with similar tlsh digests
    print("[-] Finding similar files")
    sys.stdout.flush()
    selected_md5s = set()
    similar_ids = []
    for i in range(len(md5s)):
        md5_1 = md5s[i]
        tlsh_1 = md5_tlshs[md5_1]
        sketches = all_sketches[i]

        # Sort sketches from most -> least unique, ignoring singletons
        sketches = [sketch for sketch in sketches
                    if len(sketch_md5s[sketch]) > 1]
        sketches.sort(key=lambda l:len(l))
        found_hash = False
        for sketch in sketches:
            if found_hash:
                break
            cur_md5s = sketch_md5s[sketch]
            cur_md5s = cur_md5s.difference(selected_md5s)
            if md5_1 in cur_md5s:
                cur_md5s.remove(md5_1)
            for md5_2 in cur_md5s:
                tlsh_2 = md5_tlshs[md5_2]
                diff = tlsh.diffxlen(tlsh_1, tlsh_2)
                if diff < args.tlsh_threshold:
                    print(md5_1, md5_2, tlsh_1, tlsh_2)
                    found_hash = True
                    selected_md5s.add(md5_2)
                    similar_ids.append((md5_ids[md5_1], md5_ids[md5_2]))
                    break

    # Write list of similar ids to a .pkl file
    with open(os.path.join(args.data_dir, "similar_ids.pkl"), "wb") as f:
        pickle.dump(similar_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("[+] Identified {} pairs of similar files".format(len(similar_ids)))
    sys.stdout.flush()
