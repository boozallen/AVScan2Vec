import os
import sys
import json
import argparse
import requests


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("hash_file", help="Path to file with hashes to query")
    parser.add_argument("api_file", help="Path to VirusTotal API key file")
    parser.add_argument("out_file", help="Path to file to write scan reports")
    args = parser.parse_args()

    # Read hashes from hash_file
    hashes = []
    with open(args.hash_file, "r") as f:
        for line in f:
            hashes.append(line.strip())

    # Read VirusTotal API key from api_file
    with open(args.api_file, "r") as f:
        api_key = f.read().strip()

    base_url = "https://www.virustotal.com/api/v3/files/{}"
    with open(args.out_file, "w") as f:
        for file_hash in hashes:
            print(file_hash)
            url = base_url.format(file_hash)
            headers = {"x-apikey": api_key}
            response = requests.get(url, headers=headers)
            try:
                report = json.loads(response.text)
                f.write("{}\n".format(json.dumps(report)))
            except JSONDecodeError as e:
                f.write("Unable to parse report for {}\n".format(file_hash))
