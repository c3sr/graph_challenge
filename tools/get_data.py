from __future__ import print_function

import argparse
import hashlib
import os
import urllib
import sys


# Graphs in (name, url, md5) format
GRAPHS = [
    # graphs with 0 triangles
    ("Theory-16-25-81-Bk.tsv", "https://graphchallenge.s3.amazonaws.com/synthetic/gc3/Theory-16-25-81-Bk.tsv", "4e38a437650600c8fa6cd1b85880f05b"),
    # graphs with many triangles
    ("Theory-16-25-81-B1k.tsv", "https://graphchallenge.s3.amazonaws.com/synthetic/gc3/Theory-16-25-81-B1k.tsv", "6d1e80bf560ab148b6d4de4cb429980d"),
    # graphs with some triangles
    ("Theory-16-25-81-B2k.tsv", "https://graphchallenge.s3.amazonaws.com/synthetic/gc3/Theory-16-25-81-B2k.tsv", "bb572123192ef15e21a49c6154cf2ebc"),
    ("V2a.tsv", "https://graphchallenge.s3.amazonaws.com/synthetic/gc6/V2a.tsv", "b3f08b442565a5727ddeb94af5814d6a"), # protein k-mer graph 5, 2.1G
]


def get_remote_size(url):
    site = urllib.urlopen(url)
    meta = site.info()
    print(meta.getheaders("Content-Length")[0])

def get_file_size(path):
    return os.stat(path).st_size

def is_file(path):
    return os.path.isfile(path)

parser = argparse.ArgumentParser()

parser.add_argument("--only", help="only download graphs with these names")
parser.add_argument("--out", default=".", help="output dir")

args = parser.parse_args()

output_dir = args.out

if args.only:
    matching_graphs = [(name, url, sha) for (name, url, sha) in GRAPHS if name in args.only]
else:
    matching_graphs = GRAPHS



for (name, url, expected_md5) in matching_graphs:
    needs_download = ""
    dst = os.path.join(output_dir, name)

    # check if dst exists
    if not needs_download and not is_file(dst):
        needs_download = "file missing"

    # compare hashes
    if not needs_download and expected_md5:
        try:
            with open(dst, "rb") as f:
                hasher = hashlib.md5()
                buf = f.read(65536)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(65536)
                if hasher.hexdigest().lower() != expected_md5.lower():
                    needs_download = "hash mismatch"
        except IOError as e:
            print(dst, "does not exist")
            needs_download = "file open error"

    if needs_download:
        print("DOWNLOAD", dst, "reason:", needs_download)
        urllib.urlretrieve(url, dst)
    else:
        print("SUCCESS", dst)