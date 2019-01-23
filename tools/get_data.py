"""
Download tsv and tsv.gz files
"""

from __future__ import print_function

import argparse
import hashlib
import os
import urllib
import sys
import urlparse
import struct
import gzip
import shutil

# Graphs in (name, url, md5) format
GRAPHS = [
    # graphs with 0 triangles
    ("https://graphchallenge.s3.amazonaws.com/synthetic/gc3/Theory-16-25-81-Bk.tsv", "4e38a437650600c8fa6cd1b85880f05b"),
    # graphs with many triangles
    ("https://graphchallenge.s3.amazonaws.com/synthetic/gc3/Theory-16-25-81-B1k.tsv", "6d1e80bf560ab148b6d4de4cb429980d"),
    # graphs with some triangles
    ("https://graphchallenge.s3.amazonaws.com/synthetic/gc3/Theory-16-25-81-B2k.tsv", "bb572123192ef15e21a49c6154cf2ebc"),
    # protein k-mer
    ("https://graphchallenge.s3.amazonaws.com/synthetic/gc6/V2a.tsv", "b3f08b442565a5727ddeb94af5814d6a"), # 5, 2.1G
    # Synthetic Datasets
    ("https://graphchallenge.s3.amazonaws.com/synthetic/graph500-scale18-ef16/graph500-scale18-ef16_adj.tsv.gz", "b942970d403218b1ec4ed2d4cd76b52c"),
    ("https://graphchallenge.s3.amazonaws.com/synthetic/graph500-scale22-ef16/graph500-scale22-ef16_adj.tsv.gz", "15d99816ffc4f4e166c4ba46c31b72b1"),
    # SNAP datasets
    ("https://graphchallenge.s3.amazonaws.com/snap/amazon0302/amazon0302_adj.tsv", "8b6f22a1e4fda1aeb5dd8132c9c860af"),
    ("https://graphchallenge.s3.amazonaws.com/snap/amazon0312/amazon0312_adj.tsv", "3bd20f592f00e03291ed314eef9d8333"),
    ("https://graphchallenge.s3.amazonaws.com/snap/amazon0505/amazon0505_adj.tsv", "3644d53c530658164d2ee0c7a40bcb6b"),
    ("https://graphchallenge.s3.amazonaws.com/snap/amazon0601/amazon0601_adj.tsv", "149551a622d68e76c0227603e53d8e46"),
    ("https://graphchallenge.s3.amazonaws.com/snap/soc-Slashdot0902/soc-Slashdot0902_adj.tsv", "fe7a3d71eeb11a94ecdf0a0b84766c93"),
    ("https://graphchallenge.s3.amazonaws.com/snap/roadNet-CA/roadNet-CA_adj.tsv", "d0e4b76f314e86ca78c313bb64ab5aa7"),
    ("https://graphchallenge.s3.amazonaws.com/snap/roadNet-PA/roadNet-PA_adj.tsv", "7ee3faf91c95b22b1398618daa31fb3a"),
    ("https://graphchallenge.s3.amazonaws.com/snap/roadNet-TX/roadNet-TX_adj.tsv", "1bd453e8551b1432eb8a81eab7325c88"),
    # MAWI Datasets
    ("https://graphchallenge.s3.amazonaws.com/synthetic/gc5/201512012345.v18571154_e38040320.tsv","919a22f1456d9fd978ba8d12ea96579c"), # 1
    ("https://graphchallenge.s3.amazonaws.com/synthetic/gc5/201512020330.v226196185_e480047894.tsv", "3b7f0546835d1f10cc41312f7a12b8d1"), # 5
]


def get_remote_size(url):
    site = urllib.urlopen(url)
    meta = site.info()
    print(meta.getheaders("Content-Length")[0])

def get_file_size(path):
    return os.stat(path).st_size

def is_file(path):
    return os.path.isfile(path)

def hash_file(path):
    with open(path, "rb") as f:
        hasher = hashlib.md5()
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
        return hasher.hexdigest().lower()

def get_basename(url):
    parsed = urlparse.urlparse(url)
    path = os.path.basename(parsed.path)
    return path

parser = argparse.ArgumentParser()

parser.add_argument("--out", default=".", help="output dir")

args = parser.parse_args()

output_dir = args.out

# check if output directory exists
if not os.path.isdir(output_dir):
    print("output directory {} does not exist".format(output_dir))
    sys.exit(-1)


matching_graphs = GRAPHS






for (url, expected_md5) in matching_graphs:
    needs_download = ""
    name = get_basename(url)
    dst = os.path.join(output_dir, name)

    # check if dst exists
    if not needs_download and not is_file(dst):
        needs_download = name + " missing"

    # compare hashes
    if not needs_download and expected_md5:
        try:
            actual_md5 = hash_file(dst)
            if actual_md5 != expected_md5.lower():
                needs_download = "hash mismatch"

        except IOError as e:
            needs_download = "file open error"


    if needs_download:
        print("DOWNLOAD", dst, "reason:", needs_download)
        urllib.urlretrieve(url, dst)
        if hash_file(dst) != expected_md5:
            print("MISMATCH", dst)
    else:
        print("MD5_MATCH", dst)

    # check if the file needs to be extracted
    if dst.endswith(".gz"):
        needs_extract = ""
        extracted_path = dst[:-3]
        if not needs_extract:
            try:
                actual_size = os.path.getsize(extracted_path)
            except OSError:
                needs_extract = extracted_path + " missing"

        if not needs_extract:
            with open(dst, "rb") as f:
                f.seek(-4, os.SEEK_END)
                buf = f.read(4)
                expected_size = struct.unpack("I", buf)[0]
            
            if actual_size % 2**32 != expected_size:
                needs_extract = "size mismatch"

        if needs_extract:
            print("EXTRACT", dst, "reason:", needs_extract)
            with gzip.open(dst, 'rb') as f_in, open(extracted_path, "w") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            print("EXTRACT_MATCH", extracted_path)
