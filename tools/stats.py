#! /usr/bin/env python

'''convert a graph challenge tsv ascii edge list to binary format (.bel)
the binary format is for each edge
* 64-bit integer dst
* 64-bit integer src
* 64-bit integer weight
all numbers are stored little endian (least significant byte first)

the number of edges is the byte-length of the file divided by 24

you can view the produced file with 
xxd -c 24 <file> to see one edge per line
'''

from __future__ import print_function
import sys
import struct
import os
import math
import logging

def histogram(xs, num_buckets):
    min_x = min(xs)
    max_x = max(xs)
    bucket_size = (max_x - min_x + num_buckets) / num_buckets
    buckets = [0 for i in range(num_buckets)]
    for x in xs:
        bucket_id = int((x - min_x) / bucket_size)
        buckets[bucket_id] += 1
    return buckets

def histogram_power(xs):
    max_x = max(xs)
    max_bucket = int(math.ceil(math.log(max_x, 2.0)))
    if max_x % 2 == 0:
        max_bucket += 1
    buckets = [0 for i in range(max_bucket)]
    for x in xs:
        bucket_id = int(math.log(x, 2.0))
        buckets[bucket_id] += 1
    return buckets

def make_sf(num_edges, min_degree, max_degree, num_buckets, gamma):
    bucket_size = (max_degree - min_degree + num_buckets) / num_buckets
    buckets = [0 for _ in range(num_buckets)]
    for i in range(num_buckets):
        bucket_floor = min_degree + i * num_buckets
        bucket_ceil = bucket_floor + bucket_size
        bucket_count = 0
        for k in range(bucket_floor, bucket_ceil):
            p = k ** (-1 * gamma)
            bucket_count += p * num_edges
        buckets[i] = bucket_count

    # scale = num_edges / sum(buckets)
    # buckets = [int(round(b * scale)) for b in buckets ]
    buckets = [int(round(b)) for b in buckets ]
    return buckets

def avg(xs):
    return sum(xs) / float(len(xs))

def med(xs):
    s = sorted(xs)
    if len(s) % 2 == 0:
        return s[len(s) / 2]
    else:
        return (s[len(s) / 2 - 1] + s[len(s) / 2]) / 2.0

def var(xs):
    x_bar = avg(xs)
    num = sum((x - x_bar)**2 for x in xs)
    den = len(xs) - 1
    return num / den

print("graph, nodes, edges, in_min, in_max, in_avg, in_med, out_min, out_max, out_avg, out_med, out_var, in_ssd, out_ssd, buckets")

for bel_path in sys.argv[1:]:
    assert bel_path.endswith(".bel")

    print("{}, ".format(os.path.basename(bel_path)), end = "")
    sys.stdout.flush()

    in_degree = {}
    out_degree = {}
    nodes = set()

    with open(bel_path, 'rb') as inf:
        buf = inf.read(24)
        while buf:
            if len(buf) != 24:
                logging.error("expected 24B, read {}B from {}".format(len(buf), bel_path))
                sys.exit(1)
            dst, src, _ = struct.unpack("<QQQ", buf)
            
            if src < dst:
                # update incidence
                if dst not in in_degree:
                    in_degree[dst] = 0
                in_degree[dst] += 1

                # update adjacency
                if src not in out_degree:
                    out_degree[src] = 0
                out_degree[src] += 1

                # update nodes
                nodes.add(src)
                nodes.add(dst)

            buf = inf.read(24)

    # compute nodes
    num_nodes = len(nodes)
    # print("nodes:", len(nodes))

    # compute in degree and out degree
    in_degree = in_degree.values()
    out_degree = out_degree.values()

    num_edges = sum(in_degree)

    # in-degree statistics
    min_in = min(in_degree)
    max_in = max(in_degree)
    avg_in = avg(in_degree)
    med_in = med(in_degree)

    # out-degree statistics
    min_out = min(out_degree)
    max_out = max(out_degree)
    avg_out = avg(out_degree)
    med_out = med(out_degree)
    var_out = var(out_degree)


    ssd_in = sum(d ** 2 for d in in_degree)
    ssd_out = sum(d ** 2 for d in out_degree)
    # print("out ssd:", out_ssd)
    # print("in ssd :", in_ssd)

    # NUM_BUCKETS = 20
    # histo_out = histogram(out_degree, NUM_BUCKETS)
    # histo_in = histogram(in_degree, NUM_BUCKETS)
    histo_out = histogram_power(out_degree)
    histo_in = histogram_power(in_degree)

    # histo_sf2 = make_sf(sum(in_degree), min_out, max_out, NUM_BUCKETS, 2)
    # histo_sf3 = make_sf(sum(in_degree), min_out, max_out, NUM_BUCKETS, 3)

    # print("sf2:", histo_sf2)
    # print("sf3:", histo_sf3)

    print("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
        num_nodes,
        num_edges,
        min_in,
        max_in,
        avg_in,
        med_in,
        min_out,
        max_out,
        avg_out,
        med_out,
        var_out,
        ssd_in,
        ssd_out,
        ", ".join(str(e) for e in histo_out)
    ))