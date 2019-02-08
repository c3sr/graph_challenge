from __future__ import print_function
import matplotlib.pyplot as plt
import logging
import sys
import os
import struct

def histogram(xs, num_buckets):
    min_x = min(xs)
    max_x = max(xs)
    bucket_size = (max_x - min_x + num_buckets) / num_buckets
    buckets = [0 for i in range(num_buckets)]
    bucket_edges = [i * bucket_size for i in range(num_buckets+1)]
    for x in xs:
        bucket_id = int((x - min_x) / bucket_size)
        buckets[bucket_id] += 1
    return buckets, bucket_edges

def cumsum(xs):
    c = [0 for _ in xs]
    total = 0
    for i,x in enumerate(xs):
        total += x
        c[i] = total
    return c


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

            buf = inf.read(24)


        out_degree = out_degree.values()

        # Use the histogram function to bin the data
        counts, bin_edges = histogram(out_degree, 20)

        s = sum(counts)
        counts = [0] + [e/s for e in counts]

        # Now find the cdf
        cdf = cumsum(counts)

        # And finally plot the cdf
        plt.yscale('log')
        plt.plot(bin_edges, cdf)

        plt.show()