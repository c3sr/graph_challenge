from __future__ import print_function
import logging
import sys
import os
import struct
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


DEVICE_ID_TO_COLOR = {
    0: "red",
    1: "green",
    2: "blue",
    3: "orange",
}


# rotate/flip a quadrant appropriately
def rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n-1 - x
            y = n-1 - y
        
        # Swap x and y
        return y,x
    return x,y

# convert (x,y) to d
def xy2d (n, x, y):
    d = 0
    s = int(n / 2)
    while s > 0:
        rx = int((x & s) > 0)
        ry = int((y & s) > 0)
        a = s * s * ((3 * rx) ^ ry)
        d += a
        x, y = rot(s, x, y, rx, ry)
        s = int(s/2)
    return d


def color_rows(ax, colors, width):
    rects = []
    for r,color in enumerate(colors):
        rects += [Rectangle( (0, r-0.5), width, 1)]

    pc = PatchCollection(rects, edgecolor=None, alpha=0.5, facecolors=colors)

    ax.add_collection(pc)


def chunks(l, n):
    """ Yield n successive chunks from l.
    """
    newn = int(1.0 * len(l) / n + 0.5)
    for i in range(0, n-1):
        yield l[i*newn:i*newn+newn]
    yield l[n*newn-newn:]

def partition_hilbert(adj, n, maxSrc, maxDst):

    pow2 = 2 ** int(math.ceil(math.log2(max(maxSrc+1, maxDst+1))))

    hilbertPath = []
    for row, cols in adj.items():
        for col in cols:
            d = xy2d(pow2, col, row)
            hilbertPath += [(d, row, col)]
    hilbertPath = sorted(hilbertPath)
    parts = [c for c in chunks(hilbertPath, n)]
    return parts



def partition_bynz(adj, n, maxSrc, maxDst):
    partitions = [[] for i in range(n)]

    num_rows = len(adj.keys())
    num_edges = sum(len(cols) for _, cols in adj.items())

    edgeId = 0
    for row, cols in adj.items():
        for col in cols:
            partitions[int(edgeId * n / num_edges)] += [(edgeId, row, col)]
            edgeId += 1

    return partitions


def get_row_colors(partitions):
    accessedBy = {}
    for i, part in enumerate(partitions):
        print(i)
        for _, r, c in part:
            if r not in accessedBy:
                accessedBy[r] = set()
            if c not in accessedBy:
                accessedBy[c] = set()
            accessedBy[r].add(i)
            accessedBy[c].add(i)
    # print(accessedBy)

    colors = ["0.0" for i in range(max(accessedBy.keys())+1)]
    for row, accessors in accessedBy.items():
        if len(accessors) == 1:
            colors[row] = DEVICE_ID_TO_COLOR[next(iter(accessors))]
        else:
            colors[row] = str(0.25 * len(accessors))

    # print(colors)
    return colors


for bel_path in sys.argv[1:]:
    assert bel_path.endswith(".bel")

    adj = {}
    maxSrc = -1
    maxDst = -1

    with open(bel_path, 'rb') as inf:
        buf = inf.read(24)
        while buf:
            if len(buf) != 24:
                logging.error("expected 24B, read {}B from {}".format(len(buf), bel_path))
                sys.exit(1)
            dst, src, _ = struct.unpack("<QQQ", buf)

            if src > dst:
                maxSrc = max(maxSrc, src)
                maxDst = max(maxDst, dst)
                # update adjacency
                if src not in adj:
                    adj[src] = set()
                adj[src].add(dst)

            buf = inf.read(24)

    n = 2 ** int(math.ceil(math.log2(max(maxSrc+1, maxDst+1))))
    print(n)

    hilbertPath = partition_hilbert(adj, 4, maxSrc, maxDst)
    nzParts = partition_bynz(adj, 4, maxSrc, maxDst)

    fig, ax = plt.subplots(1)
    ax.invert_yaxis()
    
    for dev, part in enumerate(hilbertPath):
        y = [ r for _, r, _ in part]
        x = [ c for _, _, c in part]

        color = DEVICE_ID_TO_COLOR[dev]
        ax.plot(x, y, color=color)

    rowColors = get_row_colors(hilbertPath)
    color_rows(ax, rowColors, maxDst)


    fig, ax = plt.subplots(1)
    ax.invert_yaxis()
    for dev, part in enumerate(nzParts):
        y = [ r for _, r, _ in part]
        x = [ c for _, _, c in part]

        color = DEVICE_ID_TO_COLOR[dev]
        ax.plot(x, y, color=color)

    rowColors = get_row_colors(nzParts)
    color_rows(ax, rowColors, maxDst)


    plt.show()

