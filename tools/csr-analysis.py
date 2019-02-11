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
import partition


logging.basicConfig(level=logging.DEBUG)


DEVICE_ID_TO_COLOR = {
    0: "red",
    1: "green",
    2: "blue",
    3: "orange",
}

PAGE_SIZE = 65536
ELEMENT_SIZE = 4

def pct(f):
    x = f * 100
    if x > 10:
        return int(x + 0.5)
    elif x > 1:
        return round(x, 1)
    return round(x, 2)

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
    maxDst = -1

    logging.info("reading file")
    with open(bel_path, 'rb') as inf:
        buf = inf.read(24)
        while buf:
            if len(buf) != 24:
                logging.error("expected 24B, read {}B from {}".format(len(buf), bel_path))
                sys.exit(1)
            dst, src, _ = struct.unpack("<QQQ", buf)

            if src > dst:
                maxDst = max(maxDst, dst)
                # update adjacency
                if src not in adj:
                    adj[src] = []
                adj[src].append(dst)

            buf = inf.read(24)

    logging.info("nnz")
    csrRowPtr = []
    csrNnz = sum(len(row) for row in adj.values())
    csrColInd = [0 for _ in range(csrNnz)]

    logging.info("building csr")
    curRow = 0
    curNzIdx = 0
    for row in sorted(adj.keys()):
        while curRow <= row:
            csrRowPtr.append(curNzIdx)
            curRow += 1
        for col in adj[row]:
            csrColInd[curNzIdx] = col
            curNzIdx += 1
    csrRowPtr.append(curNzIdx)

    rowCounter = [set() for i in range(len(csrRowPtr)-1)] # which device accessed each row
    pageCounter = [set() for i in range(int((csrNnz * ELEMENT_SIZE + PAGE_SIZE - 1) / PAGE_SIZE))] # which device accessed each page
    logging.info("csr nzs covers {} pages".format(len(pageCounter)))

    logging.info("partitioning")
    NUM_PARTS = 2
    edges = partition.rowwise(adj, NUM_PARTS)
    for i in range(NUM_PARTS):
        print(sum(1 for _,_,p in edges if p == i), "edges in partition", i)




    for edge in edges:
        src, dst, part = edge
        # row access for src
        rowCounter[src].add(part)

        # row access for dst
        rowCounter[dst].add(part)

        # page access for src
        rowStartOff = csrRowPtr[src]
        rowEndOff = csrRowPtr[src + 1]
        if rowStartOff != rowEndOff:
            pageFirst = int(rowStartOff * ELEMENT_SIZE / PAGE_SIZE)
            pageLast = int((rowEndOff - 1) * ELEMENT_SIZE / PAGE_SIZE)
            for page in range(pageFirst, pageLast+1):
                pageCounter[page].add(part)

        # page access for dst
        rowStartOff = csrRowPtr[dst]
        rowEndOff = csrRowPtr[dst + 1]
        if rowStartOff != rowEndOff:
            pageFirst = int(rowStartOff * ELEMENT_SIZE / PAGE_SIZE)
            pageLast = int((rowEndOff - 1) * ELEMENT_SIZE / PAGE_SIZE)
            for page in range(pageFirst, pageLast+1):
                pageCounter[page].add(part)


    numPages = (csrNnz * ELEMENT_SIZE + PAGE_SIZE - 1) / PAGE_SIZE
    numRows = len(csrRowPtr) - 1

    print("cfx: page count")
    conflictCount = {}
    for parts in pageCounter:
        cfx = len(parts)
        if cfx not in conflictCount:
            conflictCount[cfx] = 0
        conflictCount[cfx] += 1
    for k,v in sorted(conflictCount.items()):
        print(k, ":", v, pct(v/numPages))

    print("cfx: row count")
    conflictCount = {}
    for parts in rowCounter:
        cfx = len(parts)
        if cfx not in conflictCount:
            conflictCount[cfx] = 0
        conflictCount[cfx] += 1
    for k,v in sorted(conflictCount.items()):
        print(k, ":", v, pct(v/ numRows))



