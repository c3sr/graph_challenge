from __future__ import print_function

import logging
import math

logger = logging.getLogger(__name__)

def ceil_int(num, den=1):
    if isinstance(num, int) and isinstance(den, int):
        return int((num + den - 1) / den)
    return int(math.ceil(float(num) / float(den)))

def nnz(adj, n):
    logger.info("nnz")
    edges = []

    csrNnz = len(adj.edges())
    nnzPerPart = int((csrNnz + n - 1) / n)

    nz = 0
    for src, dst in adj.edges():
        part = int(nz / nnzPerPart)
        edges.append((src, dst, part))
        nz += 1

    return edges


def strided_nnz(adj, n, tileSize=10000):
    logger.info("strided_nnz, tileSize={}".format(tileSize))
    edges = []

    nz = 0
    for src, row in adj.items():
        for dst in row:
            part = int(nz / tileSize) % n
            edges.append((src, dst, part))
            nz += 1

    return edges


def strided_rows(adj, n, tileSize=10000):
    logger.info("strided_row, tileSize={}".format(tileSize))
    edges = []

    for src, cols in adj.items():
        part = int(src / tileSize) % n
        for dst in cols:
            edges.append((src, dst, part))

    return edges





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


def chunks(l, n):
    """ Yield n successive chunks from l.
    """
    newn = int(1.0 * len(l) / n + 0.5)
    for i in range(0, n-1):
        yield l[i*newn:i*newn+newn]
    yield l[n*newn-newn:]


def hilbert(adj, n, maxSrc, maxDst):
    logger.info("hilbert")
    pow2 = 2 ** ceil_int(math.log(max(maxSrc+1, maxDst+1), 2))
    assert pow2 >= maxSrc
    assert pow2 >= maxDst

    hilbertPath = []
    for row, col in adj.edges():
        d = xy2d(pow2, col, row)
        hilbertPath += [(d, row, col)]
    hilbertPath = sorted(hilbertPath)
    edges = []
    for part, chunk in enumerate(chunks(hilbertPath, n)):
        for (_, src, dst) in chunk:
            edges.append((src, dst, part))
    return edges

def tiled_hilbert(adj, n, maxSrc, maxDst, tileSize = 50000):
    """ only order edges by which tileSize X tileSize tile they fall in"""
    logger.info("tiled_hilbert, tileSize={}".format(tileSize))
    # determine the number of tiles in each dimension
    numSrcTiles = ceil_int(maxSrc, tileSize)
    numDstTiles = ceil_int(maxDst, tileSize)

    # determine the largest power of 2 that covers the tiles
    exp = math.log(max(numSrcTiles+1, numDstTiles+1), 2)
    pow2 = 2 ** ceil_int(exp)
    assert pow2 >= numSrcTiles
    assert pow2 >= numDstTiles

    hilbertPath = []
    for row, cols in adj.items():
        srcTile = ceil_int(row, tileSize)
        for col in cols:
            dstTile = ceil_int(col, tileSize)
            dTile = xy2d(pow2, dstTile, srcTile)
            hilbertPath += [(dTile, row, col)]
    hilbertPath = sorted(hilbertPath)
    edges = []
    for part, chunk in enumerate(chunks(hilbertPath, n)):
        for (_, src, dst) in chunk:
            edges.append((src, dst, part))
    return edges

def metis(adj, n):
    import pymetis
    (edgecuts, parts) = pymetis.part_graph(n, adjacency=adj)
    edges = []
    for src, dst in adj.edges():
        edges.append((src, dst, parts[src]))
    return edges