from __future__ import print_function

import math

def rowwise(adj, n):
    edges = []

    csrNnz = sum(len(row) for row in adj.values())
    nnzPerPart = int((csrNnz + n - 1) / n)

    nz = 0
    for src, row in adj.items():
        for dst in row:
            part = int(nz / nnzPerPart)
            edges.append((src, dst, part))
            nz += 1

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

    pow2 = 2 ** int(math.ceil(math.log(max(maxSrc+1, maxDst+1), 2)))

    hilbertPath = []
    for row, cols in adj.items():
        for col in cols:
            d = xy2d(pow2, col, row)
            hilbertPath += [(d, row, col)]
    hilbertPath = sorted(hilbertPath)
    edges = []
    for part, chunk in enumerate(chunks(hilbertPath, n)):
        for (_, src, dst) in chunk:
            edges.append((src, dst, part))
    return edges

def tiled_hilbert(adj, n, maxSrc, maxDst, tileSize = 32):
    """ only order edges by which tileSize X tileSize tile they fall in"""
    # determine the number of tiles in each dimension
    numSrcTiles = ceil_int(maxSrc, tileSize)
    numDstTiles = ceil_int(maxDst, tileSize)

    # determine the largest power of 2 that covers the tiles
    pow2 = 2 ** ceil_int(math.log(max(numSrcTiles, numDstTiles), 2))

    hilbertPath = []
    for row, cols in adj.items():
        for col in cols:
            srcTile = ceil_int(row, tileSize)
            dstTile = ceil_int(col, tileSize)
            dTile = xy2d(pow2, srcTile, dstTile)
            hilbertPath += [(d, row, col)]
    hilbertPath = sorted(hilbertPath)
    parts = [c for c in chunks(hilbertPath, n)]
    return parts