from __future__ import print_function

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