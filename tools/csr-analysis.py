from __future__ import print_function
import logging
import sys
import os
import struct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import partition
import numpy as np
import networkx as nx
import pymetis


logging.basicConfig(level=logging.INFO)


def get_color(i):
    colors = ["red", "green", "blue", "orange"]
    return colors[i % len(colors)]
    
    
PAGE_SIZE = 65536
PAGE_SIZE = 4096
ELEMENT_SIZE = 4
NUM_PARTS = 4



def plot_nonzeros(ax, edges, color):
    y = [ src for src, _, _ in edges]
    x = [ dst for _, dst, _ in edges]
    ax.scatter(x,y, s=1, marker=',', color=color)
    # ax.set_xlim(left=0)
    # ax.set_ylim(top=0)


def plot_pages(ax, pages, color):
    if not pages:
        return
    rects = []
    for p in pages:
        rects.append(Rectangle( (p * PAGE_SIZE, 0), PAGE_SIZE, 1))
    
    maxP = max(p for p in pages)

    colors = [color for p in pages]

    pc = PatchCollection(rects, edgecolor=None, alpha=0.5, facecolors=colors)
    cur_lim, _ = ax.get_xlim()
    ax.set_xlim(right = max(cur_lim, (maxP + 1) * PAGE_SIZE))
    # ax.set_xlim(right = max(rows))
    # ax.set_ylim(top=0)
    # ax.set_ylim(bottom = max(rows))

    ax.add_collection(pc)

def plot_rows(ax, rows, color):
    if not rows:
        return
    rects = [Rectangle( (0, r-0.5), r, 1) for r in rows]
    colors = [color for r in rows]

    pc = PatchCollection(rects, edgecolor=None, alpha=0.5, facecolors=colors)
    ax.set_xlim(left = 0)
    ax.set_ylim(top = 0)

    _, oldRight = ax.get_xlim()
    oldBottom, _ = ax.get_ylim()

    newRight = max(max(rows), oldRight)
    newBottom = max(max(rows), oldBottom)

    ax.set_xlim(right = newRight)
    ax.set_ylim(bottom = newBottom)

    ax.add_collection(pc)



def pct(f):
    """return a perect of f rounded to two places"""
    x = f * 100
    if x > 10:
        return int(x + 0.5)
    elif x > 1:
        return round(x, 1)
    return round(x, 2)






def color_rows(ax, colors, width):
    rects = []
    for r,color in enumerate(colors):
        rects += [Rectangle( (0, r-0.5), width, 1)]

    pc = PatchCollection(rects, edgecolor=None, alpha=0.5, facecolors=colors)

    ax.add_collection(pc)




def ceil_int(num, den=1):
    if isinstance(num, int) and isinstance(den, int):
        return int((num + den - 1) / den)
    return int(math.ceil(float(num) / float(den)))



for bel_path in sys.argv[1:]:
    assert bel_path.endswith(".bel")

    adj = nx.DiGraph()
    # adj.add_node(0)
    maxDst = -1
    maxSrc = -1

    logging.info("reading file")
    with open(bel_path, 'rb') as inf:
        buf = inf.read(24)
        while buf:
            if len(buf) != 24:
                logging.error("expected 24B, read {}B from {}".format(len(buf), bel_path))
                sys.exit(1)
            dst, src, _ = struct.unpack("<QQQ", buf)
            dst -= 1
            src -= 1
            assert dst >= 0
            assert src >= 0
            if src > dst:
                maxSrc = max(maxSrc, src)
                maxDst = max(maxDst, dst)
                # update adjacency
                adj.add_edge(src, dst)

            buf = inf.read(24)


    logging.info("sort node list")
    nodelist = sorted(adj.nodes())
    logging.info("build csr")
    csr = nx.to_scipy_sparse_matrix(adj, nodelist=nodelist)


    logging.info("build page graph")
    pageGraph = nx.Graph()
    curNzIdx = 0


    for row, col in adj.edges():
        # print(row)
        rowStartOff = csr.indptr[row]
        rowEndOff = csr.indptr[row + 1]
        rowNnz = rowEndOff - rowStartOff
        assert rowNnz >= 0
        if rowNnz > 0:
            pageFirst = int(rowStartOff * ELEMENT_SIZE / PAGE_SIZE)
            pageLast = int((rowEndOff - 1) * ELEMENT_SIZE / PAGE_SIZE)
            rowPages = range(pageFirst, pageLast+1)
        else:
            rowPages = []

        colStartOff = csr.indptr[col]
        colEndOff = csr.indptr[col + 1]
        colNnz = colEndOff - colStartOff
        assert colNnz >= 0
        if colNnz > 0:
            pageFirst = int(colStartOff * ELEMENT_SIZE / PAGE_SIZE)
            pageLast = int((colEndOff - 1) * ELEMENT_SIZE / PAGE_SIZE)
            colPages = range(pageFirst, pageLast+1) 
        else:
            colPages = []

        for rowPage in rowPages:
            for colPage in colPages:
                if pageGraph.has_edge(rowPage, colPage):
                    pageGraph[rowPage][colPage]['weight'] += 1
                else:
                    pageGraph.add_edge(rowPage, colPage, weight=1)



    logging.info("partition pageGraph")
    (edgecuts, parts) = pymetis.part_graph(NUM_PARTS, adjacency=pageGraph)
    print(edgecuts, "edgecuts")


    # logging.info("plot pageGraph")
    # fig, ax = plt.subplots(1)
    # labels = nx.get_edge_attributes(pageGraph,'weight')
    # # colors = [float(w)/1000 for _, _, w in pageGraph.edges.data('weight')]
    # # print(colors)
    # ax.set_title("Unweighted page partition")
    # nx.drawing.draw_circular(pageGraph, ax=ax, node_color=parts, with_labels=True)
    # pos=nx.get_node_attributes(pageGraph,'pos')
    # nx.draw_networkx_edge_labels(pageGraph, pos=nx.circular_layout(pageGraph), ax=ax, edge_labels=labels)

    logging.info("build page graph csr")
    pageGraphCsr = nx.to_scipy_sparse_matrix(pageGraph, weight='weight')

    logging.info('weighted page graph partition')
    xadj = pageGraphCsr.indptr
    adjncy = pageGraphCsr.indices
    adjwgt = pageGraphCsr.data
    (edgecuts, parts) = pymetis.part_graph(NUM_PARTS, xadj=xadj, adjncy=adjncy, eweights=adjwgt)
    print(edgecuts, "edge cuts")
    # sys.exit(1)


    # print(pageGraph.edges(data=True))
    # logging.info("plot pageGraph")
    # fig, ax = plt.subplots(1)
    # ax.set_title("Weighted page partition")
    # nx.drawing.draw_kamada_kawai(pageGraph, ax=ax, node_color=parts, with_labels=True)
    # pos=nx.get_node_attributes(pageGraph,'pos')
    # labels = nx.get_edge_attributes(pageGraph,'weight')
    # nx.draw_networkx_edge_labels(pageGraph, pos=nx.kamada_kawai_layout(pageGraph), ax=ax, edge_labels=labels)
    # plt.show()


    logging.info("build line graph")
    lg = nx.line_graph(adj)

    rowCounter = [set() for i in range(len(csr.indptr)-1)] # which partition accessed each row
    pageCounter = [set() for i in range(int((csr.nnz * ELEMENT_SIZE + PAGE_SIZE - 1) / PAGE_SIZE))] # which partition accessed each page
    logging.info("csr nzs covers {} pages".format(len(pageCounter)))

    logging.info("partitioning")

    # edges = partition.nnz(adj, NUM_PARTS)
    edges = partition.hilbert(adj, NUM_PARTS, maxSrc, maxDst)
    # edges = partition.tiled_hilbert(adj, NUM_PARTS, maxSrc, maxDst)
    # edges = partition.strided_rows(adj, NUM_PARTS)
    # edges = partition.strided_nnz(adj, NUM_PARTS)
    # edges = partition.metis(adj, NUM_PARTS)

    for i in range(NUM_PARTS):
        print(sum(1 for _,_,p in edges if p == i), "edges in partition", i)

    logging.info("determining row and page accesses for each partition")
    for edge in edges:
        src, dst, part = edge
        # row access for src
        rowCounter[src].add(part)

        # row access for dst
        rowCounter[dst].add(part)

        # page access for src
        rowStartOff = csr.indptr[src]
        rowEndOff = csr.indptr[src + 1]
        if rowStartOff != rowEndOff:
            pageFirst = int(rowStartOff * ELEMENT_SIZE / PAGE_SIZE)
            pageLast = int((rowEndOff - 1) * ELEMENT_SIZE / PAGE_SIZE)
            for page in range(pageFirst, pageLast+1):
                pageCounter[page].add(part)

        # page access for dst
        rowStartOff = csr.indptr[dst]
        rowEndOff = csr.indptr[dst + 1]
        if rowStartOff != rowEndOff:
            pageFirst = int(rowStartOff * ELEMENT_SIZE / PAGE_SIZE)
            pageLast = int((rowEndOff - 1) * ELEMENT_SIZE / PAGE_SIZE)
            for page in range(pageFirst, pageLast+1):
                pageCounter[page].add(part)


    numPages = (csr.nnz * ELEMENT_SIZE + PAGE_SIZE - 1) / PAGE_SIZE
    numRows = len(csr.indptr) - 1
    print(numPages, "pages")
    print(numRows, "rows")

    print("cfx: page count")
    conflictCount = {}
    for parts in pageCounter:
        cfx = len(parts)
        if cfx not in conflictCount:
            conflictCount[cfx] = 0
        conflictCount[cfx] += 1
    for k,v in sorted(conflictCount.items()):
        print(k, ":", v, pct(float(v)/ float(numPages)))
    print("page dup:", float(sum(len(c) for c in pageCounter)) / len(pageCounter))

    print("cfx: row count")
    conflictCount = {}
    for parts in rowCounter:
        cfx = len(parts)
        if cfx not in conflictCount:
            conflictCount[cfx] = 0
        conflictCount[cfx] += 1
    for k,v in sorted(conflictCount.items()):
        print(k, ":", v, pct(float(v)/ float(numRows)))


    # sys.exit(1)
    # plot edge assignments
    logging.info("plotting nonzeros")
    fig, ax = plt.subplots(1)
    ax.invert_yaxis()
    for i in range(NUM_PARTS):
        partEdges  = [(s, d, p) for s,d,p in edges if p == i]
        plot_nonzeros(ax, partEdges, get_color(i))

    logging.info("plotting rows")
    fig, ax = plt.subplots(1)
    ax.invert_yaxis()
    # plot row access conflicts
    logging.info("enumerate row conflicts")
    rows = [row for row, counter in enumerate(rowCounter) if len(counter) > 1]
    logging.info("plot row conflicts")
    plot_rows(ax, rows, 'gray')

    # plot row accesses
    for part in range(NUM_PARTS):
        logging.info("enumerate rows for partition {}".format(part))
        rows = [row for row, counter in enumerate(rowCounter) if len(counter) == 1 and next(iter(counter)) == part]
        logging.info("plot rows")
        plot_rows(ax, rows, get_color(part))

    fig, ax = plt.subplots(1)
    # plot page access conflicts
    for page, counter in enumerate(rowCounter):
        pages = [page for page, counter in enumerate(pageCounter) if len(counter) > 1]
    plot_pages(ax, pages, 'gray')

    # plot pages
    for part in range(NUM_PARTS):
        logging.info("enumerate pages for partition {}".format(part))
        pages = [page for page, counter in enumerate(pageCounter) if len(counter) == 1 and next(iter(counter)) == part]
        logging.info("plot pages")
        plot_pages(ax, pages, get_color(part))

    logging.basicConfig(level=logging.WARN)    

    plt.show()