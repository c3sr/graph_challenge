
from scipy.sparse import *
from scipy import *
import numpy as np
import sys

A = np.array([ 
[0,1,1,0,0],
[0,0,1,1,1],
[0,0,0,1,1],
[0,0,0,0,1],
[0,0,0,0,0],
 ])


U = csr_matrix(np.array([ 
[0,1,1,0,0],
[0,0,1,1,1],
[0,0,0,1,1],
[0,0,0,0,1],
[0,0,0,0,0],
 ]))

L = csr_matrix(np.array([ 
[0,0,0,0,0],
[1,0,0,0,0],
[1,1,0,0,0],
[0,1,1,0,0],
[0,1,1,1,0],
]))


print("L.L.*L",L.dot(L).multiply(L).sum())
print("L.U.*L",L.dot(U).multiply(L).sum())
print("U.L.*L",U.dot(L).multiply(L).sum())
print("U.U.*L",U.dot(U).multiply(L).sum())
print("L.L.*U",L.dot(L).multiply(U).sum())
print("L.U.*U",L.dot(U).multiply(U).sum())
print("U.L.*U",U.dot(L).multiply(U).sum())
print("U.U.*U",U.dot(U).multiply(U).sum())

print("L.L.*A",L.dot(L).multiply(L+U).sum())
print("L.U.*A",L.dot(U).multiply(L+U).sum())
print("U.L.*A",U.dot(L).multiply(L+U).sum())
print("U.U.*A",U.dot(U).multiply(L+U).sum())


# square two CSR matrices 
def square_scipy(A):
    P = A.dot(A).tocsr().sorted_indices()
    return P

# square two CSR matrices 
def square_esc(A, rows=None):
    """
    square CSR A using the esc method in Dalton (2012)
    """

    if not rows:
        rows = range(A.shape[0])

    # expand
    C_hat = []
    for i in rows: # for each row in C
        # print("row", i)
        # get the non-zero columns of A
        for k in A.indices[A.indptr[i]:A.indptr[i+1]]:
            # print(k)
            # each non-zero column of A grabs all entries from the corresponding row of B
            for j in A.indices[A.indptr[k]:A.indptr[k+1]]:
                C_hat += [(i, j)]
    print("after expand", C_hat)

    #sort
    C_hat = sorted(C_hat)
    # print("after sort  ", C_hat)

    #contract
    data = []
    rowInd = []
    colInd = []
    for i in range(A.shape[0]): # for each row in C
        v = 0
        J = [j for x,j in C_hat if x == i]
        for ji, j in enumerate(J):
            # print(ji, len(J))
            v += 1
            if ji >= len(J) - 1 or J[ji] != J[ji+1]:
                data += [v]
                rowInd += [i]
                colInd += [j]
                v = 0

    return csr_matrix((data, (rowInd, colInd))).sorted_indices()


# square two CSR matrices 
def square_esc_local(A, localMemSz, rows=None):
    """
    square CSR A using the esc method in Dalton (2012)
    """

    if not rows:
        rows = range(A.shape[0])

    # resulting matrix data
    data = []
    rowInd = []
    colInd = []

    # expand
    
    for i in rows: # for each row in C
        C_hat = [] # per-row C_hat
        # print("row", i)
        # get the non-zero columns of A
        for k in A.indices[A.indptr[i]:A.indptr[i+1]]:
            # print(k)
            # each non-zero column of A grabs all entries from the corresponding row of B
            for j in A.indices[A.indptr[k]:A.indptr[k+1]]:
                C_hat += [(i, j)]
        assert len(C_hat) <= 4
        print("row", i, "after expand", C_hat)

        #sort
        C_hat = sorted(C_hat)
        print("row", i, "after sort  ", C_hat)

    #contract
        v = 0
        J = [j for x,j in C_hat if x == i]
        for ji, j in enumerate(J):
            # print(ji, len(J))
            v += 1
            if ji >= len(J) - 1 or J[ji] != J[ji+1]:
                data += [v]
                rowInd += [i]
                colInd += [j]
                v = 0

    return csr_matrix((data, (rowInd, colInd))).sorted_indices()

def analysis_esc(A, n):
    """
    return sets of rows that produce (fewer, more) than n partial products
    """
    
    smallRows = set()
    bigRows = set()
    for i in range(A.shape[0]): # for each row in C
        numPartialProducts = 0
        # print("row", i)
        # get the non-zero columns of A
        for k in A.indices[A.indptr[i]:A.indptr[i+1]]:
            # print(k)
            # each non-zero column of A grabs all entries from the corresponding row of B
            numPartialProducts += A.indptr[k+1] - A.indptr[k]
        if numPartialProducts > n:
            bigRows.add(i)
        else:
            smallRows.add(i)

    return smallRows, bigRows

print(square_scipy(U))

smallRows, bigRows = analysis_esc(U, 2)
print("small:", smallRows, "big:", bigRows)

smallP = square_esc_local(U, 2, smallRows)
bigP = square_esc(U, bigRows)
print(bigP)
print(smallP)





sys.exit()


