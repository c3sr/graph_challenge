
A = [
    [0,1,4],
    [0,1],
    [2],
    [3,5],
    [4],
    [3,5],
]

B = [
    [0,1,4],
    [0,1],
    [2],
    [3,5],
    [4],
    [3,5],
]

A = [
    [1,2],
    [2,3,4],
    [3,4],
    [4],
    [],
]

B = A


def esc(A,B, rows=set([0,1,2,3,4])):
    C = 0
    # expansion
    C_hat = []
    for i in rows: # range(C.num_rows)
        for k in A[i]:
            for j in B[k]:
                C_hat += [(i, j)]
    print(len(C_hat), C_hat)


    # sort
    C_hat = sorted(C_hat)
    print(C_hat)

    # contract
    for i in rows: # range(C.num_rows)
        v = 0
        # extract partial products for row i
        J = [t[1] for t in C_hat if t[0] == i]
        print(i, J, A[i])
        for j in J: # if i,j is non-zero in A
            if j in A[i]:
                C += 1
    print(C)
    return C


# should also load A into shared memory.
# assign one thread block to each chunk of a row of A
def local_esc(A,B):
    ## local storage for each row
    sharedSz = 4
    sharedC_hat = [[]  for i in range(5)]
    sharedC = [0 for i in range(5)]

    ## rows that won't fit in local memory
    globalRows = set()

    C = 0

    # analysis
    # figure out how many expanded entries will exist for each row
    # implemented as a block prefix scan
    for i in range(5): # range(C.num_rows)
        numPartialProducts = 0
        for k in A[i]:
            for j in B[k]:
                numPartialProducts += 1
        if numPartialProducts > sharedSz:
            globalRows.add(i)
    print(globalRows)

    # expansion
    # place of expanding into C_hat is from the prefix scan
    C_hat = []
    for i in range(5): # range(C.num_rows)
        if i not in globalRows:
            for k in A[i]:
                for j in B[k]:
                    sharedC_hat[i] += [(i, j)]
    print("local memory: (after expand)")
    for i,s in enumerate(sharedC_hat):
        print(s)


    # sort
    # CUB thread block sort
    for i in range(5): # range(C.num_rows)
        sharedC_hat[i] = sorted(sharedC_hat[i])
    print("local memory: (after sort)")
    for i,s in enumerate(sharedC_hat):
        print(s)

    # contract
    for i in range(5): # range(C.num_rows)
        # extract partial products for row i
        J = sharedC_hat[i]
        print(i, J, A[i])
        for j in J: # if i,j is non-zero in A
            if j in A[i]: # FIXME: this should only go over our piece of A
                sharedC[i] += 1
    print("local memory: (after contract)")
    for i,s in enumerate(sharedC):
        print(s)

    # reduction over  C
    C = 0
    for i in range(5):
        C += sharedC[i]
    print("C after local")
    print(C)

    C += esc(A,B, rows=globalRows)
    print("C after global")
    print(C)
    return C




local_esc(A,B)