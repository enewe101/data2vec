import scipy.sparse

def write_pairlist(path, pairlist):
    with open(path, 'w') as pair_file:
        for pair in pairlist:
            pair_file.write('{},{}\n'.format(*pair))


def read_pairlist(path):
    with open(path) as pair_file:
        for line in pair_file:
            yield tuple(int(element) for element in line.strip().split(','))


def pairlist_to_coo(pairs, shape=None, symmetric=False):
    """
    Convert `pairs` to a scipy.sparse.coo_matrix.

    Inputs
     - pairs - dict - The counter whose keys are pairs, and whose
        values represent the count the number of occurrences of those pairs.
     - shape - 2-tuple or None - force the shape of the recorded matrix to be
        the shape provided, by default it will only be big enough to express
        the index pairs provided.
     - symmetric - bool - whether to cause the recorded matrix to be symmetric
        i.e. the count at every (i,j) will be duplicated to (j,i).  Note that
        symmetric only works if the shape is square, so you should force a 
        square shape if using `symmetric=True`.
    """

    data = list(pairs.values())
    I = [i for i,j in pairs.keys()]
    J = [j for i,j in pairs.keys()]

    coo_matrix = scipy.sparse.coo_matrix((data, (I, J)), shape=shape)

    if symmetric:
        diags = scipy.sparse.diags(coo_matrix.diagonal(), shape=shape)
        # This implicitly casts to CSR format; we need to cast back to COO.
        csr_matrix = coo_matrix + coo_matrix.T - diags
        coo_matrix = csr_matrix.tocoo()

    return coo_matrix



