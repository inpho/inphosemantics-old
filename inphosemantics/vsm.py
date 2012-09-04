import numpy as np

from viewer.similarity import row_norms



def row_normalize(m):
    """
    Takes a 2-d array and returns it with its rows normalized
    """
    norms = row_norms(m)

    return m / norms[:, np.newaxis]



def test_row_normalize():

    m = np.random.random((5,7))

    m = row_normalize(m)

    print np.allclose(row_norms(m), np.ones(5))
