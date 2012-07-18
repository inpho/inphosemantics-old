import os
import numpy as np

def load_matrix(filename):

    # The slice [()] is for the cases where np.save has stored a
    # sparse matrix in a zero-dimensional array

    return np.load(filename)[()]


def dump_matrix(matrix, filename):

    np.save(filename, matrix)

