import time
from multiprocessing import Pool

import numpy as np
from scipy.sparse import issparse

from inphosemantics import load_matrix
from inphosemantics import dump_matrix

# TODO: Fix the datatypes during conversion from sparse to dense data
# structures. (TF matrices have floats as scalars now, for example.)

def norm(v):
    return np.sum(v**2)**(1./2)


def vector_cos(v,w):
    '''
    Computes the cosine of the angle between numpy arrays v and w.
    '''
    return np.sum(v*w) / (norm(v) * norm(w))



def row_fn(i):
    """
    """
    w = row_fn.matrix[i,:]
    if issparse(w):
        w = np.squeeze(w.toarray())

    return i, vector_cos(row_fn.v, w)


class Model(object):
    """
    """
    def __init__(self, matrix=None):

        self.matrix = matrix


    def load_matrix(self, filename):

        self.matrix = load_matrix(filename)


    def dump_matrix(self, filename):

        dump_matrix(self.matrix, filename)


    def filter_rows(self, row_filter):

        #Used to filter stop words. Zeroing the stop word row makes
        #the resulting cosine undefined; undefined cosines are later
        #filtered out.

        for i in row_filter:
            self.matrix[i,:] = np.zeros(self.matrix.shape[1])



    def similar_rows(self, row, filter_nan=False):

        return similar_rows(row, self.matrix, filter_nan=filter_nan)


    def similar_columns(self, column, filter_nan=False):

        return similar_rows(column, self.matrix.T, filter_nan=filter_nan)


    def simmat_rows(self, row_indices):

        sim_matrix = SimilarityMatrix(row_indices)
        sim_matrix.compute(self.matrix)
        return sim_matrix


    def simmat_columns(self, column_indices):

        sim_matrix = SimilarityMatrix(column_indices)
        sim_matrix.compute(self.matrix.T)
        return sim_matrix

        



def similar_rows(row, matrix, filter_nan=False):

    row_fn.v = matrix[row,:]
    if issparse(row_fn.v):
        row_fn.v = np.squeeze(row_fn.v.toarray())

    row_fn.matrix = matrix
    
    p = Pool()
    results = p.map(row_fn, range(matrix.shape[0]))
    p.close()
    
    # Filter out undefined results
    if filter_nan:
        results = [(r,v) for r,v in results if np.isfinite(v)]
        
    dtype = [('row', np.uint32), ('value', np.float)]
    # NB: numpy >= 1.4 sorts NaN to the end
    results = np.array(results, dtype=dtype)
    results.sort(order='value')
    results = results[::-1]
    
    return results



# TODO: Compress symmetric similarity matrix. Cf. scipy.spatial.distance.squareform
class SimilarityMatrix(object):

    def __init__(self, indices=None, matrix=None):

        self.indices = indices

        if matrix == None:
            self.matrix = np.zeros((len(self.indices), len(self.indices)))


    def compute(self, data_matrix):
        """
        Comparisons are row-wise
        """

        if issparse(data_matrix):

            row_fn.matrix = np.zeros((len(self.indices), data_matrix.shape[1]))

            for i in xrange(len(self.indices)):
                row_fn.matrix[i,:] = data_matrix[self.indices[i],:].toarray()

        else:
            row_fn.matrix = data_matrix[self.indices]
            

        for idx_sim,idx in enumerate(self.indices):

            row_fn.v = data_matrix[idx,:]
            if issparse(row_fn.v):
                row_fn.v = np.squeeze(row_fn.v.toarray())

            p = Pool()
            results = p.map(row_fn, range(idx_sim, len(self.indices)))
            p.close()

            values = zip(*results)[1]

            self.matrix[idx_sim,idx_sim:] = np.asarray(values)

        # Redundant representation of a symmetric matrix
        self.matrix += self.matrix.T - np.diag(np.diag(self.matrix))
