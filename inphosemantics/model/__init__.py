import time
from multiprocessing import Pool

import numpy as np
from scipy.sparse import issparse

from inphosemantics import load_matrix
from inphosemantics import dump_matrix as _dump_matrix




def norm(v):
    return np.sum(v**2)**(1./2)


def vector_cos(v,w):
    '''
    Computes the cosine of the angle between (row) vectors v and w.
    '''
    if issparse(v):
        v = v.todense()
    if issparse(w):
        w = w.todense()
    
    return np.sum(v*w) / (norm(v) * norm(w))



# Sit out here for multiprocessing
def row_fn(i):
    
    return i, vector_cos(row_fn.v1, row_fn.matrix[i,:])


def column_fn(i):

    return i, vector_cos(column_fn.v1, column_fn.matrix[:,i].T)




class Model(object):
    """
    """
    def __init__(self, matrix=None):

        self.matrix = matrix


    def train(self,
              corpus,
              token_type,
              stoplist=None):

        print 'This class should not be used directly. '\
              'Try a subclass corresponding to a model type.'


    def load_matrix(self, filename):

        self.matrix = load_matrix(filename)


    def dump_matrix(self, filename):

        _dump_matrix(self.matrix, filename)


    def filter_rows(self, row_filter):

        #Used to filter stop words. Zeroing the stop word row makes
        #the resulting cosine undefined; undefined cosines are later
        #filtered out.

        for i in row_filter:
            for j in xrange(self.matrix.shape[1]):
                self.matrix[i,j] = 0

            # self.matrix[i,:] = np.zeros(self.matrix.shape[1])



    #TODO: These are almost the same function....
    def similar_rows(self, row, filter_nan=False):

        row_fn.v1 = self.matrix[row,:]
        row_fn.matrix = self.matrix

        p = Pool()
        results = p.map(row_fn, range(self.matrix.shape[0]))
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



    def similar_columns(self, column, filter_nan=False):

        column_fn.v1 = self.matrix[:,column].T
        column_fn.matrix = self.matrix

        p = Pool()
        results = p.map(column_fn, range(self.matrix.shape[1]))
        p.close()

        # Filter out undefined results
        if filter_nan:
            results = [(c,v) for c,v in results if np.isfinite(v)]

        dtype = [('column', np.uint32), ('value', np.float)]
        # NB: numpy >= 1.4 sorts NaN to the end
        results = np.array(results, dtype=dtype)
        results.sort(order='value')
        results = results[::-1]

        return results
