import time
from multiprocessing import Pool

import numpy as np

from inphosemantics import load_picklez
from inphosemantics.model.matrix import load_matrix





# Assumes a row vector
def norm(v):
    return np.sqrt(np.dot(v,v.T)).flat[0]

def vector_cos(v,w):
    '''
    Computes the cosine of the angle between (row) vectors v and w.
    '''
    return (np.dot(v,w.T) / (norm(v) * norm(w))).flat[0]



# Sit out here for multiprocessing
def term_fn(i):
    return i, vector_cos(term_fn.v1, term_fn.matrix[i,:].todense())

def document_fn(i):
    return i, vector_cos(document_fn.v1,
                         document_fn.matrix[:,i].todense().T)



class Model(object):
    """
    """
    def __init__(self, matrix=None):

        self.matrix = matrix


    def train(self,
              corpus,
              document_type,
              stoplist=None):
        pass


    def load_matrix(self, filename):

        self.matrix = load_matrix(filename)


    def dumpz(self, filename):
        
        self.matrix.dumpz(filename, comment=time.asctime())


    def apply_stoplist(self, stoplist):

        #Filter stoplist. Zeroing the stop word row makes the
        #resulting cosine undefined; undefined cosines are later
        #filtered out.

        for i in stoplist:
            self.matrix[i,:] = np.zeros(self.matrix.shape[1])



    #TODO: These are almost the same function....
    def similar_terms(self, term, filter_nan=False):

        term_fn.v1 = self.matrix[term,:].todense()
        term_fn.matrix = self.matrix

        p = Pool()
        results = p.map(term_fn, range(self.matrix.shape[0]))
        p.close()

        # Filter out undefined results
        if filter_nan:
            results = [(t,v) for t,v in results if np.isfinite(v)]

        dtype = [('term', np.uint32), ('value', np.float)]
        # NB: numpy >= 1.4 sorts NaN to the end
        results = np.array(results, dtype=dtype)
        results.sort(order='value')
        results = results[::-1]

        return results



    def similar_documents(self, document, filter_nan=False):

        document_fn.v1 = self.matrix[:,document].todense().T
        document_fn.matrix = self.matrix

        p = Pool()
        results = p.map(document_fn, range(self.matrix.shape[1]))
        p.close()

        # Filter out undefined results
        if filter_nan:
            results = [(t,v) for t,v in results if np.isfinite(v)]

        dtype = [('term', np.uint32), ('value', np.float)]
        # NB: numpy >= 1.4 sorts NaN to the end
        results = np.array(results, dtype=dtype)
        results.sort(order='value')
        results = results[::-1]

        return results
