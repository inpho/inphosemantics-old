from __future__ import division

import time
from multiprocessing import Pool

import numpy as np

from inphosemantics.model.matrix\
    import SparseMatrix, DenseMatrix, load_matrix
from inphosemantics import load_picklez


# TODO: Replace CorpusModel with a system of classes that gives a
# generic interface to viewer classes specific to each model class



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
    return i, vector_cos(term_fn.v1, term_fn.td_matrix[i,:].todense())

def document_fn(i):
    return i, vector_cos(document_fn.v1,
                         document_fn.td_matrix[:,i].todense().T)


class TFModel(object):
    """
    """
    def __init__(self, matrix_filename=None, document_type=None):

        self.matrix_filename = matrix_filename
        self.document_type = document_type


    def train(self, corpus):

        if self.document_type:
            documents = corpus.view_tokens(self.document_type)
        else:
            documents = corpus.view_tokens('documents')
        
        shape = (len(corpus.term_types), len(documents))

        self.td_matrix =\
            SparseMatrix(shape, filename=self.matrix_filename)

        
        for j,document in enumerate(documents):
            for term in document:
                self.td_matrix[term,j] += 1


    def load_matrix(self):

        self.td_matrix = load_matrix(self.matrix_filename)


    def dumpz(self):
        
        self.td_matrix.dumpz(comment=time.asctime())


    def apply_stoplist(self, stoplist):

        #Filter stoplist. Zeroing the stop word row makes the
        #resulting cosine undefined; undefined cosines are later
        #filtered out.

        for i in stoplist:
            self.td_matrix[i,:] = np.zeros(self.td_matrix.shape[1])



    #TODO: These are almost the same function....
    def similar_terms(self, term, filter_nan=False):

        term_fn.v1 = self.td_matrix[term,:].todense()
        term_fn.td_matrix = self.td_matrix

        p = Pool()
        results = p.map(term_fn, range(self.td_matrix.shape[0]))
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



    def similar_documents(self, document, stoplist=None, filter_nan=False):

        document_fn.v1 = self.td_matrix[:,document].todense().T
        document_fn.td_matrix = self.td_matrix

        p = Pool()
        results = p.map(document_fn, range(self.td_matrix.shape[1]))
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

    
    def cfs(self):
        """
        """
        pass

    
class TFIDFModel(TFModel):

    def idf(self, term):

        # Count the number of non-zero entries in the row and
        # scale
        return np.log(self.td_matrix.shape[1]
                      / self.td_matrix[term,:].nnz)

    
    def train(self, corpus):
        
        super(TFIDFModel, self).train(corpus)

        for i in xrange(self.td_matrix.shape[0]):
            self.td_matrix[i,:] *= self.idf(i)


    def idfs(self):

        results = [(i, self.idf(i))
                   for i in xrange(self.td_matrix.shape[0])]
        dtype = [('t', np.int32), ('v', np.float)]
        results = np.array(results, dtype=dtype)
        results.sort(order='v')
        results = results[::-1]
        
        return results



class CorpusModel(object):

    def __init__(self, corpus=None, model=None):

        self.corpus = corpus
        self.model = model
        

    def similar_terms(self, term, filter_nan=False):

        i = self.corpus.term_types_str.index(term)
        
        cosines = self.model.similar_terms(i, filter_nan=filter_nan)

        return [(self.corpus.term_types_str[t], v)
                for t,v in cosines]


    def similar_documents(self, documentm, filter_nan=False):

        doc_names = self.corpus.tokens_meta['articles']
        doc_names_alist = zip(*doc_names.iteritems())
        doc_names_rev = dict(zip(doc_names_alist[1], doc_names_alist[0]))

        i = doc_names_rev[document]
        
        cosines = self.model.similar_documents(i, filter_nan=filter_nan)

        return [(doc_names[d], v) for d,v in cosines]




def test_TFModel():

    corpus_filename =\
        'test-data/iep/selected/corpus/iep-selected.pickle.bz2'
    matrix_filename =\
        'test-data/iep/selected/models/iep-selected-tf-word-article.mtx.bz2'

    corpus = load_picklez(corpus_filename)

    model = TFModel(matrix_filename, 'articles')

    model.train(corpus)

    model.dumpz()

    model = TFModel(matrix_filename, 'articles')

    model.load_matrix()

    return corpus, model


def test_TFIDFModel():

    corpus_filename =\
        'test-data/iep/selected/corpus/iep-selected.pickle.bz2'
    matrix_filename =\
        'test-data/iep/selected/models/iep-selected-tfidf-word-article.mtx.bz2'

    corpus = load_picklez(corpus_filename)

    model = TFIDFModel(matrix_filename, 'articles')

    model.train(corpus)

    model.dumpz()

    model = TFIDFModel(matrix_filename, 'articles')

    model.load_matrix()

    return corpus, model



def test_CorpusModel():

    corpus_filename =\
        'test-data/iep/selected/corpus/iep-selected.pickle.bz2'
    matrix_filename =\
        'test-data/iep/selected/models/iep-selected-tf-word-article.mtx.bz2'

    corpus = load_picklez(corpus_filename)

    model = TFModel(matrix_filename, 'articles')

    model.load_matrix()

    cm = CorpusModel(corpus, model)

    return cm
