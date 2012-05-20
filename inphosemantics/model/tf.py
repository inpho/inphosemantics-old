import time
from multiprocessing import Pool

import numpy as np

from inphosemantics import load_picklez
from inphosemantics.model.matrix\
    import SparseMatrix, DenseMatrix, load_matrix
from inphosemantics.model.viewdata import ViewData

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


class TfModel(object):
    """
    """
    def __init__(self, matrix=None):

        self.matrix = matrix


    def train(self, corpus, document_type):

        documents = corpus.view_tokens(document_type)
        shape = (len(corpus.term_types), len(documents))

        self.matrix = SparseMatrix(shape)
        
        for j,document in enumerate(documents):
            for term in document:
                self.matrix[term,j] += 1


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


    def cf(self, term):
        pass
    
    def cfs(self):
        """
        """
        pass



class ViewTfData(ViewData):

    def __init__(self,
                 corpus=None,
                 corpus_filename=None, 
                 model=None,
                 matrix=None,
                 matrix_filename=None,
                 document_type=None,
                 stoplist=None):

        if matrix:
            super(ViewData, self)\
                .__init__(self,
                          corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          model_type=TfModel,
                          matrix=matrix,
                          matrix_filename=matrix_filename,
                          document_type=document_type,
                          stoplist=stoplist))
        else:
            super(ViewData, self)\
                .__init__(self,
                          corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          matrix_filename=matrix_filename,
                          document_type=document_type,
                          stoplist=stoplist))


    def cf(self, term):
        pass

    def cfs(self):
        pass


    
def test_TfModel():

    corpus_filename =\
        'test-data/iep/selected/corpus/iep-selected.pickle.bz2'
    matrix_filename =\
        'test-data/iep/selected/models/iep-selected-tf-word-article.mtx.bz2'
    document_type = 'articles'

    corpus = load_picklez(corpus_filename)

    model = TfModel()

    model.train(corpus, 'articles')

    model.dumpz(matrix_filename)

    model.load_matrix(matrix_filename)

    return corpus, model, document_type



###########################################################################
#                   Deprecate in favor of ViewData

# class CorpusModel(object):

#     def __init__(self, corpus=None, model=None):

#         self.corpus = corpus
#         self.model = model
        

#     def similar_terms(self, term, filter_nan=False):

#         i = self.corpus.term_types_str.index(term)
        
#         cosines = self.model.similar_terms(i, filter_nan=filter_nan)

#         return [(self.corpus.term_types_str[t], v)
#                 for t,v in cosines]


#     def similar_documents(self, documentm, filter_nan=False):

#         doc_names = self.corpus.tokens_meta['articles']
#         doc_names_alist = zip(*doc_names.iteritems())
#         doc_names_rev = dict(zip(doc_names_alist[1], doc_names_alist[0]))

#         i = doc_names_rev[document]
        
#         cosines = self.model.similar_documents(i, filter_nan=filter_nan)

#         return [(doc_names[d], v) for d,v in cosines]



# def test_CorpusModel():

#     corpus_filename =\
#         'test-data/iep/selected/corpus/iep-selected.pickle.bz2'
#     matrix_filename =\
#         'test-data/iep/selected/models/iep-selected-tf-word-article.mtx.bz2'

#     corpus = load_picklez(corpus_filename)

#     model = TfModel(matrix_filename, 'articles')

#     model.load_matrix()

#     cm = CorpusModel(corpus, model)

#     return cm
