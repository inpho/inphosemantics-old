from __future__ import division

import numpy as np

from inphosemantics import load_picklez
from inphosemantics.model.tf import TfModel, TfViewer



class TfIdfModel(TfModel):

    def idf(self, term):

        # Count the number of non-zero entries in the row and
        # scale
        return np.log(self.matrix.shape[1]
                      / self.matrix[term,:].nnz)

    
    def train(self, corpus, column_type, row_filter=None):
        """
        row_filter is ignored in training this type of model.
        """
        super(TfIdfModel, self)\
            .train(corpus, column_type, row_filter=row_filter)

        for i in xrange(self.matrix.shape[0]):
            self.matrix[i,:] *= self.idf(i)


    def idfs(self):

        results = [(i, self.idf(i))
                   for i in xrange(self.matrix.shape[0])]
        dtype = [('row', np.int32), ('value', np.float)]
        results = np.array(results, dtype=dtype)
        results.sort(order='value')
        results = results[::-1]
        
        return results



class TfIdfViewer(TfViewer):

    def __init__(self,
                 corpus=None,
                 corpus_filename=None, 
                 model=None,
                 matrix=None,
                 matrix_filename=None,
                 document_type=None,
                 stoplist=None):

        if matrix or matrix_filename:
            super(TfIdfViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          model_type=TfIdfModel,
                          matrix=matrix,
                          matrix_filename=matrix_filename,
                          document_type=document_type,
                          stoplist=stoplist)
        else:
            super(TfIdfViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          matrix_filename=matrix_filename,
                          document_type=document_type,
                          stoplist=stoplist)


    def idf(self, term):
        pass

    def idfs(self):
        pass




def test_TfIdfModel():

    corpus_filename =\
        'test-data/iep/selected/corpus/iep-selected.pickle.bz2'
    matrix_filename =\
        'test-data/iep/selected/models/iep-selected-tfidf-word-article.mtx.bz2'

    corpus = load_picklez(corpus_filename)

    model = TfIdfModel(matrix_filename, 'articles')

    model.train(corpus)

    model.dumpz()

    model = TfIdfModel(matrix_filename, 'articles')

    model.load_matrix()

    return corpus, model

