from __future__ import division

import numpy as np

from inphosemantics import load_picklez
from inphosemantics.model.tf import TFModel



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

