from __future__ import division

import numpy as np

from inphosemantics.model import Model
from inphosemantics.model.tf import TfModel


class TfIdfModel(Model):
    
    def train(self,
              corpus,
              token_type,
              stoplist=None,
              tf_matrix=None):
        """
        stoplist is ignored in training this type of model.
        """
        if tf_matrix:
            self.matrix = tf_matrix
        else:
            tf_model = TfModel()
            tf_model.train(corpus, token_type, stoplist)
            self.matrix = tf_model.matrix

        for i in xrange(self.matrix.shape[0]):
            self.matrix[i,:] *= self.idf(i)

    def idf(self, term):

        # Count the number of non-zero entries in the row and
        # scale
        return np.log(self.matrix.shape[1]
                      / self.matrix[term,:].nnz)

    def idfs(self):

        results = [(i, self.idf(i))
                   for i in xrange(self.matrix.shape[0])]
        dtype = [('row', np.int32), ('value', np.float)]
        results = np.array(results, dtype=dtype)
        results.sort(order='value')
        results = results[::-1]
        
        return results







def test_TfIdfModel():

    from inphosemantics import load_picklez

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

