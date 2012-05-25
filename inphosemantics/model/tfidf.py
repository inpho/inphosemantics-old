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

