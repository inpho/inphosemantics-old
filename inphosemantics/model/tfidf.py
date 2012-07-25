import multiprocessing as mp

from scipy import sparse
import numpy as np

from inphosemantics import model
from inphosemantics.model import tf


"""
A term may occur in every document. Then the idf of that term will be
0; so the tfidf will also be zero.

A term may occur in no documents at all. This typically happens only
when that term has been masked. In that case the idf of that term is
undefined (division by zero). So also the tfidf of that term will be
undefined.
"""


def idf(row):
    """
    Count the number of non-zero entries in the row and scale
    """
    return np.log(np.float32(row.shape[1]) / row.nnz)



def train_fn(row_index):

    print 'Computing row', row_index

    row = train_fn.tf_matrix[row_index,:].astype('float32')
    
    return row * idf(row)



class TfIdfModel(model.Model):
    """
    """
    def train(self, corpus, tok_name, tf_matrix=None):

        if tf_matrix:

            train_fn.tf_matrix = tf_matrix.tocsr()

        else:

            tf_model = tf.TfModel()
            tf_model.train(corpus, tok_name)
            
            train_fn.tf_matrix = tf_model.matrix.tocsr()


        del tf_matrix


        # Suppress division by zero errors
        old_settings = np.seterr(divide='ignore')

        # Single-processor map for debugging
        # rows = map(train_fn, range(train_fn.tf_matrix.shape[0]))


        p = mp.Pool()

        rows = p.map(train_fn, range(train_fn.tf_matrix.shape[0]), 1000)

        p.close()


        # Restore default handling of floating-point errors
        np.seterr(**old_settings)

        print 'Updating data matrix'

        self.matrix = sparse.vstack(rows)
