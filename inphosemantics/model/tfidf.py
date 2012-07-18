from __future__ import division

import multiprocessing as mp

from scipy import sparse
import numpy as np

from inphosemantics import model
from inphosemantics.model import tf



def idf(row):
    """
    Count the number of non-zero entries in the row and scale
    """
    print row.shape[1], row.nnz
    
    return np.log(row.shape[1] / row.nnz)



def train_fn(row_index):

    if row_index in train_fn.row_mask:

        return np.zeros((1, train_fn.tf_matrix.shape[1]))


    row = train_fn.tf_matrix[row_index,:]
    
    row *= idf(row)
    
    return row



class TfIdfModel(model.Model):
    """
    """
    def train(self, corpus, tok_name, tf_matrix=None):

        super(TfIdfModel, self).train(corpus)

        if tf_matrix:

            train_fn.tf_matrix = tf_matrix.tocsr().astype(np.float32)

        else:

            tf_model = tf.TfModel()
            tf_model.train(corpus, tok_name)
            
            train_fn.tf_matrix = tf_model.matrix.tocsr().astype(np.float32)

        train_fn.row_mask = self.row_mask

        rows = map(train_fn, range(train_fn.tf_matrix.shape[0]))

        self.matrix = sparse.vstack(rows)


    # def idfs(self):

    #     results = [(i, self.idf(i))
    #                for i in xrange(self.matrix.shape[0])]
    #     dtype = [('row', np.int32), ('value', np.float)]
    #     results = np.array(results, dtype=dtype)
    #     results.sort(order='value')
    #     results = results[::-1]
        
    #     return results







from inphosemantics import corpus
    
def test_TfIdfModel():

    corpus_filename = 'test-data/iep/selected/corpus/'\
                      'iep-plato-freq1-nltk.npz'
    
    matrix_filename = 'test-data/iep/selected/models/'\
                      'iep-plato-tfidf-word-article.npz'

    tok_name = 'articles'

    c = corpus.MaskedCorpus.load(corpus_filename)

    model = TfIdfModel()

    model.train(c, tok_name)

    model.save_matrix(matrix_filename)

    model.load_matrix(matrix_filename)

    return c, model, tok_name







# class TfIdfModel(Model):
#     """
#     """
#     def train(self, corpus, tok_name):

#         train_fn.n_rows = corpus.terms.size

#         tokens = corpus.view_tokens(tok_name)

#         p = mp.Pool()

#         columns = p.map(train_fn, tokens, 1)

#         p.close()
        
#         self.matrix = sparse.hstack(columns)
        

#     def cf(self, term):
#         """
#         """
#         row = self.matrix.tocsr()[term,:]
        
#         return row.sum(1)[0, 0]

    
#     def cfs(self):
#         """
#         """
#         return self.matrix.tocsr().sum(1)[0, 0]



# def train_fn(token):

#     shape = (train_fn.n_rows, 1)

#     column = sparse.lil_matrix(shape, dtype=np.uint32)

#     for term in token:
        
#         if term is not np.ma.masked:
        
#             column[term, 0] += 1
    
#     return column
