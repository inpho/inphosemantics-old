import numpy as np

from scipy import sparse
from scipy.sparse import linalg as linalgs

from inphosemantics import model
from inphosemantics.model import tfidf

# TODO: train needs either 1) a matrix or 2) corpus and tok_name;
# provide feedback to this effect

class LsaModel(model.Model):
    """
    """
    def train(self,
              corpus=None,
              tok_name=None,
              td_matrix=None,
              k_factors=300):

        if td_matrix is None:

            tfidf_model = tfidf.TfIdfModel()

            tfidf_model.train(corpus, tok_name)
            
            td_matrix = tfidf_model.matrix

            del tfidf_model

        if td_matrix.shape[0] < k_factors:

            k_factors = td_matrix.shape[0] - 1
            
        if td_matrix.shape[1] < k_factors:

            k_factors = td_matrix.shape[1] - 1

        td_matrix = sparse.csc_matrix(td_matrix, dtype=np.float64)

        print 'Performing sparse SVD'

        u, s, v = linalgs.svds(td_matrix, k=k_factors)

        self.matrix = np.float32(u), np.float32(s), np.float32(v)

        

    @property
    def term_matrix(self):

        return self.matrix[0]



    @property
    def eigenvalues(self):

        return self.matrix[1]



    @property
    def doc_matrix(self):

        return self.matrix[2]
        


    def save_matrix(self, file):

        print 'Not yet implemented'
