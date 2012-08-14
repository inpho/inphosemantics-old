import numpy as np
from scipy import sparse
# from scipy.sparse import linalg

import sparsesvd as ssvd

from inphosemantics import model
from inphosemantics.model import tfidf

# TODO: train needs either 1) a matrix or 2) corpus and tok_name;
# provide feedback to this effect

# TODO: tf-idf matrices are only one possible input matrix. Generalize
# accordingly.

# TODO: the reduction function should be a parameter.

class LsaModel(model.Model):
    """
    """
    def train(self, corpus=None, tok_name=None, tfidf_matrix=None):

        if not tfidf_matrix:

            tfidf_model = tfidf.TfIdfModel()
            tfidf_model.train(corpus, tok_name)
            
            tfidf_matrix = tfidf_model.matrix


        tfidf_matrix = sparse.csc_matrix(tfidf_matrix, dtype=np.float64)


        self.tfidf_matrix = tfidf_matrix

        print 'Performing SVD'
        
        u, s, v = ssvd.sparsesvd(tfidf_matrix, tfidf_matrix.shape[1])

        print 'Reducing eigenvalue matrix'

        # Reduction: the largest eigenvalues until their sum exceeds
        # half the sum of all the eigenvalues
        rs = np.zeros((s.size,))

        acc = 0

        i = 0

        share = .5 * np.sum(s)

        while acc < share:

            rs[i] = s[i]

            acc += s[i]
            
            i += 1

        print 'Retaining', i, 'eigenvalues out of', s.size

        print 'Generating sparse matrices for u, s, v'

        sp_u = sparse.bsr_matrix(u.T)

        # sp_rs = sparse.spdiags(rs, 0, rs.size, rs.size, format='bsr')

        sp_rs = sparse.spdiags(s, 0, rs.size, rs.size, format='bsr')
        
        sp_v = sparse.bsr_matrix(v)

        print 'Reconstructing term-document matrix'

        self.matrix = sp_u * (sp_rs * sp_v)
