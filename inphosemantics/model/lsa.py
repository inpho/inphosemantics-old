import numpy as np
from scipy import sparse
from scipy.sparse import linalg

# import sparsesvd as ssvd

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

        u, s, v = linalg.svds(tfidf_matrix, k=300)
        
        u = np.float16(u)
        s = np.float16(np.diag(s))
        v = np.float16(v)

        # print 'Reducing eigenvalue matrix'

        # Reduction: the largest eigenvalues until their sum exceeds
        # half the sum of all the eigenvalues

        # rs = []

        # acc = 0

        # i = 0

        # share = .5 * np.sum(s)

        # while acc < share:

        #     rs.append(s[i])

        #     acc += s[i]
            
        #     i += 1

        # print 'Retaining', i, 'eigenvalues out of', s.size

        # rs = np.diag(s)

        # ru = u[:,:rs.shape[0]]

        # rv = v[:rs.shape[1],:]

        print 'Reconstructing term-document matrix'

        self.matrix = np.dot(u, np.dot(s, v))
