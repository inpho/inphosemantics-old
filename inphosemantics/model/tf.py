import numpy as np

from inphosemantics.model.matrix\
    import SparseMatrix, DenseMatrix, load_matrix


# Assumes a row vector
def norm(v):
    return np.sqrt(np.dot(v,v.T)).flat[0]

def vector_cos(v,w):
    '''
    Computes the cosine of the angle between (row) vectors v and w.
    '''
    return (np.dot(v,w.T) / (norm(v) * norm(w))).flat[0]


# Sits out here for multiprocessing
def cosine_fn(v2):
    return vector_cos(cosine_fn.v1, v2)


class TFModel(object):
    """
    Takes an IntegerCorpus with a 'documents' partitioning
    """
    def __init__(self, corpus):

        self.terms = corpus
        self.documents = corpus.view_partition('documents')
        
        shape = (len(self.terms), len(self.documents))
        self.td_matrix = SparseMatrix(shape)


    def train(self):
        
        for j,document in enumerate(self.documents):
            for term in document:
                self.td_matrix[term,j] += 1


    #TODO: These are almost the same function....
    def similar_terms(self, term):

        cosine_fn.v1 = self.td_matrix[term,:].todense()

        results = [(t2, cosine_fn(self.td_matrix[t2,:].todense())) 
                   for t2 in xrange(self.td_matrix.shape[0])]

        results.sort(key=lambda p: p[1], reverse = True)

        return results


    def similar_documents(self, document):

        cosine_fn.v1 = self.td_matrix[:,document].todense().T

        results = [(t2, cosine_fn(self.td_matrix[:,t2].todense().T)) 
                   for t2 in xrange(self.td_matrix.shape[1])]

        results.sort(key=lambda p: p[1], reverse = True)

        return results



class TFIDFModel(TFModel):
    
    def train(self):
        
        super(TFIDFModel, self).train()

        for i in xrange(self.td_matrix.shape[0]):

            # Count the number of non-zero entries in the row and
            # scale
            df = np.log(self.td_matrix.shape[1] / self.td_matrix[i,:].nnz)

            self.td_matrix[i,:] /= df



