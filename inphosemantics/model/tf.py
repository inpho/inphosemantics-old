import multiprocessing as mp

import numpy as np
from scipy import sparse

from inphosemantics import model



class TfModel(model.Model):
    """
    """
    def train(self, corpus, tok_name):

        super(TfModel, self).train(corpus)


        train_fn.n_rows = corpus.terms.shape[0]

        tokens = corpus.view_tokens(tok_name)

        p = mp.Pool()

        columns = p.map(train_fn, tokens, 1)

        p.close()
        
        self.matrix = sparse.hstack(columns)
        

    def cf(self, term):
        """
        """
        row = self.matrix.tocsr()[term,:]
        
        return row.sum(1)[0, 0]

    
    def cfs(self):
        """
        """
        return self.matrix.tocsr().sum(1)[0, 0]



def train_fn(token):

    shape = (train_fn.n_rows, 1)

    column = sparse.lil_matrix(shape, dtype=np.uint32)

    for term in token:
        
        if term is not np.ma.masked:
        
            column[term, 0] += 1
    
    return column






from inphosemantics import corpus
    
def test_TfModel():

    corpus_filename = 'test-data/iep/selected/corpus/'\
                      'iep-plato-freq1-nltk.npz'
    
    matrix_filename = 'test-data/iep/selected/models/'\
                      'iep-plato-tf-word-article.npz'

    tok_name = 'articles'

    c = corpus.MaskedCorpus.load(corpus_filename)

    m = TfModel()

    m.train(c, tok_name)

    m.save(matrix_filename)

    m = model.Model.load(matrix_filename)

    return c, m, tok_name
