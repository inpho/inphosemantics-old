import multiprocessing as mp

import numpy as np
from scipy import sparse

from inphosemantics import model



class TfModel(model.Model):
    """
    """
    def train(self, corpus, tok_name):

        train_fn.n_rows = corpus.terms.shape[0]

        tokens = corpus.view_tokens(tok_name)

        p = mp.Pool()

        columns = p.map(train_fn, tokens, 1)

        p.close()
        
        self.matrix = sparse.hstack(columns)
        




def train_fn(token):

    shape = (train_fn.n_rows, 1)

    column = sparse.lil_matrix(shape, dtype=np.uint32)

    for term in token:
        
        if term is not np.ma.masked:
        
            column[term, 0] += 1
    
    return column
