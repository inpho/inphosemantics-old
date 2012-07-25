import multiprocessing as mp

import numpy as np
from scipy import sparse

from inphosemantics import model



class TfModel(model.Model):
    """
    """
    def train(self, corpus, tok_name):

        print 'Viewing tokens:', tok_name

        tokens = corpus.view_tokens(tok_name)

        p = mp.Pool()

        columns = p.map(train_fn, tokens, 1)

        p.close()

        # non-parallel map for debugging
        # columns = map(train_fn, tokens)

        print 'Updating data matrix'

        shape = (corpus.terms.shape[0], len(tokens))

        self.matrix = sparse.lil_matrix(shape)

        for j, column in enumerate(columns):

            for i, val in column.iteritems():

                self.matrix[i, j] = val
        




def train_fn(token):

    column = dict()

    print 'Training on token', token

    for term in token:

        if term in column:

            column[term] += 1

        else:

            column[term] = 1

    return column
