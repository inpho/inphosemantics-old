import numpy as np

from inphosemantics.model import Model
from inphosemantics.model.matrix import DenseMatrix


class BeagleEnvironment(Model):

    def train(self,
              corpus,
              token_type='sentences',
              stoplist=None,
              n_columns=2048):

        shape = (len(corpus.term_types), n_columns)
        matrix = np.random.random(shape)
        self.matrix = DenseMatrix(matrix)
        
