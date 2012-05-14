import os.path

import numpy as np

from inphosemantics.corpus import Corpus

from inphosemantics.model.matrix\
    import SparseMatrix, DenseMatrix, load_matrix









class ModelBase(Corpus):

    def __init__(self, corpus, corpus_param, model, model_param):

        Corpus.__init__(self, corpus, corpus_param)

        self.model = model
        self.model_param = model_param

        self.model_path =\
            os.path.join(Corpus.data_root, self.corpus, self.corpus_param,\
                             self.model, self.model_param)
        




class BaseModel(object):

    def __init__(self, corpus = '', corpus_param = '', 
                 model = '', model_param = ''):

        self.corpus = corpus
        self.corpus_param = corpus_param
        self.model = model
        self.model_param = model_param
        

    def train(self, lexicon = [], corpus = [], stopwords = []):

        pass

    def lexicon(self, corpus):

        return lexicon

    def stopwords(self):
        
        pass
