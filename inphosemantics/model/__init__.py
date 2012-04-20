import os.path

from inphosemantics.corpus import Corpus

class ModelBase(Corpus):

    def __init__(self, corpus, corpus_param, model, model_param):

        Corpus.__init__(self, corpus, corpus_param)

        self.model = model
        self.model_param = model_param

        self.model_path =\
            os.path.join(Corpus.data_root, self.corpus, self.corpus_param,
                         self.model, self.model_param)
            





