from inphosemantics.model import ModelBase


class SparseVector(object):
    
    def __init__(self):
        pass


class TermFrequency(ModelBase):

    def __init__(self, corpus, corpus_param, model_param = 'default'):
        
        ModelBase.__init__(self, corpus, corpus_param,
                                  'tf', model_param)

        self.stored_tt_vectors = None
        self.stored_td_vectors = None
        self.stored_dd_vectors = None


    def write_tt_vector(self, vector, index):
        pass

    def write_td_vector(self, vector, index):
        pass

    def write_dd_vector(self, vector, index):
        pass

    def tt_vector(self, index):
        pass

    def td_vector(self, index):
        pass

    def dd_vector(self, index):
        pass

    def tt_vectors(self):
        pass

    def td_vectors(self):
        pass

    def dd_vectors(self):
        pass
    
    #Compute cosine with sparse vectors as input
    
