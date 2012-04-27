from inphosemantics.model.vectorspacemodel import VectorSpaceModel

class TermFrequencyModel(VectorSpaceModel):

    def __init__(self, corpus, corpus_param, model_param):
        
        VectorSpaceModel.__init__(self, corpus, corpus_param,
                                  'tf', model_param)




