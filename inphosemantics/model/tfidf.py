from inphosemantics.model.tf import TermFrequencyModel

class TfIdfModel(TermFrequencyModel):

    def __init__(self, corpus, corpus_param, model_param):
        
        TermFrequencyModel.__init__(self, corpus, corpus_param,
                                    'tfidf', model_param)

