from inphosemantics.model.tf import TermFrequency

class TfIdf(TermFrequency):

    def __init__(self, corpus, corpus_param, model_param = 'default'):
        
        TermFrequency.__init__(self, corpus, corpus_param,
                               'tfidf', model_param)

class TfIdfNormal(TfIdf):
    
    def __init__(self, corpus, corpus_param):
        
        TfIdf.__init__(self, corpus, corpus_param, 'normal')

