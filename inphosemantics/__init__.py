# To deprecate
from inphosemantics.corpus import Corpus
from inphosemantics.model.vectorspacemodel\
    import VectorSpaceModel as Model

######################################################################
# from inphosemantics.model.beagle\
#     import BeagleEnvironment, BeagleContext, BeagleOrder, BeagleComposite
# from inphosemantics.model.tf import TermFrequency
# from inphosemantics.model.tfidf import TfIdf, TfIdfNormal
# from inphosemantics.model.lsa import Lsa300


# model_data =\
#     {'corpora':
#          {'sep' :
#               {'complete':
#                    {'class': Corpus}},
#           'iep' :
#               {'complete':
#                    {'class': Corpus}}},
#      'models':
#          {'beagle':
#               {'environment':
#                    {'class': BeagleEnvironment},
#                'context':
#                    {'class': BeagleEnvironment},
#                'order':
#                    {'class': BeagleOrder},
#                'composite':
#                    {'class': BeagleComposite}},
#           'tf' :
#               {'default':
#                    {'class': TermFrequency}},
#           'tfidf':
#               {'default':
#                    {'class': TfIdf},
#                'normal':
#                    {'class': TfIdfNormal}},
#           'lsa':
#               {'default':
#                    {'class': Lsa300}}}}



# def model(corpus, corpus_param, model, model_param):
    
#     model_class = model_data['models'][model][model_param]['class']
    
#     return model_class(corpus, corpus_param)

