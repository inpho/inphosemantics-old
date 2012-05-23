import os

from nltk.corpus import stopwords as nltk_stopwords

from inphosemantics import *
from inphosemantics.model.tf import TfModel, TfViewer
from inphosemantics.model.tfidf import TfIdfModel, TfIdfViewer
from inphosemantics.corpus.tokenizer import IepTokens, SepTokens

root = '/var/inphosemantics/data'


#TODO: These dictionaries could be consolidated. As well, additional
#data might be included.

viewer_dict = dict(tf = TfViewer,
                   tfidf = TfIdfViewer,
                   # beagle-environment = BeagleEnvironmentViewer,
                   # beagle-context = BeagleContextViewer,
                   # beagle-order = BeagleOrderViewer,
                   # beagle-composite = BeagleCompositeViewer
                   )


model_dict = dict(tf = TfModel,
                  tfidf = TfIdfModel,
                  # beagle-environment = BeagleEnvironmentModel,
                  # beagle-context = BeagleContextModel,
                  # beagle-order = BeagleOrderModel,
                  # beagle-composite = BeagleCompositeModel
                  )


#TODO: If paths do not exist, create them.

#TODO: Get permission to overwrite data


def gen_corpus_filename(corpus,
                        corpus_param):

    corpus_filename =\
        corpus + '-' + corpus_param + '.pickle.bz2'
    corpus_filename =\
        os.path.join(root,
                     corpus,
                     corpus_param,
                     'corpus',
                     corpus_filename)

    return corpus_filename


def gen_matrix_filename(corpus,
                        corpus_param,
                        model,
                        model_param):

    matrix_filename =\
        '-'.join([corpus,
                  corpus_param,
                  model,
                  model_param]) + '.mtx.bz2'
    matrix_filename =\
        os.path.join(root,
                     corpus,
                     corpus_param,
                     'matrices',
                     matrix_filename)        

    return matrix_filename



class InphoViewer(object):

    def __new__(cls,
                corpus,
                corpus_param,
                model,
                model_param,
                stoplist=None):

        
        corpus_filename =\
            gen_corpus_filename(corpus, corpus_param)


        matrix_filename =\
            gen_matrix_filename(corpus,
                                corpus_param,
                                model,
                                model_param)


        #determine a model type from model                          
        try:
            viewer_type = viewer_dict[model]

        except KeyError:
            print '*********************************\n'\
                  '* Model type was not recognized *\n'\
                  '*********************************\n\n'\
                  'Available model types:\n'\
                  '   ', ', '.join(viewer_dict.keys())
            
            raise
            

        #determine a document type from model_param
        document_type = model_param.split('-')[0]


        #TODO: Robust handling of various stoplists
        if not stoplist:
            stoplist = nltk_stopwords.words('english')


        return viewer_type(corpus_filename=corpus_filename,
                            matrix_filename=matrix_filename,
                            document_type=document_type,
                            stoplist=stoplist)



class InphoTrainer(object):

    def __init__(self,
                 corpus,
                 corpus_param,
                 model,
                 model_param,
                 stoplist=None):

        self.corpus_filename =\
            gen_corpus_filename(corpus, corpus_param)


        self.matrix_filename =\
            gen_matrix_filename(corpus,
                                corpus_param,
                                model,
                                model_param)            


        #determine a model type from model                          
        try:
            self.model_type = model_dict[model]

        except KeyError:
            print '*********************************\n'\
                  '* Model type was not recognized *\n'\
                  '*********************************\n\n'\
                  'Available model types:\n'\
                  '   ', ', '.join(model_dict.keys())
            
            raise
            

        #determine a document type from model_param
        self.document_type = model_param.split('-')[0]


        #TODO: Robust handling of various stoplists
        if not stoplist:
            self.stoplist = nltk_stopwords.words('english')


    def train(self):

        print 'Loading corpus from\n'\
              '  ', self.corpus_filename
        corpus = load_picklez(self.corpus_filename)


        print 'Training model of type', self.model_type.__name__
        model = self.model_type()

        model.train(corpus, self.document_type, self.stoplist)

        print 'Writing matrix to\n'\
              '  ', self.matrix_filename
        model.dumpz(self.matrix_filename)





tokenizer_dict = dict(iep=IepTokens,
                      sep=SepTokens)




class InphoTokenizer(object):

    def __init__(self, corpus, corpus_param):

        self.plain_path =\
            os.path.join(root,
                         corpus,
                         corpus_param,
                         'corpus/plain/')
        
        self.corpus_filename =\
            gen_corpus_filename(corpus, corpus_param)

        try:
            self.tokenizer_type = tokenizer_dict[corpus]

        except KeyError:
            print '*************************************\n'\
                  '* Tokenizer type was not recognized *\n'\
                  '*************************************\n\n'\
                  'Available tokenizer types:\n'\
                  '   ', ', '.join(tokenizer_dict.keys)
            
            raise


        self.tokens = None


    def tokenize(self):

        self.tokens = self.tokenizer_type(self.plain_path)


    def generate_corpus(self):

        if not self.tokens:
            self.tokenize()

        else:
            corpus = Corpus(self.tokens.word_tokens,
                            self.tokens.tokens_dict,
                            self.tokens.tokens_metadata)

            print 'Writing corpus to'\
                  '  ', self.corpus_filename
            
            corpus.dumpz(self.corpus_filename)
