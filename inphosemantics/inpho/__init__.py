import os
import copy


# TODO: This import situation is out of control. Set __all__
# parameters in the submodules and use from <package> import <module>
# pattern.

from inphosemantics import *

import inphosemantics

import inphosemantics.util

from inphosemantics.corpus import Corpus

from inphosemantics.corpus.tokenizer import ArticlesTokenizer

from inphosemantics.model.tf import TfModel
from inphosemantics.viewer.tfviewer import TfViewer

from inphosemantics.model.tfidf import TfIdfModel
from inphosemantics.viewer.tfidfviewer import TfIdfViewer

from inphosemantics.model.beagleenvironment import BeagleEnvironment
from inphosemantics.viewer.beagleenvironmentviewer\
     import BeagleEnvironmentViewer

from inphosemantics.model.beaglecontext import BeagleContext
from inphosemantics.viewer.beaglecontextviewer\
     import BeagleContextViewer

from inphosemantics.model.beagleorder import BeagleOrder
from inphosemantics.viewer.beagleorderviewer\
     import BeagleOrderViewer

from inphosemantics.model.beaglecomposite import BeagleComposite
from inphosemantics.viewer.beaglecompositeviewer\
     import BeagleCompositeViewer


root = '/var/inphosemantics/data'


stoplist_dict = dict(\
    nltk='stoplists/stoplist-nltk-english.txt',
    jones='stoplists/stoplist-jones-beagle.txt',
    inpho_beagle_supp='stoplists/stoplist-inpho-beagle-supplementary.txt')


def dump_nltk_stoplist():

    from nltk.corpus import stopwords as nltk_stopwords

    stoplist = nltk_stopwords.words('english')

    filename = os.path.join(root, stoplist_dict['nltk'])
    with open(filename, 'w') as f:
        for word in stoplist:
            print >>f, word


def load_stoplist(name):

    filename = os.path.join(root, stoplist_dict[name])
    
    with open(filename, 'r') as f:
        words = f.read().split('\n')

    words = [word for word in words if word]

    return words


def merge_stoplists(stoplists):

    merged = []
    for stoplist in stoplists:
        merged.extend(load_stoplist(stoplist))

    merged = list(set(merged))

    return merged


# TODO: Load model_dict from a JSON file

model_dict = dict()

model_dict['tf'] = dict()
model_dict['tf']['model_type'] = TfModel
model_dict['tf']['viewer_type'] = TfViewer
model_dict['tf']['stoplist'] = ['nltk']
model_dict['tf']['token_type'] = 'articles'

model_dict['tfidf'] = dict()
model_dict['tfidf']['model_type'] = TfIdfModel
model_dict['tfidf']['viewer_type'] = TfIdfViewer
model_dict['tfidf']['stoplist'] = ['nltk']
model_dict['tfidf']['token_type'] = 'articles'
model_dict['tfidf']['tf_matrix'] = 'tf'

model_dict['beagleenvironment'] = dict()
model_dict['beagleenvironment']['model_type'] = BeagleEnvironment
model_dict['beagleenvironment']['viewer_type'] = BeagleEnvironmentViewer
model_dict['beagleenvironment']['n_columns'] = 2048

model_dict['beaglecontext'] = dict()
model_dict['beaglecontext']['model_type'] = BeagleContext
model_dict['beaglecontext']['viewer_type'] = BeagleContextViewer
model_dict['beaglecontext']['stoplist'] = ['nltk','jones','inpho_beagle_supp']
model_dict['beaglecontext']['token_type'] = 'sentences'
model_dict['beaglecontext']['env_matrix'] = 'beagleenvironment'

model_dict['beagleorder'] = dict()
model_dict['beagleorder']['model_type'] = BeagleOrder
model_dict['beagleorder']['viewer_type'] = BeagleOrderViewer
model_dict['beagleorder']['stoplist'] = ['nltk','jones','inpho_beagle_supp']
model_dict['beagleorder']['token_type'] = 'sentences'
model_dict['beagleorder']['env_matrix'] = 'beagleenvironment'

model_dict['beaglecomposite'] = dict()
model_dict['beaglecomposite']['model_type'] = BeagleComposite
model_dict['beaglecomposite']['viewer_type'] = BeagleCompositeViewer
model_dict['beaglecomposite']['stoplist'] = ['nltk','jones','inpho_beagle_supp']
model_dict['beaglecomposite']['token_type'] = 'sentences'
model_dict['beaglecomposite']['ctx_matrix'] = 'beaglecontext'
model_dict['beaglecomposite']['ord_matrix'] = 'beagleorder'




            

#TODO: If paths do not exist, create them.

#TODO: Get permission to overwrite data



def _values(d):
    """
    Takes a dictionary, sorts the entries by keyword and returns a
    list of values.
    """
    alist = list(d.iteritems())

    alist.sort(key=lambda p: p[0])

    values = zip(*alist)[1]

    return values


def get_model_params(name):
    """
    Takes a top-level key from model_dict and returns a dictionary of
    just the model parameters.
    """
    params = copy.deepcopy(model_dict[name])
    del params['model_type']
    del params['viewer_type']
    return params



def get_Word2Word_csv(corpus, corpusParam, model, phrase, matrixWidth):
    """
    Returns a matrix in CSV format suitable for Word2Word.
    corpus, corpusParam, model, & phrase as literals, matrixWidth
    as an integer.
    """
    # Grab a viewer
    viewer = InphoViewer(corpus, corpusParam, model)

    # Figure out which terms will appear in the matrix
    terms = zip(*viewer.similar_terms(phrase, True)[:matrixWidth])[0]

    # Create said matrix
    similarityMatrix = viewer.simmat_terms(terms)

    # Export the data!
    return inphosemantics.util.gen_word2word(similarityMatrix.matrix, similarityMatrix.indices)
    


def gen_corpus_filename(corpus_name,
                        corpus_param,
                        term_types_only=False):

    if term_types_only:
        corpus_filename =\
            corpus_name + '-' + corpus_param + '-term-types.pickle.bz2'

    else:
        corpus_filename =\
            corpus_name + '-' + corpus_param + '.pickle.bz2'

    corpus_filename =\
        os.path.join(root,
                     corpus_name,
                     corpus_param,
                     'corpus',
                     corpus_filename)

    return corpus_filename


def gen_matrix_filename(corpus_name,
                        corpus_param,
                        model_name,
                        model_params):

    # Changes to model_params are only for the sake of generating the
    # filename
    model_params = copy.deepcopy(model_params)

    # Omitting stoplist names from filename
    if 'stoplist' in model_params:
        del model_params['stoplist']

    # Omitting input matrix names from filename
    for param in model_params.keys():
        if param.endswith('matrix'):
            del model_params[param]


    model_params_str =\
        '-'.join([str(x) for x in _values(model_params)])

    matrix_filename =\
        '-'.join([corpus_name,
                  corpus_param,
                  model_name,
                  model_params_str]) + '.npy'
    matrix_filename =\
        os.path.join(root,
                     corpus_name,
                     corpus_param,
                     'matrices',
                     matrix_filename)        

    return matrix_filename



class InphoViewer(object):

    def __new__(cls,
                corpus_name,
                corpus_param,
                model_name,
                term_types_only=False,
                **_model_params):


        viewer_type = model_dict[model_name]['viewer_type']

        model_params = get_model_params(model_name)
        model_params.update(_model_params)


        viewer_params = dict()
        
        viewer_params['corpus_filename'] =\
            gen_corpus_filename(corpus_name,
                                corpus_param,
                                term_types_only=term_types_only)


        viewer_params['matrix_filename'] =\
            gen_matrix_filename(corpus_name,
                                corpus_param,
                                model_name,
                                model_params)


        if 'token_type' in model_params:
            viewer_params['token_type'] = model_params['token_type']


        if 'stoplist' in model_params:

            stoplist = merge_stoplists(model_params['stoplist'])

            viewer_params['stoplist'] = stoplist


        viewer = viewer_type(**viewer_params)

        viewer.corpus_name = corpus_name
        viewer.corpus_param = corpus_param
        viewer.model_name = model_name
        viewer.term_types_only = term_types_only

        return viewer





class InphoTrainer(object):

    def __init__(self,
                 corpus_name,
                 corpus_param,
                 model_name,
                 **model_params):


        self.model_type = model_dict[model_name]['model_type']


        self.model_params = get_model_params(model_name)
        self.model_params.update(model_params)



        corpus_filename =\
            gen_corpus_filename(corpus_name, corpus_param)

        print 'Loading corpus from\n'\
              '  ', corpus_filename
        
        self.corpus = load_picklez(corpus_filename)



        self.matrix_filename =\
            gen_matrix_filename(corpus_name,
                                corpus_param,
                                model_name,
                                self.model_params)            



        if 'stoplist' in self.model_params:

            stoplist = merge_stoplists(self.model_params['stoplist'])

            print 'Encoding stoplist'
            stoplist = self.corpus.encode_tokens_str(stoplist)

            self.model_params['stoplist'] = stoplist



        for param in self.model_params:
            
            if param.endswith('matrix'):

                model_name = self.model_params[param]
                model_params = get_model_params(model_name)

                filename = gen_matrix_filename(corpus_name,
                                               corpus_param,
                                               model_name,
                                               model_params)             

                print 'Loading matrix\n'\
                      '  ', filename

                self.model_params[param] = load_matrix(filename)


    def train(self):

        print 'Training model of type', self.model_type.__name__
        model = self.model_type()

        model.train(self.corpus, **self.model_params)

        print 'Writing matrix to\n'\
              '  ', self.matrix_filename
        model.dump_matrix(self.matrix_filename)




tokenizer_dict = {
    'iep': ArticlesTokenizer,
    'sep': ArticlesTokenizer,
    'sep-iep': ArticlesTokenizer,
    'malaria': ArticlesTokenizer,
    'philpapers': ArticlesTokenizer
}




class InphoTokenizer(object):

    def __init__(self, corpus_name, corpus_param):

        self.plain_path =\
            os.path.join(root,
                         corpus_name,
                         corpus_param,
                         'corpus/plain/')
        
        self.corpus_filename =\
            gen_corpus_filename(corpus_name, corpus_param)

        self.term_types_filename =\
            gen_corpus_filename(corpus_name,
                                corpus_param,
                                term_types_only=True)


        try:
            self.tokenizer_type = tokenizer_dict[corpus_name]

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


    def gen_corpus(self):

        if not self.tokens:
            self.tokenize()

        corpus = Corpus(self.tokens.word_tokens,
                        self.tokens.tokens_dict,
                        self.tokens.tokens_metadata)

        print 'Writing corpus to\n'\
              '  ', self.corpus_filename

        corpus.dumpz(self.corpus_filename)

        print 'Writing term types to\n'\
              '  ', self.term_types_filename

        corpus.dumpz(self.term_types_filename,
                     term_types_only=True)
