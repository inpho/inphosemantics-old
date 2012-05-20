import os

import numpy as np
from nltk.corpus import stopwords as nltk_stopwords

from inphosemantics import *
from inphosemantics.corpus.tokenizer import IepTokens
from inphosemantics.corpus import Corpus
from inphosemantics.model.tf import TfModel, ViewTfData
from inphosemantics.model.tfidf import TfIdfModel, ViewTfIdfData



corpus = '/var/inphosemantics/data/iep/complete/corpus'
models = '/var/inphosemantics/data/iep/complete/models'
plain = 'plain'
complete = 'iep-complete.pickle.bz2'
tf_word_article = 'iep-complete-tf-word-article.mtx.bz2'
tfidf_word_article = 'iep-complete-tfidf-word-article.mtx.bz2'


# TODO: Replace CorpusModel with ViewData subclasses

def gen_corpus_file():

    print 'Tokenizing IEP'
    t = IepTokens(os.path.join(corpus, plain))
    
    print 'Digitizing IEP'
    c = Corpus(t.word_tokens, t.tokens_dict, t.tokens_metadata,
               os.path.join(corpus, complete))
    
    print 'Saving IEP'
    c.dumpz()


# def digitize_stoplist(c):

#     stoplist = nltk_stopwords.words('english')

#     decoder = dict(zip(c.term_types_str, xrange(len(c.term_types))))
    
#     out = c.decode(stoplist, decoder)
#     # It's possible that the stop list and the corpus term types do
#     # not completely overlap
#     out = [i for i in out if np.isfinite(i)]

#     return out


def train_tf_model():

    print 'Loading corpus'
    c = load_picklez(os.path.join(corpus, complete))

    document_type = 'articles'

    print 'Training term frequency model'
    m = TfModel(os.path.join(models, tf_word_article), document_type)
    m.train(c)
    
    print 'Saving matrix'
    m.dumpz()


def train_tfidf_model():

    print 'Loading corpus'
    c = load_picklez(os.path.join(corpus, complete))

    document_type = 'articles'

    print 'Training TF-IDF model'
    m = TfIdfModel(os.path.join(models, tfidf_word_article), document_type)
    m.train(c)
    
    print 'Saving matrix'
    m.dumpz()


def tf_viewer():

    corpus_filename = os.path.join(corpus, complete)
    model_filename = os.path.join(models, tf_word_article)

    stoplist = nltk_stopwords.words('english')
    
    v = ViewTfData(corpus_filename=corpus_filename,
                   matrix_filename=matrix_filename,
                   document_type='articles', stoplist=stoplist)

    return v



# def tf_viewer():

#     print 'Loading corpus'
#     c = load_picklez(os.path.join(corpus, complete))
#     s = digitize_stoplist(c)

#     document_type = 'articles'

#     print 'Loading term frequency model'
#     m = TfModel(os.path.join(models, tf_word_article), document_type)
#     m.load_matrix()
#     m.apply_stoplist(s)

#     return CorpusModel(c, m)



def tfidf_viewer():

    print 'Loading corpus'
    c = load_picklez(os.path.join(corpus, complete))
    s = digitize_stoplist(c)

    document_type = 'articles'

    print 'Loading term frequency model'
    m = TfIdfModel(os.path.join(models, tfidf_word_article), document_type)
    m.load_matrix()
    m.apply_stoplist(s)

    return CorpusModel(c, m)
