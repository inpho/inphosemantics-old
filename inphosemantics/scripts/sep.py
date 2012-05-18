import os

from inphosemantics.corpus.tokenizer import SepTokens
from inphosemantics.corpus import Corpus


corpus = '/var/inphosemantics/data/sep/complete/corpus'
models = '/var/inphosemantics/data/sep/complete/models'
plain_text = 'plain'
corpus_filename = 'sep-complete.pickle.bz2'
tf_filename = 'sep-complete-tf.mtx.bz2'


def gen_corpus_file():

    print 'Tokenizing SEP'
    t = SepTokens(os.path.join(corpus, plain_text))
    
    print 'Digitizing SEP'
    c = Corpus(t.word_tokens, t.tokens_dict, t.tokens_metadata,
               os.path.join(corpus, corpus_filename))
    
    print 'Saving SEP'
    c.dumpz()


def train_tf_model():
    pass
