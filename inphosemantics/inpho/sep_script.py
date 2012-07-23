from inphosemantics import corpus
from inphosemantics.corpus import tokenizer

from inphosemantics.model import tf
from inphosemantics.viewer import tfviewer



root = '/var/inphosemantics/data/sep/complete/'

plain_path = '/var/inphosemantics/data/sep/complete/corpus/plain'

corpus_filename = root + 'corpus/sep-complete-nltk.npz'

tok_name = 'articles'

stoplist_filename = '/var/inphosemantics/data/stoplists/'\
                    'stoplist-nltk-english.txt'

tf_filename = root + 'matrices/sep-complete-nltk-tf.npy'

tfidf_filename = root + 'matrices/sep-complete-nltk-tfidf.npy'




def gen_corpus():


    # Tokenization
    tokens = tokenizer.ArticlesTokenizer(plain_path)

    c = corpus.MaskedCorpus(tokens.terms,
                            tok_names=tokens.tok_names,
                            tok_data=tokens.tok_data)


    # Stoplist to mask
    with open(stoplist_filename, 'r') as f:
        
        stoplist = f.read().split('\n')

    stoplist = [word for word in stoplist if word]

    corpus.mask_from_stoplist(c, stoplist)



    c.save(corpus_filename)




def train_tf():

    c = corpus.MaskedCorpus.load(corpus_filename)

    c = c.compressed_corpus()

    m = tf.TfModel()

    m.train(c, tok_name)

    m.save(matrix_filename)



def tf_viewer():

    v = tfviewer.TfViewer(corpus_filename=corpus_filename,
                          corpus_masked=True,
                          tf_matrix_filename=tf_matrix_filename,
                          tok_name=tok_name)

    return v


def train_tfidf():

    pass
