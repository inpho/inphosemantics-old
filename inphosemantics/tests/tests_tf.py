from inphosemantics.model import tf
from inphosemantics.viewer import tfviewer
from inphosemantics import corpus


corpus_filename = 'inphosemantics/tests/data/iep/selected/corpus/'\
                  'iep-selected-nltk-compressed.npz'

matrix_filename = 'inphosemantics/tests/data/iep/selected/matrices/'\
                  'iep-selected-nltk-tf-article.npy'

tok_name = 'articles'



def test_TfModel():

    c = corpus.Corpus.load(corpus_filename)

    m = tf.TfModel()

    m.train(c, tok_name)

    m.save_matrix(matrix_filename)

    m = tf.TfModel.load_matrix(matrix_filename)

    return c, m, tok_name



def test_TfViewer():

    v = tfviewer.TfViewer()

    v.load_corpus(corpus_filename)

    v.load_matrix(matrix_filename)

    v.tok_name = tok_name

    return v
