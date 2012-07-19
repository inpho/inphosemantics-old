from inphosemantics.model import tf
from inphosemantics.viewer import tfviewer
from inphosemantics import corpus


corpus_filename = 'inphosemantics/tests/data/iep/selected/corpus/'\
                  'iep-selected-nltk.npz'

matrix_filename = 'inphosemantics/tests/data/iep/selected/matrices/'\
                  'iep-selected-tf-word-article.npy'

tok_name = 'articles'



def test_TfModel():

    c = corpus.MaskedCorpus.load(corpus_filename)

    c = c.compressed_corpus()

    m = tf.TfModel()

    m.train(c, tok_name)

    m.save(matrix_filename)

    m = tf.TfModel.load(matrix_filename)

    return c, m, tok_name



def test_TfViewer():

    v = tfviewer.TfViewer(corpus_filename=corpus_filename,
                          corpus_masked=True,
                          matrix_filename=matrix_filename,
                          tok_name=tok_name)

    return v
