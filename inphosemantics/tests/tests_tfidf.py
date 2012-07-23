from inphosemantics.model import tfidf
from inphosemantics.viewer import tfidfviewer
from inphosemantics import corpus
from inphosemantics import model


corpus_filename = 'inphosemantics/tests/data/iep/selected/corpus/'\
                  'iep-selected-nltk.npz'

tf_matrix_filename = 'inphosemantics/tests/data/iep/selected/matrices/'\
                     'iep-selected-tf-word-article.npy'

matrix_filename = 'inphosemantics/tests/data/iep/selected/matrices/'\
                  'iep-selected-tfidf-word-article.npy'

tok_name = 'articles'



def test_TfIdfModel():

    c = corpus.MaskedCorpus.load(corpus_filename)

    c = c.compressed_corpus()

    m = tfidf.TfIdfModel()

    tf_matrix = model.Model.load(tf_matrix_filename)

    m.train(c, tok_name, tf_matrix=tf_matrix)

    m.save(matrix_filename)

    m = tfidf.TfIdfModel.load(matrix_filename)

    return c, m, tok_name



def test_TfIdfViewer():

    v = tfidfviewer.TfIdfViewer(corpus_filename=corpus_filename,
                                corpus_masked=True,
                                matrix_filename=matrix_filename,
                                tok_name=tok_name)

    return v

