from inphosemantics.model import tfidf
from inphosemantics.viewer import tfidfviewer
from inphosemantics import corpus
from inphosemantics import model


corpus_filename = 'inphosemantics/tests/data/iep/selected/corpus/'\
                  'iep-selected-nltk-compressed.npz'

tf_matrix_filename = 'inphosemantics/tests/data/iep/selected/matrices/'\
                     'iep-selected-nltk-tf-paragraphs.npy'

tfidf_matrix_filename = 'inphosemantics/tests/data/iep/selected/matrices/'\
                        'iep-selected-nltk-tfidf-paragraphs.npy'

tok_name = 'paragraphs'



def test_TfIdfModel():

    c = corpus.Corpus.load(corpus_filename)

    m = tfidf.TfIdfModel()

    tf_matrix = model.Model.load_matrix(tf_matrix_filename)

    m.train(c, tok_name, tf_matrix=tf_matrix)

    m.save_matrix(tfidf_matrix_filename)

    m = tfidf.TfIdfModel.load_matrix(tfidf_matrix_filename)

    return c, m, tok_name



def test_TfIdfViewer():

    v = tfidfviewer.TfIdfViewer()

    v.load_corpus(corpus_filename)

    v.load_matrix(tfidf_matrix_filename)

    v.tok_name = tok_name

    return v

