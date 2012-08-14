from inphosemantics import corpus

from inphosemantics import model
from inphosemantics.model import lsa

from inphosemantics.viewer import lsaviewer




corpus_filename = 'inphosemantics/tests/data/iep/selected/corpus/'\
                  'iep-selected-nltk-compressed.npz'

tfidf_matrix_filename = 'inphosemantics/tests/data/iep/selected/matrices/'\
                        'iep-selected-nltk-tfidf-paragraphs.npy'

lsa_matrix_filename = 'inphosemantics/tests/data/iep/selected/matrices/'\
                      'iep-selected-nltk-lsa-tfidf-paragraphs.npy'

tok_name = 'paragraphs'



def test_LsaModel():

    m = lsa.LsaModel()

    tfidf_matrix = model.Model.load_matrix(tfidf_matrix_filename)

    m.train(tfidf_matrix=tfidf_matrix)

    m.save_matrix(lsa_matrix_filename)

    return m, tfidf_matrix



def test_LsaViewer():

    v = lsaviewer.LsaViewer()

    v.load_corpus(corpus_filename)

    v.load_matrix(lsa_matrix_filename)

    v.tok_name = tok_name

    return v

