from inphosemantics.model import Model

from scipy.sparse import lil_matrix


# TODO: Write a parallel algorithm for this (too slow)

class TfModel(Model):
    """
    """
    def train(self, corpus, token_type, stoplist=None):
        """
        stoplist is ignored in training this type of model.
        """
        tokens = corpus.view_tokens(token_type)
        shape = (len(corpus.term_types), len(tokens))

        self.matrix = lil_matrix(shape)
        
        for j,token in enumerate(tokens):
            for term in token:
                self.matrix[term,j] += 1


    def cf(self, term):
        pass
    
    def cfs(self):
        """
        """
        pass





    
def test_TfModel():

    from inphosemantics import load_picklez, dump_matrix

    corpus_filename =\
        'test-data/iep/selected/corpus/iep-plato.pickle.bz2'
    matrix_filename =\
        'test-data/iep/selected/models/iep-plato-tf-word-article.npy'
    document_type = 'articles'

    corpus = load_picklez(corpus_filename)

    model = TfModel()

    model.train(corpus, 'articles')

    model.dump_matrix(matrix_filename)

    model.load_matrix(matrix_filename)

    return corpus, model, document_type
