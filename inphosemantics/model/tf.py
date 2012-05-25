from inphosemantics.model import Model

from scipy.sparse import lil_matrix



class TfModel(Model):
    """
    """
    def __init__(self, matrix=None):

        super(TfModel, self).__init__(matrix)


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

    from inphosemantics import load_picklez, dump_matrixz

    corpus_filename =\
        'test-data/iep/selected/corpus/iep-selected.pickle.bz2'
    matrix_filename =\
        'test-data/iep/selected/models/iep-selected-tf-word-article.mtx.bz2'
    document_type = 'articles'

    corpus = load_picklez(corpus_filename)

    model = TfModel()

    model.train(corpus, 'articles')

    dump_matrixz(model, matrix_filename)

    model.load_matrix(matrix_filename)

    return corpus, model, document_type
