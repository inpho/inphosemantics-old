from inphosemantics import *
from inphosemantics.model import Model
from inphosemantics.model.matrix import SparseMatrix
from inphosemantics.model.viewer import Viewer



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

        self.matrix = SparseMatrix(shape)
        
        for j,token in enumerate(tokens):
            for term in token:
                self.matrix[term,j] += 1


    def cf(self, term):
        pass
    
    def cfs(self):
        """
        """
        pass



class TfViewer(Viewer):

    def __init__(self,
                 corpus=None,
                 corpus_filename=None, 
                 model=None,
                 matrix=None,
                 matrix_filename=None,
                 document_type=None,
                 stoplist=None):

        if matrix or matrix_filename:
            super(TfViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          model_type=TfModel,
                          matrix=matrix,
                          matrix_filename=matrix_filename,
                          document_type=document_type,
                          stoplist=stoplist)
        else:
            super(TfViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          matrix_filename=matrix_filename,
                          document_type=document_type,
                          stoplist=stoplist)


    def cf(self, term):
        pass

    def cfs(self):
        pass


    
def test_TfModel():

    corpus_filename =\
        'test-data/iep/selected/corpus/iep-selected.pickle.bz2'
    matrix_filename =\
        'test-data/iep/selected/models/iep-selected-tf-word-article.mtx.bz2'
    document_type = 'articles'

    corpus = load_picklez(corpus_filename)

    model = TfModel()

    model.train(corpus, 'articles')

    model.dumpz(matrix_filename)

    model.load_matrix(matrix_filename)

    return corpus, model, document_type
