from inphosemantics.viewer import Viewer
from inphosemantics.model.tfidf import TfIdfModel


class TfIdfViewer(Viewer):

    def __init__(self,
                 corpus=None,
                 corpus_filename=None, 
                 model=None,
                 matrix=None,
                 matrix_filename=None,
                 document_type=None,
                 stoplist=None):

        if matrix or matrix_filename:
            super(TfIdfViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          model_type=TfIdfModel,
                          matrix=matrix,
                          matrix_filename=matrix_filename,
                          document_type=document_type,
                          stoplist=stoplist)
        else:
            super(TfIdfViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          matrix_filename=matrix_filename,
                          document_type=document_type,
                          stoplist=stoplist)


    def idf(self, term):
        pass

    def idfs(self):
        pass
