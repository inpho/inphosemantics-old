import inphosemantics.viewer as vw
import inphosemantics.model.tfidf as tfidf



class TfIdfViewer(vw.Viewer):

    def __init__(self,
                 corpus=None,
                 corpus_filename=None, 
                 model=None,
                 matrix=None,
                 matrix_filename=None,
                 token_type=None,
                 stoplist=None):

        if matrix or matrix_filename:
            super(TfIdfViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          model_type=tfidf.TfIdfModel,
                          matrix=matrix,
                          matrix_filename=matrix_filename,
                          token_type=token_type,
                          stoplist=stoplist)
        else:
            super(TfIdfViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          matrix_filename=matrix_filename,
                          token_type=token_type,
                          stoplist=stoplist)


    def similar_terms(self, term, filter_nan=False):

        return vw.similar_terms(self, term, filter_nan)


    def similar_documents(self, document, filter_nan=False):

        return vw.similar_documents(self, document, filter_nan)


    def simmat_terms(self, term_list):

        return vw.simmat_terms(self, term_list)


    def simmat_documents(self, document_list):

        return vw.simmat_documents(self, document_list)


    def idf(self, term):
        pass


    def idfs(self):
        pass
