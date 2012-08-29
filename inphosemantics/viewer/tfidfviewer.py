import inphosemantics.viewer as vw



class TfIdfViewer(vw.Viewer):
    """
    """
    def similar_terms(self, term, filter_nan=False):

        return vw.similar_terms(self.corpus, self.matrix,
                                term, filter_nan=filter_nan)



    def similar_documents(self, document, filter_nan=False):

        return vw.similar_documents(self.corpus, self.matrix,
                                    document, filter_nan=filter_nan)



    def simmat_terms(self, term_list):

        return vw.simmat_terms(self.corpus, self.matrix, term_list)



    def simmat_documents(self, document_list):

        return vw.simmat_documents(self.corpus, self.matrix, document_list)

