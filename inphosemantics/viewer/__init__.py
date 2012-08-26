import numpy as np

from inphosemantics import corpus as cps
from inphosemantics import model
import similarity


class Viewer(object):
    """

    
    Notes
    -----
    Assume that the incoming corpus does not have a mask (i.e., has
    already been compressed if so).

    """
    def __init__(self,
                 corpus=None,
                 matrix=None,
                 tok_name=None):

        self.corpus = corpus
        self.matrix = matrix
        self.tok_name = tok_name


    def load_matrix(self, filename):

        self.matrix = model.Model.load_matrix(filename)


    def load_corpus(self, filename):

        self.corpus = cps.Corpus.load(filename)


    


def simmat_terms(viewer, term_list):
    """
    """

    terms_int = viewer.corpus.terms_int

    ## Keep only the important indices.
    indices = [terms_int[term] for term in term_list]

    ## Create a similarity matrix.
    simmat = similarity.simmat_rows(indices, viewer.matrix)

    ## Replace integer indices with string indices.
    simmat.indices = term_list
    
    return simmat




def simmat_documents(viewer, document_list):

    doc_names = viewer.corpus.view_metadata[viewer.tok_name]

    doc_names_rev = dict((k,v) for k,v in doc_names.iteritems())

    indices = [doc_names_rev[doc] for doc in document_list]
    
    simmat = similarity.simmat_columns(indices, viewer.matrix)

    simmat.indices = document_list

    return simmat



def similar_terms(viewer, term, filter_nan=False):


    i = viewer.corpus.terms_int[term]
    
    cosines = similarity.similar_rows(i, viewer.matrix,
                                      filter_nan=filter_nan)
    
    return [(viewer.corpus.terms[t], v) for t,v in cosines]



def similar_documents(viewer, document, filter_nan=False):
    
    doc_names = viewer.corpus.view_metadata[viewer.tok_name]
    
    doc_names_rev = dict((k,v) for k,v in doc_names.iteritems())
    
    i = doc_names_rev[document]
    
    cosines = similarity.similar_columns(i, viewer.matrix,
                                         filter_nan=filter_nan)
    
    return [(doc_names[d], v) for d,v in cosines]
