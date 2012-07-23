import numpy as np

from inphosemantics import corpus as corp
from inphosemantics import model as mod
from inphosemantics.model import similarity





class Viewer(object):

    def __init__(self,
                 corpus=None,
                 corpus_filename=None,
                 corpus_masked=None,
                 model=None,
                 matrix=None,
                 matrix_filename=None,
                 tok_name=None):


        if corpus:
            if corpus_filename:
                raise Exception("Both a corpus and a "
                                "corpus filename were given.")
            self.corpus = corpus

        elif corpus_filename:
            
            if corpus_masked:

                self.corpus = corp.MaskedCorpus.load(corpus_filename)

                self.corpus = self.corpus.compressed_corpus()


            elif corpus_masked == False:

                self.corpus = corp.Corpus.load(corpus_filename)

            else:

                raise Exception("Whether or not the corpus is masked "
                                "needs to be specified.")


        else:
            raise Exception("Neither a corpus nor a "
                            "corpus filename was given.")
            



        if model:
            if matrix:
                raise Exception("Both a model and a "
                                "matrix were given.")
            elif matrix_filename:
                raise Exception("Both a model and a "
                                "matrix filename were given.")
            elif model_type:
                raise Exception("Both a model and a "
                                "model type were given.")
            else:
                self.matrix = model.matrix


        elif matrix:
            if matrix_filename:
                raise Exception("Both a matrix and a "
                                "matrix filename were given.")
            else:
                self.matrix = matrix


        elif matrix_filename:
            
            self.matrix = mod.Model.load(matrix_filename)

        else:
            raise Exception("Neither a model, matrix nor "
                            "matrix filename were given.")



        self.tok_name = tok_name





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
