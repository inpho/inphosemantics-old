import numpy as np

from inphosemantics import load_picklez


#TODO: Add term_types parameter and instantiate an empty corpus object
#with it. This will speed up many viewing tasks.

class Viewer(object):

    def __init__(self,
                 corpus=None,
                 corpus_filename=None,
                 model=None,
                 model_type=None,
                 matrix=None,
                 matrix_filename=None,
                 token_type=None,
                 stoplist=None):

        if corpus:
            if corpus_filename:
                raise Exception("Both a corpus and a "
                                "corpus filename were given.")
            self.corpus = corpus

        elif corpus_filename:
            
            print 'Loading corpus from\n'\
                  '  ', corpus_filename
            self.corpus = load_picklez(corpus_filename)

        else:
            raise Exception("Neither a corpus nor a "
                            "corpus filename were given.")
            

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
                self.model = model


        elif matrix:
            if matrix_filename:
                raise Exception("Both a matrix and a "
                                "matrix filename were given.")
            else:
                self.model = model_type(matrix)


        elif matrix_filename:
            
            self.model = model_type()
            print 'Loading matrix from\n'\
                  '  ', matrix_filename
            self.model.load_matrix(matrix_filename)

        else:
            raise Exception("Neither a model, matrix nor "
                            "matrix filename were given.")

        self.token_type = token_type

        if stoplist:
            self.stoplist = self.corpus.encode_tokens_str(stoplist)
            print 'Applying stoplist to matrix'
            self.model.filter_rows(self.stoplist)




def simmat_terms(viewer, term_list):

    terms = viewer.corpus.term_types_str

    terms_rev = dict(zip(terms, xrange(len(terms))))

    indices = [terms_rev[term] for term in term_list]

    simmat = viewer.model.simmat_rows(indices)

    simmat.indices = term_list

    return simmat



def simmat_documents(viewer, document_list):

    doc_names = viewer.corpus.tokens_meta[viewer.token_type]

    doc_names_rev = dict((k,v) for k,v in doc_names.iteritems())

    indices = [doc_names_rev[doc] for doc in document_list]
    
    simmat = viewer.model.simmat_columns(indices)

    simmat.indices = document_list

    return simmat



def similar_terms(viewer, term, filter_nan=False):

    i = viewer.corpus.term_types_str.index(term)
    
    cosines = viewer.model.similar_rows(i, filter_nan=filter_nan)
    
    return [(viewer.corpus.term_types_str[t], v)
            for t,v in cosines]



def similar_documents(viewer, document, filter_nan=False):
    
    doc_names = viewer.corpus.tokens_meta[viewer.token_type]
    doc_names_rev = dict((k,v) for k,v in doc_names.iteritems())
    
    i = doc_names_rev[document]
    
    cosines = viewer.model.similar_columns(i, filter_nan=filter_nan)
    
    return [(doc_names[d], v) for d,v in cosines]

    


def test_Viewer():

    from inphosemantics.model.tf import TfModel

    corpus_filename =\
        'test-data/iep/selected/corpus/iep-selected.pickle.bz2'
    matrix_filename =\
        'test-data/iep/selected/models/iep-selected-tf-word-article.npy'
    
    v = Viewer(corpus_filename=corpus_filename,
                   model_type=TfModel,
                   matrix_filename=matrix_filename,
                   token_type='articles')

    print v

    return v

