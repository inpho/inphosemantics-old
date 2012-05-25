from inphosemantics.viewer import Viewer
from inphosemantics.viewer import similar_terms as _similar_terms
from inphosemantics.model.beagleenvironment import BeagleEnvironment


class BeagleEnvironmentViewer(Viewer):

    def __init__(self,
                 corpus=None,
                 corpus_filename=None, 
                 model=None,
                 matrix=None,
                 matrix_filename=None,
                 document_type=None,
                 stoplist=None):

        if matrix or matrix_filename:
            super(BeagleEnvironmentViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          model_type=BeagleEnvironment,
                          matrix=matrix,
                          matrix_filename=matrix_filename,
                          document_type=document_type,
                          stoplist=stoplist)
        else:
            super(BeagleEnvironmentViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          matrix_filename=matrix_filename,
                          document_type=document_type,
                          stoplist=stoplist)


    def similar_terms(self, term, filter_nan=False):

        return _similar_terms(self, term, filter_nan)



def test_BeagleEnvironmentViewer():

    root = 'test-data/iep/plato/'

    corpus_filename =\
        root + 'corpus/iep-plato.pickle.bz2'

    matrix_filename =\
        root + 'models/iep-plato-beagleenviroment-sentences.mtx.bz2'

    v = BeagleEnvironmentViewer(corpus_filename=corpus_filename,
                                matrix_filename=matrix_filename)

    return v
