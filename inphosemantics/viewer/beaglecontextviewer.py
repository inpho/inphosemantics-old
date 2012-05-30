import inphosemantics.viewer as vw
import inphosemantics.model.beaglecontext as bc


class BeagleContextViewer(vw.Viewer):

    def __init__(self,
                 corpus=None,
                 corpus_filename=None, 
                 model=None,
                 matrix=None,
                 matrix_filename=None,
                 token_type=None,
                 stoplist=None):

        if matrix or matrix_filename:
            super(BeagleContextViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          model_type=bc.BeagleContext,
                          matrix=matrix,
                          matrix_filename=matrix_filename,
                          token_type=token_type,
                          stoplist=stoplist)
        else:
            super(BeagleContextViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          matrix_filename=matrix_filename,
                          token_type=token_type,
                          stoplist=stoplist)


    def similar_terms(self, term, filter_nan=False):

        return vw.similar_terms(self, term, filter_nan)


    def simmat_terms(self, term_list):

        return vw.simmat_terms(self, term_list)


def test_BeagleContextViewer():

    root = 'test-data/iep/plato/'

    corpus_filename =\
        root + 'corpus/iep-plato.pickle.bz2'

    matrix_filename =\
        root + 'models/iep-plato-beaglecontext-sentences.npy'

    v = BeagleContextViewer(corpus_filename=corpus_filename,
                            matrix_filename=matrix_filename)

    return v
