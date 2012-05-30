import inphosemantics.viewer as vw
import inphosemantics.model.beaglecomposite as bc


class BeagleCompositeViewer(vw.Viewer):

    def __init__(self,
                 corpus=None,
                 corpus_filename=None, 
                 model=None,
                 matrix=None,
                 matrix_filename=None,
                 token_type=None,
                 stoplist=None):

        if matrix or matrix_filename:
            super(BeagleCompositeViewer, self)\
                .__init__(corpus=corpus,
                          corpus_filename=corpus_filename, 
                          model=model,
                          model_type=bc.BeagleComposite,
                          matrix=matrix,
                          matrix_filename=matrix_filename,
                          token_type=token_type,
                          stoplist=stoplist)
        else:
            super(BeagleCompositeViewer, self)\
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



def test_BeagleCompositeViewer():

    root = 'test-data/iep/plato/'

    corpus_filename =\
        root + 'corpus/iep-plato.pickle.bz2'

    matrix_filename =\
        root + 'models/iep-plato-beaglecomposite-sentences.npy'

    v = BeagleCompositeViewer(corpus_filename=corpus_filename,
                              matrix_filename=matrix_filename)

    return v
