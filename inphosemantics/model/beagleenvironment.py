import numpy as np

from inphosemantics.model import Model


class BeagleEnvironment(Model):

    def train(self,
              corpus,
              token_type='sentences',
              stoplist=None,
              n_columns=2048):

        shape = (len(corpus.term_types), n_columns)

        self.matrix = np.random.random(shape)
        self.matrix *= 2
        self.matrix -= 1
        
        norm = np.sum(self.matrix**2, axis=1)**(1./2)

        self.matrix /= norm[:, np.newaxis]



def test_BeagleEnvironment():

    from inphosemantics import load_picklez, dump_matrix

    root = 'test-data/iep/plato/'

    corpus_filename =\
        root + 'corpus/iep-plato.pickle.bz2'

    matrix_filename =\
        root + 'models/iep-plato-beagleenviroment-sentences.npy'


    print 'Loading corpus\n'\
          '  ', corpus_filename
    c = load_picklez(corpus_filename)

    print 'Training model'
    m = BeagleEnvironment()
    m.train(c, n_columns=256)

    print 'Row norms', np.sum(m.matrix**2, axis=1)**(1./2)

    print 'Dumping matrix to\n'\
          '  ', matrix_filename
    m.dump_matrix(matrix_filename)
    
    return m
