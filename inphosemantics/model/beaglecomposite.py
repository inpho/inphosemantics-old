import numpy as np

from inphosemantics.model import Model
from inphosemantics.model.beagleenvironment import BeagleEnvironment
from inphosemantics.model.beaglecontext import BeagleContext
from inphosemantics.model.beagleorder import BeagleOrder



class BeagleComposite(Model):

    def train(self,
              corpus,
              token_type='sentences',
              stoplist=list(),
              n_columns=None,
              env_matrix=None,
              ctx_matrix=None,
              ord_matrix=None):


        if ctx_matrix == None or ord_matrix == None:

            if env_matrix == None:

                _env_matrix == None
                
                env_model = BeagleEnvironment()
                env_model.train(corpus,
                                token_type=token_type,
                                stoplist=stoplist,
                                env_matrix=env_matrix)

                env_matrix = env_model.matrix


            if ctx_matrix == None:
                ctx_model = BeagleContext()
                ctx_model.train(corpus,
                                token_type=token_type,
                                stoplist=stoplist,
                                env_matrix=env_matrix)

                ctx_matrix = ctx_model.matrix

            else:
                if _env_matrix == None:
                    print 'Warning: Context and Order models '\
                          'trained with different Environment models.'


            if ord_matrix == None:

                ord_model = BeagleOrder()
                ord_model.train(corpus,
                                token_type=token_type,
                                stoplist=stoplist,
                                env_matrix=env_matrix)

                ord_matrix = ord_model.matrix

            else:
                if _env_matrix == None:
                    print 'Warning: Context and Order models '\
                          'trained with different Environment models.'

        
        self.matrix = ctx_matrix
        self.matrix += ord_matrix
        


        

def test_BeagleComposite_1():

    import numpy as np
    from inphosemantics.corpus import BaseCorpus

    env_matrix = np.array([[2.,2.],[3.,3.]])

    c = BaseCorpus([0, 1, 1, 0, 1, 0, 0, 1, 1],
                   {'sentences': [2, 5]})

    m = BeagleComposite()
    m.train(c, env_matrix=env_matrix)

    print c.view_tokens('sentences')
    print m.matrix

    

def test_BeagleComposite_2():

    from inphosemantics import load_picklez, dump_matrix

    root = 'test-data/iep/plato/'

    corpus_filename =\
        root + 'corpus/iep-plato.pickle.bz2'

    env_filename =\
        root + 'models/iep-plato-beagleenviroment-sentences.npy'

    matrix_filename =\
        root + 'models/iep-plato-beaglecomposite-sentences.npy'


    print 'Loading corpus\n'\
          '  ', corpus_filename
    c = load_picklez(corpus_filename)
    

    print 'Loading environment model\n'\
          '  ', env_filename
    e = BeagleEnvironment()
    e.load_matrix(env_filename)
    print e.matrix

    print 'Training model'
    m = BeagleComposite()
    m.train(c, env_matrix=e.matrix)
    print m.matrix


    print 'Dumping matrix to\n'\
          '  ', matrix_filename
    m.dump_matrix(matrix_filename)
    
    return m
