import os
import shutil
import tempfile
from multiprocessing import Pool

import numpy as np

from inphosemantics import load_matrix
from inphosemantics.model import Model
from inphosemantics.model.beagleenvironment import BeagleEnvironment


# TODO: This seems much slower than it should be.

def context_fn(ind_sent_list):

    index = ind_sent_list[0]
    sent_list = ind_sent_list[1]

    stoplist = context_fn.stoplist
    env_matrix = context_fn.env_matrix
    n_columns = context_fn.n_columns
    temp_dir = context_fn.temp_dir

    mem_matrix = lil_matrix(env_matrix.shape)

    print 'Training on chunk of sentences', index

    for sent in sent_list:
        for i,word in enumerate(sent):
            context = np.delete(sent, i)
            for ctxword in context:
                if ctxword not in stoplist:
                    mem_matrix[word,:] += env_matrix[ctxword,:]
                    print mem_matrix

    print 'Chunk of sentences', index, '\n', mem_matrix
                    
    tmp_file =\
        os.path.join(temp_dir, 'context-' + str(index) + '.tmp')

    print 'Dumping to temp file\n'\
          '  ', tmp_file
    
    mem_matrix.dump(tmp_file)
    
    return tmp_file


class BeagleContext(Model):

    def train(self,
              corpus,
              token_type='sentences',
              stoplist=list(),
              n_columns=None,
              env_matrix=None,
              n_cores=20):

        context_fn.stoplist = stoplist
        context_fn.n_columns = n_columns


        if env_matrix != None:
            n_columns = env_matrix.shape[1]
        else:
            env_model = BeagleEnvironment()
            env_model.train(corpus,
                            token_type,
                            stoplist,
                            n_columns)
            env_matrix = env_model.matrix

        context_fn.env_matrix = env_matrix
        
        temp_dir = tempfile.mkdtemp()
        context_fn.temp_dir = temp_dir


        sentences = corpus.view_tokens(token_type)
        m = len(sentences) / n_cores
        sent_lists = [sentences[i*m:(i+1)*m]
                      for i in xrange(n_cores)]
        sent_lists.append(sentences[m*n_cores:])
        
        ind_sent_lists = list(enumerate(sent_lists))


        p = Pool()
        results = p.map(context_fn, ind_sent_lists, 1)
        p.close()


        # Reduce
        self.matrix = np.zeros(env_matrix.shape)
        
        for result in results:

            print 'Reducing', result

            summand = load_matrix(result)
            self.matrix += summand




        # Clean up
        print 'Deleting temporary directory\n'\
              '  ', temp_dir

        shutil.rmtree(temp_dir)

        

def test_BeagleContext_1():

    import numpy as np
    from inphosemantics.corpus import BaseCorpus

    env_matrix = np.matrix([[2,2],[3,3]])

    c = BaseCorpus([0, 1, 1, 0, 1, 0, 0, 1, 1],
                   {'sentences': [2, 5]})

    m = BeagleContext()
    m.train(c, env_matrix=env_matrix)

    print c.view_tokens('sentences')
    print m.matrix

    

def test_BeagleContext_2():

    from inphosemantics import load_picklez, dump_matrixz

    root = 'test-data/iep/plato/'

    corpus_filename =\
        root + 'corpus/iep-plato.pickle.bz2'

    env_filename =\
        root + 'models/iep-plato-beagleenviroment-sentences.mtx.bz2'

    matrix_filename =\
        root + 'models/iep-plato-beaglecontext-sentences.mtx.bz2'


    print 'Loading corpus\n'\
          '  ', corpus_filename
    c = load_picklez(corpus_filename)
    

    print 'Loading environment model\n'\
          '  ', env_filename
    e = BeagleEnvironment()
    e.load_matrix(env_filename)
    print e.matrix

    print 'Training model'
    m = BeagleContext()
    m.train(c, env_matrix=e.matrix)
    print m.matrix


    print 'Dumping matrix to\n'\
          '  ', matrix_filename
    dump_matrixz(m, matrix_filename)
    
    return m
