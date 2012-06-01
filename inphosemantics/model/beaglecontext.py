import os
import shutil
import tempfile
import multiprocessing as mp


import numpy as np
from scipy.sparse import lil_matrix

from inphosemantics import load_matrix, dump_matrix
from inphosemantics.model import Model
from inphosemantics.model.beagleenvironment import BeagleEnvironment



def context_fn(ind_sent_list):

    index = ind_sent_list[0]
    sent_list = ind_sent_list[1]

    env_matrix = context_fn.env_matrix

    temp_dir = context_fn.temp_dir

    # Very slow in SciPy v. 0.7.2. Same for dok_matrix.
    # mem_matrix = lil_matrix(env_matrix.shape)

    # Occupies too much memory
    # mem_matrix = np.zeros(env_matrix.shape, dtype=np.float32)

    # So, using dictionary as temporary sparse matrix
    mem_matrix = dict()


    print 'Training on chunk of sentences', index

    for sent in sent_list:

        for i,word in enumerate(sent):

            if word not in mem_matrix:
                mem_matrix[word] = np.zeros(env_matrix.shape[1], dtype=np.float32)

            # Left context
            for ctxword in sent[:i]:

                mem_matrix[word] += env_matrix[ctxword,:]

                # mem_matrix[word,:] += env_matrix[ctxword,:]

            # Right context
            for ctxword in sent[i+1:]:

                mem_matrix[word] += env_matrix[ctxword,:]
                
                # mem_matrix[word,:] += env_matrix[ctxword,:]


    print 'Chunk of sentences', index
    # print mem_matrix
                    
    tmp_file =\
        os.path.join(temp_dir, 'context-' + str(index) + '.tmp.npy')

    print 'Dumping to temp file\n'\
          '  ', tmp_file
    
    dump_matrix(mem_matrix, tmp_file)
    
    return tmp_file



class BeagleContext(Model):

    def train(self,
              corpus,
              token_type='sentences',
              stoplist=list(),
              n_columns=None,
              env_matrix=None):


        if env_matrix == None:
            env_model = BeagleEnvironment()
            env_model.train(corpus,
                            token_type,
                            stoplist,
                            n_columns)
        else:
            env_model = BeagleEnvironment(env_matrix)

        #Apply stoplist to environment matrix
        env_model.filter_rows(stoplist)


        __shape = env_model.matrix.shape


        context_fn.env_matrix = env_model.matrix

        del env_model
        del env_matrix


        
        temp_dir = tempfile.mkdtemp()
        context_fn.temp_dir = temp_dir


        sentences = corpus.view_tokens(token_type)
        
        # number of sentences in a chunk of sentences
        n = 500

        sent_lists = np.split(np.asarray(sentences, dtype=np.object_),
                              np.arange(n, len(sentences), n))

        ind_sent_lists = list(enumerate(sent_lists))


        # Map
        p = mp.Pool()
        results = p.map(context_fn, ind_sent_lists, 1)
        p.close()


        del context_fn.env_matrix


        # Reduce
        self.matrix = np.zeros(__shape, dtype=np.float32)
        
        for result in results:

            print 'Reducing', result

            summand = load_matrix(result)
            # self.matrix += summand

            for i,row in summand.iteritems():
                self.matrix[i,:] += row

        # Clean up
        print 'Deleting temporary directory\n'\
              '  ', temp_dir

        shutil.rmtree(temp_dir)

        

def test_BeagleContext_1():

    import numpy as np
    from inphosemantics.corpus import BaseCorpus

    env_matrix = np.array([[2.,2.],[3.,3.]])

    c = BaseCorpus([0, 1, 1, 0, 1, 0, 0, 1, 1],
                   {'sentences': [2, 5]})

    m = BeagleContext()
    m.train(c, env_matrix=env_matrix)

    print c.view_tokens('sentences')
    print m.matrix

    

def test_BeagleContext_2():

    from inphosemantics import load_picklez, dump_matrix

    root = 'test-data/iep/plato/'

    corpus_filename =\
        root + 'corpus/iep-plato.pickle.bz2'

    env_filename =\
        root + 'models/iep-plato-beagleenviroment-sentences.npy'

    matrix_filename =\
        root + 'models/iep-plato-beaglecontext-sentences.npy'


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
    m.dump_matrix(matrix_filename)
    
    return m
