import os
import shutil
import tempfile
from multiprocessing import Pool

import numpy as np

from inphosemantics.model import Model
from inphosemantics.model.matrix\
     import DenseMatrix, SparseMatrix, load_matrix
from inphosemantics.model.beagleenvironment import BeagleEnvironment


def context_fn(ind_sent_list):

    index = ind_sent_list[0]
    sent_list = ind_sent_list[1]

    stoplist = context_fn.stoplist
    env_matrix = context_fn.env_matrix
    n_columns = context_fn.n_columns
    temp_dir = context_fn.temp_dir

    mem_matrix = SparseMatrix(env_matrix.shape)

    print 'Training on chunk of sentences', index

    for sent in sent_list:
        for i,word in enumerate(sent):
            context = np.delete(sent, i)
            for ctxword in context:
                if ctxword not in stoplist:
                    mem_matrix[word,:] += env_matrix[ctxword,:]
                    
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


        if env_matrix:
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

        # print 'Length of sentence list', len(sentences)
        # print 'Sentences in partitions '\
        #       sum([len(l) for l in sent_lists])

        # results = map(context_fn, ind_sent_lists)

        p = Pool()
        results = p.map(context_fn, ind_sent_lists, 1)
        p.close()


        # Reduce
        mem_matrix = DenseMatrix(np.zeros(env_matrix.shape))
        
        for result in results:

            print 'Reducing', result

            summand = load_matrix(result)
            mem_matrix += summand
            

        # Clean up
        print 'Deleting temporary directory\n'\
              '  ', temp_dir

        shutil.rmtree(temp_dir)

        
        self.matrix = mem_matrix

        

def test_BeagleContext():

    from inphosemantics import load_picklez

    corpus_filename =\
        'test-data/iep/selected/corpus/iep-selected.pickle.bz2'

    print 'Loading corpus\n'\
          '  ', corpus_filename
    
    c = load_picklez(corpus_filename)

    m = BeagleContext()

    m.train(c, n_columns=256)

    return

