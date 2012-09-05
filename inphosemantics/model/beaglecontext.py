import os
import shutil
import tempfile
import multiprocessing as mp

import numpy as np

from inphosemantics import model
from inphosemantics.model import beagleenvironment as be



class BeagleContextSingle(model.Model):

    def train(self,
              corpus,
              env_matrix=None,
              n_columns=2048,
              tok_name='sentences'):

        if env_matrix is None:

            m = be.BeagleEnvironment()

            m.train(corpus, n_columns=n_columns)

            env_matrix = m.matrix[:, :]

        sents = corpus.view_tokens(tok_name)

        self.matrix = np.zeros_like(env_matrix)

        for sent in sents:

            for i,term in enumerate(sent):

                left_ctx = sent[:i]

                right_ctx = sent[i+1:]

                ctx_vector = np.sum(env_matrix[left_ctx], axis=0)

                ctx_vector += np.sum(env_matrix[right_ctx], axis=0)

                self.matrix[term, :] += ctx_vector



class BeagleContextMulti(model.Model):

    def train(self,
              corpus,
              env_matrix=None,
              n_columns=2048,
              tok_name='sentences',
              n_processes=20):

        if env_matrix is None:

            m = be.BeagleEnvironment()

            m.train(corpus, n_columns=n_columns)

            env_matrix = m.matrix[:]

            del m

        global _shape

        _shape = env_matrix.shape


        
        global _env_matrix

        print 'Copying env matrix to shared mp array'

        _env_matrix = mp.Array('f', env_matrix.size, lock=False)

        _env_matrix[:] = env_matrix.ravel()[:]

        del env_matrix



        print 'Gathering tokens over which to map'

        sent_lists = corpus.view_tokens(tok_name)

        k = len(sent_lists) / (n_processes - 1)
        
        sent_lists_ = [sent_lists[i * k:(i + 1) * k]
                       for i in xrange(n_processes - 1)]

        sent_lists_.append(sent_lists[(i + 1) * k:])

        tmp_dir = tempfile.mkdtemp()
        
        tmp_files = [os.path.join(tmp_dir, 'tmp_' + str(i))
                     for i in xrange(len(sent_lists_))]

        sent_lists = [(sent_lists_[i], tmp_files[i])
                      for i in xrange(len(sent_lists_))]

        del sent_lists_



        # For debugging
        # tmp_files = map(mpfn, sent_lists)

        print 'Forking'

        p = mp.Pool()

        tmp_files = p.map(mpfn, sent_lists, 1)

        p.close()

        print 'Reducing'

        self.matrix = np.zeros(_shape)

        for filename in tmp_files:

            result = np.memmap(filename, mode='r',
                               shape=_shape, dtype=np.float32)

            self.matrix[:, :] += result[:, :]

        print 'Removing', tmp_dir

        shutil.rmtree(tmp_dir)



def mpfn((sents, filename)):

    n = _shape[1]

    result = np.memmap(filename, mode='w+', shape=_shape, dtype=np.float32)

    for sent in sents:
        
        for i,t in enumerate(sent):

            left_ctx = sent[:i]
            
            right_ctx = sent[i+1:]

            ctx_vector = np.zeros(n)

            for i in left_ctx:

                ctx_vector += np.array(_env_matrix[i * n: (i + 1) *n])

            for i in right_ctx:

                ctx_vector += np.array(_env_matrix[i * n: (i + 1) *n])

            result[t] += ctx_vector

    del result
    
    return filename




class BeagleContext(BeagleContextSingle):

    pass



#
# For testing
#



def test_BeagleContextSingle():

    from inphosemantics import corpus

    n = 256

    c = corpus.random_corpus(1e4, 1e2, 1, 20, tok_name='sentences')
    
    m = BeagleContextSingle()

    m.train(c, n_columns=n)

    return m.matrix



def test_BeagleContextMulti():

    from inphosemantics import corpus

    n = 2048

    print 'Generating corpus'
    
    c = corpus.random_corpus(1e7, 1e5, 1, 20, tok_name='sentences')

    m = BeagleContextMulti()

    m.train(c, n_columns=n)

    return m.matrix



def test_compare():

    from inphosemantics import corpus

    n = 4

    c = corpus.random_corpus(1e3, 20, 1, 10, tok_name='sentences')

    em = be.BeagleEnvironment()

    em.train(c, n_columns=n)

    env_matrix = em.matrix

    print 'Training single processor model'

    sm = BeagleContextSingle()

    sm.train(c, env_matrix=env_matrix)

    print 'Training multiprocessor model'

    mm = BeagleContextMulti()

    mm.train(c, env_matrix=env_matrix)

    assert np.allclose(sm.matrix, mm.matrix, atol=1e-07)
