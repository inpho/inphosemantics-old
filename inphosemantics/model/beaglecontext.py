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
              tok_name='sentences'):
        
        if env_matrix is None:
            
            m = be.BeagleEnvironment()
            
            m.train(corpus, n_columns=n_columns)
            
            env_matrix = m.matrix[:, :]

        fn.env_matrix = env_matrix[:, :]
            
        print 'Collecting corpus data'
            
        sents = corpus.view_tokens(tok_name)
        
        n = 5000
        
        sents_iter = (sents[i:i+n] for i in xrange(0, len(sents), n))
        
        data = list(sents_iter)

        print 'Mapping work to processors'
        
        p = mp.Pool()
        
        results = p.map(mp_fn, data, 4)
        
        p.close()

        print 'Reducing'
        
        self.matrix = np.zeros_like(env_matrix)
            
        for ctx_vecs in results:
            
            for term, ctx_vec in ctx_vecs.iteritems():
                
                self.matrix[term, :] += ctx_vec
                
                
                
def mp_fn(sents):

    env_matrix = fn.env_matrix[:, :]

    print 'Working on new sentence list'
                    
    ctx_vecs = dict()

    for sent in sents:
        
        for i,term in enumerate(sent):

            left_ctx = sent[:i]
            
            right_ctx = sent[i+1:]

            ctx_vector = np.sum(env_matrix[left_ctx], axis=0)

            ctx_vector += np.sum(env_matrix[right_ctx], axis=0)

            if term in ctx_vecs:

                ctx_vecs[term] += ctx_vector

            else:

                ctx_vecs[term] = ctx_vector

    return ctx_vecs



class BeagleContext(BeagleContextSingle):

    pass
