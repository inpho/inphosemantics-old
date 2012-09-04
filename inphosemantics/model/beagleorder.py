import multiprocessing as mp

import numpy as np
from numpy import dual


from inphosemantics import model
from inphosemantics.model import beagleenvironment as be



def two_rand_perm(n, seed=None):

    np.random.seed(seed)
    
    perm1 = np.random.permutation(n)

    while True:

        perm2 = np.random.permutation(n)

        if not (perm2 == perm1).all():

            break
        
    return perm1, perm2



def rand_pt_unit_sphere(n):

    pt = np.random.random(n)

    pt = pt * 2 - 1

    pt = pt / np.dot(pt, pt)**.5

    return pt



def mk_b_conv(n, rand_perm=None):

    if rand_perm is None:

        rand_perm = two_rand_perm(n)
    
    def b_conv(v1, v2):
        
        w1 = dual.fft(v1[rand_perm[0]])

        w2 = dual.fft(v2[rand_perm[1]])

        return np.real_if_close(dual.ifft(w1 * w2))

    return b_conv



def naive_cconv(v, w):

    out = np.empty_like(v)

    for i in xrange(v.shape[0]):

        out[i] = np.dot(v, np.roll(w[::-1], i + 1))

    return out



def ngram_slices(i, n, l):
    """
    Given index i, n-gram width n and array length l, returns slices
    for all n-grams containing an ith element
    """
    out = []

    a = i - n + 1

    if a < 0:

        a = 0

    b = i + 1

    if b + n > l:

        b = l - n + 1

    d = b - a

    for k in xrange(d):

        start = a + k

        stop = start + n
            
        out.append(slice(start, stop))

    return out



def reduce_ngrams(fn, a, n, i, flat=True):
    """
    Given array a, reduce with fn all n-grams with width n and less and
    that contain the element at i.

    Memoizes.
    """
    m = n if n < a.shape[0] else a.shape[0]

    out = { 1: { i: a[i] } }
    
    for j in xrange(2, m + 1):

        slices = ngram_slices(i, j, a.shape[0])

        init = slices[0]

        out[j] = { init.start: reduce(fn, a[init]) }

        for s in slices[1:]:

            prev = out[j - 1][s.start]

            out[j][s.start] = fn(prev, a[s.stop - 1])

    # Delete the 1-gram

    del out[1]

    if flat:

        out = [v for d in out.values() for v in d.values()]
    
    return out



class BeagleOrderSingle(model.Model):

    def train(self,
              corpus,
              env_matrix=None,
              psi=None,
              rand_perm=None,
              n_columns=2048,
              lmda = 7,
              tok_name='sentences'):
        
        if env_matrix is None:
            
            m = be.BeagleEnvironment()

            m.train(corpus, n_columns=n_columns)

            env_matrix = m.matrix[:, :]

        else:

            #TODO: Catch if user has a mismatch here

            n_columns = env_matrix.shape[1]

        b_conv = mk_b_conv(n_columns, rand_perm)

        if psi is None:

            psi = rand_pt_unit_sphere(n_columns)

        self.matrix = np.zeros_like(env_matrix)

        sents = corpus.view_tokens(tok_name)

        for sent in sents:

            for i in xrange(sent.shape[0]):

                left = [env_matrix[term] for term in sent[:i]]

                right = [env_matrix[term] for term in sent[i+1:]]

                sent_vecs = np.array(left + [psi] + right)

                conv_ngrams = reduce_ngrams(b_conv, sent_vecs, lmda, i)

                ord_vec = np.sum(conv_ngrams, axis=0)

                self.matrix[sent[i], :] += ord_vec

                

class BeagleOrderMulti(model.Model):

    def train(self,
              corpus,
              env_matrix=None,
              psi=None,
              rand_perm=None,
              n_columns=2048,
              lmda = 7,
              tok_name='sentences'):
        
        if env_matrix is None:

            m = be.BeagleEnvironment()

            m.train(corpus, n_columns=n_columns)

            env_matrix = m.matrix[:]

        else:

            #TODO: Catch if user has a mismatch here

            n_columns = env_matrix.shape[1]

        b_conv = mk_b_conv(n_columns, rand_perm)

        if psi is None:

            psi = rand_pt_unit_sphere(n_columns)

        self.matrix = np.zeros_like(env_matrix)

        print 'Mapping'

        sents = corpus.view_tokens(tok_name)

        mpfn.env_matrix = env_matrix[:]

        mpfn.psi = psi[:]

        mpfn.b_conv = b_conv

        mpfn.lmda = lmda

        # For debugging
        # results = map(mpfn, sents)

        p = mp.Pool()

        results = p.map(mpfn, sents, 1000)

        p.close()

        print 'Reducing'

        for result in results:

            for term, vec in result.iteritems():

                self.matrix[term, :] += vec



def mpfn(sent):

    result = dict()

    for i,t in enumerate(sent):
        
        left = [mpfn.env_matrix[term] for term in sent[:i]]
        
        right = [mpfn.env_matrix[term] for term in sent[i+1:]]
        
        sent_vecs = np.array(left + [mpfn.psi] + right)
        
        conv_ngrams = reduce_ngrams(mpfn.b_conv, sent_vecs, mpfn.lmda, i)
        
        ord_vec = np.sum(conv_ngrams, axis=0)

        if t in result:
            
            result[t] += ord_vec

        else:

            result[t] = ord_vec

    return result



class BeagleOrder(BeagleOrderMulti):

    pass



#
# Tests
#



def test_BeagleOrderMulti():

    from inphosemantics import corpus

    n = 256

    c = corpus.random_corpus(1e5, 1e2, 1, 10, tok_name='sentences')

    m = BeagleOrderMulti()

    m.train(c, n_columns=n)

    return m.matrix



def test_BeagleOrderSingle():

    from inphosemantics import corpus

    n = 256

    c = corpus.random_corpus(1e5, 5e3, 1, 20, tok_name='sentences')

    m = BeagleOrderSingle()

    m.train(c, n_columns=n)

    return m.matrix


def test_compare():

    from inphosemantics import corpus

    n = 4

    c = corpus.random_corpus(1e3, 20, 1, 10, tok_name='sentences')

    em = be.BeagleEnvironment()

    em.train(c, n_columns=n)

    env_matrix = em.matrix

    psi = rand_pt_unit_sphere(n)

    rand_perm = two_rand_perm(n)

    print 'Training single processor model'

    sm = BeagleOrderSingle()

    sm.train(c, psi=psi, env_matrix=env_matrix, rand_perm=rand_perm)

    print 'Training multiprocessor model'

    mm = BeagleOrderMulti()

    mm.train(c, psi=psi, env_matrix=env_matrix, rand_perm=rand_perm)

    print np.allclose(sm.matrix, mm.matrix)

    return sm, mm
    



def test10():

    a = np.arange(5)

    def fn(x,y):

        if isinstance(x, tuple):

            return x + (y,)

        return (x, y)

    import pprint

    print 'array length', a.shape[0]
        
    for i in xrange(a.shape[0]):

        n = 3

        print 'ngram length', n

        print 'index', i

        pprint.pprint(reduce_ngrams(fn, a, n, i))

    for i in xrange(a.shape[0]):

        n = 4

        print 'ngram length', n

        print 'index', i

        pprint.pprint(reduce_ngrams(fn, a, n, i))

    for i in xrange(a.shape[0]):
                
        n = 5
            
        print 'ngram length', n

        print 'index', i
        
        pprint.pprint(reduce_ngrams(fn, a, n, i))



def test11():

    a = np.arange(5)

    def fn(x,y):

        return x + y

    import pprint

    print 'array length', a.shape[0]
        
    for i in xrange(a.shape[0]):

        n = 3

        print 'ngram length', n

        print 'index', i

        pprint.pprint(reduce_ngrams(fn, a, n, i))

    for i in xrange(a.shape[0]):

        n = 4

        print 'ngram length', n

        print 'index', i

        pprint.pprint(reduce_ngrams(fn, a, n, i))

    for i in xrange(a.shape[0]):
                
        n = 5
            
        print 'ngram length', n

        print 'index', i
        
        pprint.pprint(reduce_ngrams(fn, a, n, i))

