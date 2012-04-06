import sys


class RandomWordGen:
    def __init__(self, d = 1024, seed = None, *args):
        self.d = d
        self.start = np.random.randint(sys.maxint)
    
    def make_rep(self, word):
        np.random.seed(self.start ^ abs(hash(word)))
        return normalize(np.random.randn(self.d))

def init_gen_randvec():
    gen_randvec.randgen = RandomWordGen(d = dim)    

def gen_randvec(word):
    return gen_randvec.randgen.make_rep(word)

def gen_vecs():
    init_gen_randvec()

    p = Pool()
    vecs = p.map(gen_randvec, lexicon, 1000)
    p.close()

    print 'Environment vectors computed.'

    vecs = np.array(vecs)
    write_vecs(vecs)
    return
