from inphosemantics import *
from ...corpus import *
from ...beagle import *
from inphosemantics.viewers import *
import sys


read_vecs = mk_read_vecs(__path__[0])
write_vecs = mk_write_vecs(__path__[0])

gen_cosines = mk_gen_cosines(__path__[0])

similar = mk_similar(lexicon, stopwords, __path__[0])
display_similar = mk_display_similar(lexicon, stopwords, __path__[0])


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
