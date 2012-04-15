import sys
import numpy as np
from math import sqrt


def norm(v):
    return sqrt(np.dot(v,v))


def normalize(v):
    if np.any(v):
        return v / norm(v) # component-wise division
    else:
        return v # would be division by zero


def vector_cos(v,w):
    '''
    Computes the cosine of the angle between vectors v and w. A result
    of -2 denotes undefined, i.e., when v or w is a zero vector
    '''
    if not v.any() or not w.any():
        return -2
    else:
        return np.dot(v,w) / (norm(v) * norm(w))


class RandomWordGen:
    def __init__(self, d = 1024, seed = None, *args):
        self.d = d
        self.start = np.random.randint(sys.maxint)
    
    def make_rep(self, word):
        np.random.seed(self.start ^ abs(hash(word)))
        return normalize(np.random.randn(self.d))


def init_gen_randvec(dimension):
    gen_randvec.randgen = RandomWordGen(d = dimension)


def gen_randvec(word):
    return gen_randvec.randgen.make_rep(word)
