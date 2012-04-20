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


class RandomVectors(object):

    def __init__(self, dimension):
        self.dimension = dimension
        self.start = np.random.randint(sys.maxint)
    
    def meaning_vector(self, word):

        np.random.seed(self.start ^ abs(hash(word)))
        
        vector = normalize(np.random.randn(self.dimension))
        
        return vector


class RandomPermutations(object):
    

    def __init__(self, dimension, n):

        np.random.seed()
        
        self.permutations = dict()

        for i in xrange(n):
            
            idx_array = np.random.permutation(dimension)
            
            self.permutations[i+1] = self.mk_permutation(idx_array)


    @staticmethod
    def mk_permutation(idx_array):
        
        def p(vector):
            return vector[idx_array]

        return p
        
    
