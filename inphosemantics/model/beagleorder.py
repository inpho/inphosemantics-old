import numpy as np


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
        
    
